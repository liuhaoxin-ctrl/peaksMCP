import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { INotebookTracker, NotebookActions, NotebookPanel } from '@jupyterlab/notebook';
import { Kernel } from '@jupyterlab/services';

const TARGET = 'peaksMCP:frontend';

function cellJSON(panel: NotebookPanel): any {
  const notebook = panel.content;
  const cell = notebook.activeCell;
  if (!cell) { return {}; }
  return {
    id: cell.model.id,
    index: notebook.activeCellIndex,
    cell_type: cell.model.type,
    source: cell.model.sharedModel.getSource(),
    outputs: cell.model.type === 'code' ? (cell.model as any).outputs?.toJSON() ?? [] : []
  };
}

async function handle(panel: NotebookPanel, comm: Kernel.IComm, data: any): Promise<void> {
  if (data.type !== 'request') { return; }
  const request_id = data.request_id;
  const notebook = panel.content;
  try {
    let result: any = {};
    switch (data.operation) {
      case 'read_active_cell': result = cellJSON(panel); break;
      case 'read_notebook':
        result = { path: panel.context.path, active_index: notebook.activeCellIndex,
          cells: Array.from({length: notebook.widgets.length}, (_, i) => {
            const cell = notebook.widgets[i];
            return {id: cell.model.id, index: i, cell_type: cell.model.type, source: cell.model.sharedModel.getSource()};
          }) };
        break;
      case 'move_cursor':
        notebook.activeCellIndex = data.direction === 'index' ? data.index : Math.max(0, Math.min(notebook.widgets.length - 1, notebook.activeCellIndex + (data.direction === 'previous' ? -1 : 1)));
        result = cellJSON(panel); break;
      case 'request_consent':
        result = {approved: window.confirm(`peaksMCP requests ${data.requested_operation ?? 'a notebook change'}\n\n${data.details?.code ?? ''}`)}; break;
      case 'execute_code':
        NotebookActions.insertBelow(notebook); notebook.activeCell?.model.sharedModel.setSource(data.code ?? '');
        await NotebookActions.run(notebook, panel.sessionContext); result = cellJSON(panel); break;
      case 'execute_active_cell':
        await NotebookActions.run(notebook, panel.sessionContext); result = cellJSON(panel); break;
      case 'add_cell':
        NotebookActions.insertBelow(notebook);
        if (data.cell_type === 'markdown') { NotebookActions.changeCellType(notebook, 'markdown'); }
        else if (data.cell_type === 'raw') { NotebookActions.changeCellType(notebook, 'raw'); }
        notebook.activeCell?.model.sharedModel.setSource(data.source ?? ''); result = cellJSON(panel); break;
      case 'delete_cell':
        if (typeof data.index === 'number') { notebook.activeCellIndex = data.index; }
        NotebookActions.deleteCells(notebook); result = {deleted: true, active_index: notebook.activeCellIndex}; break;
      case 'apply_patch':
        notebook.activeCellIndex = data.index; notebook.activeCell?.model.sharedModel.setSource(data.source ?? ''); result = cellJSON(panel); break;
      case 'restart_kernel':
        // Frontend-initiated restart so JupyterLab reconnects the session and the
        // extension re-opens the Comm (a REST restart would leave the UI detached).
        await panel.sessionContext.restartKernel();
        result = {restarted: true, kernel: panel.sessionContext.session?.kernel?.id}; break;
      default: throw new Error(`Unsupported frontend operation: ${data.operation}`);
    }
    comm.send({request_id, ok: true, result});
  } catch (error) {
    comm.send({request_id, ok: false, error: error instanceof Error ? error.message : String(error)});
  }
}

const plugin: JupyterFrontEndPlugin<void> = {
  id: 'peaksmcp-jupyterlab:bridge', autoStart: true, requires: [INotebookTracker],
  activate: (_app: JupyterFrontEnd, tracker: INotebookTracker) => {
    let comm: Kernel.IComm | null = null;
    let connectedKernel = '';
    let heartbeatTimer: number | null = null;
    const attach = (panel: NotebookPanel | null): void => {
      if (!panel) { return; }

      const teardown = (why: string): void => {
        console.log(`[peaksmcp] teardown comm (${why})`);
        // Clear the shared heartbeat first: a stale interval from a previous
        // connection would otherwise fire on the dead Comm and tear down the
        // freshly reconnected one.
        if (heartbeatTimer !== null) { window.clearInterval(heartbeatTimer); heartbeatTimer = null; }
        if (comm) { try { comm.close(); } catch { /* noop */ } comm = null; }
        connectedKernel = '';
      };

      const connect = async (): Promise<void> => {
        const kernel = panel?.sessionContext.session?.kernel;
        if (!kernel || kernel.isDisposed) { return; }
        const status = String(kernel.status);
        console.log(`[peaksmcp] connect status=${status} connected=${connectedKernel === kernel.id && !!comm}`);
        // The kernel id stays identical across a restart, so the id+comm guard
        // alone cannot detect a stale Comm. Tear down on every transitional
        // status and rebuild once the new kernel reaches idle.
        if (status === 'restarting' || status === 'autorestarting' || status === 'starting' || status === 'connecting') {
          teardown(`kernel ${status}`);
          return;
        }
        if (connectedKernel === kernel.id && comm) { return; }
        teardown('kernel changed');
        connectedKernel = kernel.id;
        let newComm: Kernel.IComm | null = null;
        try {
          newComm = kernel.createComm(TARGET);
        } catch (error) {
          console.error('[peaksmcp] createComm failed:', String(error));
          teardown('createComm failed');
          return;
        }
        newComm.onMsg = msg => { void handle(panel, newComm!, (msg.content.data as any) ?? {}); };
        newComm.onClose = () => { if (comm === newComm) { teardown('comm closed'); } };
        comm = newComm;
        try {
          await newComm.open({kernel_id: kernel.id});
        } catch (error) {
          console.error('[peaksmcp] comm open failed:', String(error));
          teardown('open failed');
          return;
        }
        console.log(`[peaksmcp] comm open ok for kernel ${kernel.id}`);
        const publish = () => {
          try { newComm?.send({type: 'active_cell', cell: cellJSON(panel), outputs: cellJSON(panel).outputs}); }
          catch { teardown('publish send failed'); }
        };
        heartbeatTimer = window.setInterval(() => {
          try { newComm?.send({type: 'heartbeat'}); }
          catch { teardown('heartbeat send failed'); }
        }, 2000);
        const close = () => {
          if (heartbeatTimer !== null) { window.clearInterval(heartbeatTimer); heartbeatTimer = null; }
          try { newComm?.send({type: 'frontend_closing'}); } catch { /* noop */ }
          try { void newComm?.close(); } catch { /* noop */ }
          if (comm === newComm) { comm = null; connectedKernel = ''; }
        };
        panel.disposed.connect(close);
        window.addEventListener('beforeunload', close, {once: true});
        panel.content.activeCellChanged.connect(publish);
        publish();
      };

      // Reconnect on any kernel object change, transitional status, or when a
      // stale Comm is closed — so an external/frontend restart always recovers.
      panel.sessionContext.kernelChanged.connect(() => { teardown('kernelChanged'); void connect(); });
      panel.sessionContext.statusChanged.connect((_sc: any, status: any) => {
        const value = String(status ?? '');
        if (value === 'restarting' || value === 'autorestarting' || value === 'starting' || value === 'connecting') {
          teardown(`session ${value}`);
        } else {
          void connect();
        }
      });
      const kernel = panel.sessionContext.session?.kernel;
      if (kernel && !kernel.isDisposed) {
        kernel.statusChanged.connect((_k: any, status: any) => {
          const value = String(status ?? '');
          if (value === 'restarting' || value === 'autorestarting' || value === 'starting' || value === 'connecting') {
            teardown(`kernel ${value}`);
          } else {
            void connect();
          }
        });
      }
      void connect();
    };
    tracker.currentChanged.connect((_sender, panel) => { attach(panel); });
    if (tracker.currentWidget) { attach(tracker.currentWidget); }
  }
};
export default plugin;
