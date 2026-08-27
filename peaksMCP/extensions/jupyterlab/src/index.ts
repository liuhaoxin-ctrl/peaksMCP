import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { Dialog, showDialog } from '@jupyterlab/apputils';
import { INotebookTracker, NotebookActions, NotebookPanel } from '@jupyterlab/notebook';
import { Kernel } from '@jupyterlab/services';
import { Widget } from '@lumino/widgets';

const TARGET = 'peaksMCP:frontend';

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]!));
}

/**
 * Structured consent dialog (JupyterLab-native), replacing window.confirm.
 * Shows the operation, the exact code about to run, and any security notes
 * (e.g. figure-save consent) before the user decides.
 */
async function showConsentDialog(operation: string, details: any, targetCell?: any): Promise<boolean> {
  const body = document.createElement('div');
  body.style.maxWidth = '680px';
  body.style.fontSize = '13px';

  const header = document.createElement('div');
  header.style.marginBottom = '10px';
  header.innerHTML = `<strong>请求的操作:</strong> <code>${escapeHtml(operation)}</code>`;
  body.appendChild(header);

  // Which cell is being touched, and what does it currently contain?
  const cell = details?.cell ?? {};
  if (typeof cell.index === 'number') {
    const target = document.createElement('div');
    target.style.marginBottom = '10px';
    target.style.padding = '8px 10px';
    target.style.background = '#eef4ff';
    target.style.border = '1px solid #b8cdf0';
    target.style.borderRadius = '4px';
    target.style.color = '#1a3a6b';
    const actionText = cell.action === 'delete' ? '删除' : cell.action === 'overwrite' ? '覆盖' : '修改';
    target.innerHTML = `<strong>目标 cell #${escapeHtml(String(cell.index))}（${actionText}）</strong>`;
    body.appendChild(target);
    if (targetCell) {
      const current = document.createElement('div');
      current.style.marginBottom = '10px';
      const label = document.createElement('div');
      label.innerHTML = '<strong>该 cell 当前内容:</strong>';
      current.appendChild(label);
      const pre = document.createElement('pre');
      pre.textContent = String(targetCell);
      pre.style.maxHeight = '160px';
      pre.style.overflow = 'auto';
      pre.style.background = '#f5f5f5';
      pre.style.padding = '8px';
      pre.style.borderRadius = '4px';
      pre.style.border = '1px solid #ddd';
      pre.style.whiteSpace = 'pre-wrap';
      current.appendChild(pre);
      body.appendChild(current);
    }
  }

  const code = String(details?.code ?? '');
  if (code) {
    const label = document.createElement('div');
    label.style.margin = '10px 0 6px';
    label.innerHTML = '<strong>将执行的代码:</strong>';
    body.appendChild(label);
    const pre = document.createElement('pre');
    pre.textContent = code;
    pre.style.maxHeight = '280px';
    pre.style.overflow = 'auto';
    pre.style.background = '#f5f5f5';
    pre.style.padding = '10px';
    pre.style.borderRadius = '4px';
    pre.style.border = '1px solid #ddd';
    pre.style.whiteSpace = 'pre-wrap';
    pre.style.wordBreak = 'break-word';
    body.appendChild(pre);
  }

  const scan = details?.scan ?? {};
  const consentIssues: any[] = Array.isArray(scan.requires_explicit_consent) ? scan.requires_explicit_consent : [];
  for (const issue of consentIssues) {
    const note = document.createElement('div');
    note.style.marginTop = '10px';
    note.style.padding = '10px';
    note.style.background = '#fff3cd';
    note.style.border = '1px solid #ffc107';
    note.style.borderRadius = '4px';
    note.style.color = '#856404';
    note.innerHTML = `<strong>⚠️ ${escapeHtml(String(issue.description ?? '需要您确认的操作'))}</strong>`;
    body.appendChild(note);
  }

  const widget = new Widget({ node: body });
  const result = await showDialog({
    title: `peaksMCP — 确认 ${operation}`,
    body: widget,
    buttons: [
      Dialog.cancelButton({ label: '拒绝' }),
      Dialog.okButton({ label: '允许' })
    ],
    defaultButton: 1
  });
  return result.button.label === '允许';
}

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
      case 'request_consent': {
        // Pass the target cell's current source so the consent dialog can show
        // exactly which cell will be deleted/overwritten and what it holds now.
        const cellInfo = data.details?.cell;
        let targetSource: string | undefined;
        if (cellInfo && typeof cellInfo.index === 'number') {
          const targetWidget = notebook.widgets[cellInfo.index];
          if (targetWidget) { targetSource = targetWidget.model.sharedModel.getSource(); }
        }
        result = { approved: await showConsentDialog(data.requested_operation ?? 'notebook operation', data.details ?? {}, targetSource) }; break;
      }
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
    // Persist notebook mutations (executed / inserted / deleted / patched cells)
    // to disk so the analysis history survives a supervisor or JupyterLab restart.
    if (['execute_code', 'execute_active_cell', 'add_cell', 'delete_cell', 'apply_patch'].includes(data.operation)) {
      try { await panel.context.save(); } catch { /* save is best-effort */ }
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
