import { JupyterFrontEnd, JupyterFrontEndPlugin } from '@jupyterlab/application';
import { Dialog, showDialog } from '@jupyterlab/apputils';
import { INotebookTracker, NotebookActions, NotebookPanel } from '@jupyterlab/notebook';
import { Kernel } from '@jupyterlab/services';
import { Widget } from '@lumino/widgets';

const TARGET = 'peaksMCP:frontend';
const MAX_COMM_IMAGE_BYTES = 8 * 1024 * 1024;
const MAX_COMM_IMAGE_TOTAL_BYTES = 16 * 1024 * 1024;
const OMITTED_IMAGE_MIME = 'application/vnd.peaksmcp.image-omitted+json';

function outputString(value: any): string {
  return Array.isArray(value) ? value.join('') : String(value ?? '');
}

function utf8Bytes(value: string): number {
  let bytes = 0;
  for (const character of value) {
    const point = character.codePointAt(0) ?? 0;
    bytes += point <= 0x7f ? 1 : point <= 0x7ff ? 2 : point <= 0xffff ? 3 : 4;
  }
  return bytes;
}

function imageBytes(mime: string, payload: string): number {
  if (mime === 'image/svg+xml') { return utf8Bytes(payload); }
  const compact = payload.replace(/\s/g, '');
  const padding = compact.endsWith('==') ? 2 : compact.endsWith('=') ? 1 : 0;
  return Math.max(0, Math.floor(compact.length * 3 / 4) - padding);
}

function boundedOutputs(
  outputs: any[],
  perImageLimit = MAX_COMM_IMAGE_BYTES,
  totalLimit = MAX_COMM_IMAGE_TOTAL_BYTES,
): any[] {
  let includedBytes = 0;
  return (Array.isArray(outputs) ? outputs : []).map(output => {
    if (!output || typeof output !== 'object' || !output.data || typeof output.data !== 'object') {
      return output;
    }
    const data: any = {...output.data};
    const omitted: any[] = [];
    for (const mime of ['image/png', 'image/jpeg', 'image/svg+xml']) {
      if (!data[mime]) { continue; }
      const bytes = imageBytes(mime, outputString(data[mime]));
      if (bytes > perImageLimit || includedBytes + bytes > totalLimit) {
        delete data[mime];
        omitted.push({
          mime_type: mime,
          decoded_bytes: bytes,
          per_image_limit_bytes: perImageLimit,
          response_limit_bytes: totalLimit,
          reason: bytes > perImageLimit ? 'per_image_limit' : 'response_limit',
        });
      } else {
        includedBytes += bytes;
      }
    }
    if (omitted.length > 0) { data[OMITTED_IMAGE_MIME] = omitted; }
    return {...output, data};
  });
}

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

/**
 * Save consent card: shows the REAL result about to be written (path, kind,
 * size, sha256 of the exact staged bytes, and the data structure/statistics
 * or JSON preview) instead of code. Approving publishes those staged bytes;
 * there is no code-level approve anywhere.
 */
async function showSaveCard(payload: any): Promise<boolean> {
  const body = document.createElement('div');
  body.style.maxWidth = '680px';
  body.style.fontSize = '13px';

  const structure: any = payload?.structure ?? {};
  const stats: any = structure?.stats;

  const header = document.createElement('div');
  header.style.marginBottom = '10px';
  header.innerHTML = '<strong>将写入此文件（内容已固定，批准即原子写入）:</strong>';
  body.appendChild(header);

  const rows: Array<[string, string]> = [
    ['路径', String(payload?.path ?? '')],
    ['类型', String(structure?.kind ?? payload?.kind ?? '')],
    ['大小', `${(Number(payload?.size_bytes ?? 0) / 1024).toFixed(1)} KiB`],
    ['sha256', String(payload?.sha256 ?? '').slice(0, 16) + '…'],
  ];
  if (structure?.name != null) { rows.push(['名称', String(structure.name)]); }
  if (structure?.dims) { rows.push(['维度', structure.dims.join(', ')]); }
  if (structure?.sizes) {
    rows.push(['形状', Object.entries(structure.sizes).map(([d, s]) => `${d}×${s}`).join(', ')]);
  }
  if (structure?.dtype) { rows.push(['dtype', String(structure.dtype)]); }
  if (structure?.units) { rows.push(['单位', String(structure.units)]); }
  if (stats?.min != null && stats?.max != null) {
    rows.push(['数值范围', `[${Number(stats.min).toExponential(4)}, ${Number(stats.max).toExponential(4)}]`]);
  }
  if (stats?.nan_fraction != null) {
    rows.push(['NaN 占比', `${(Number(stats.nan_fraction) * 100).toFixed(3)}%`]);
  }

  const table = document.createElement('table');
  table.style.borderCollapse = 'collapse';
  table.style.width = '100%';
  for (const [label, value] of rows) {
    const tr = document.createElement('tr');
    const tdLabel = document.createElement('td');
    tdLabel.textContent = label;
    tdLabel.style.fontWeight = 'bold';
    tdLabel.style.padding = '3px 10px 3px 0';
    tdLabel.style.verticalAlign = 'top';
    tdLabel.style.whiteSpace = 'nowrap';
    const tdValue = document.createElement('td');
    tdValue.textContent = value;
    tdValue.style.padding = '3px 0';
    tdValue.style.wordBreak = 'break-all';
    tr.appendChild(tdLabel);
    tr.appendChild(tdValue);
    table.appendChild(tr);
  }
  body.appendChild(table);

  if (structure?.json_preview != null) {
    const label = document.createElement('div');
    label.style.margin = '10px 0 6px';
    label.innerHTML = '<strong>内容预览:</strong>';
    body.appendChild(label);
    const pre = document.createElement('pre');
    pre.textContent = String(structure.json_preview);
    pre.style.maxHeight = '240px';
    pre.style.overflow = 'auto';
    pre.style.background = '#f5f5f5';
    pre.style.padding = '8px';
    pre.style.borderRadius = '4px';
    pre.style.border = '1px solid #ddd';
    body.appendChild(pre);
  }

  const widget = new Widget({ node: body });
  const result = await showDialog({
    title: 'peaksMCP — 保存确认',
    body: widget,
    buttons: [
      Dialog.cancelButton({ label: '拒绝' }),
      Dialog.okButton({ label: '保存' })
    ],
    defaultButton: 0
  });
  return result.button.label === '保存';
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
    outputs: cell.model.type === 'code'
      ? boundedOutputs((cell.model as any).outputs?.toJSON() ?? [])
      : []
  };
}

async function handle(panel: NotebookPanel, comm: Kernel.IComm, data: any): Promise<void> {
  if (data.type !== 'request') { return; }
  const request_id = data.request_id;
  const notebook = panel.content;
  // Execution output is identity-scoped. It must not masquerade as a cursor
  // change when a delayed Matplotlib image arrives after the user moved away.
  const publishExecution = (cellData: any) => {
    try {
      comm.send({
        type: 'cell_output', cell_id: cellData.id,
        cell: cellData, outputs: cellData.outputs,
      });
    } catch { /* noop */ }
  };
  const watchExecutionOutputs = (cell: any, snapshot: () => any) => {
    publishExecution(snapshot());
    const outputs = cell?.model?.type === 'code' ? cell.model.outputs : null;
    if (!outputs?.changed?.connect || !outputs?.changed?.disconnect) { return; }
    const onChanged = () => { publishExecution(snapshot()); };
    outputs.changed.connect(onChanged);
    window.setTimeout(() => {
      try { outputs.changed.disconnect(onChanged); } catch { /* noop */ }
    }, 30000);
  };
  try {
    let result: any = {};
    switch (data.operation) {
      case 'read_active_cell': result = cellJSON(panel); break;
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
      case 'save_ticket': {
        // Save consent card: shows the staged result's real summary (path,
        // kind, size, sha256, structure/stats) - approval publishes exactly
        // those bytes through the kernel-side gateway.
        result = { approved: await showSaveCard(data.details ?? {}) }; break;
      }
      case 'execute_code': {
        // Append-only: always append the new cell at the END of the notebook,
        // regardless of where the user's cursor is. Never insert mid-document
        // and never overwrite an existing cell.
        notebook.activeCellIndex = notebook.widgets.length - 1;
        NotebookActions.insertBelow(notebook);
        const executed = notebook.activeCell;  // the cell we are about to run
        if (!executed) { throw new Error('could not create a code cell'); }
        executed.model.sharedModel.setSource(data.code ?? '');
        // run() acts on the whole UI selection. Only this new cell is authorised.
        const executionSuccess = await NotebookActions.runCells(notebook, [executed], panel.sessionContext);
        // Keep reporting the executed cell even if the user moves the cursor.
        const executedJSON = () => ({
          id: executed?.model.id, index: notebook.widgets.findIndex(w => w.model.id === executed?.model.id),
          cell_type: executed?.model.type, source: executed?.model.sharedModel.getSource(),
          execution_success: executionSuccess,
          outputs: executed && executed.model.type === 'code'
            ? boundedOutputs((executed.model as any).outputs?.toJSON() ?? [])
            : [],
        });
        result = executedJSON();
        // Matplotlib images may arrive after text output and after the user has
        // moved the cursor. Watch this exact cell, preserving its identity.
        watchExecutionOutputs(executed, executedJSON);
        break;
      }
      case 'add_cell':
        // Append-only: the new cell always lands at the END of the notebook.
        notebook.activeCellIndex = notebook.widgets.length - 1;
        NotebookActions.insertBelow(notebook);
        if (data.cell_type === 'markdown') { NotebookActions.changeCellType(notebook, 'markdown'); }
        else if (data.cell_type === 'raw') { NotebookActions.changeCellType(notebook, 'raw'); }
        notebook.activeCell?.model.sharedModel.setSource(data.source ?? ''); result = cellJSON(panel); break;
      case 'save_notebook':
        try {
          await panel.context.save();
          result = { saved: true };
        } catch (saveErr) {
          result = { saved: false, error: saveErr instanceof Error ? saveErr.message : String(saveErr) };
        }
        break;
      case 'restart_kernel':
        // Frontend-initiated restart so JupyterLab reconnects the session and the
        // extension re-opens the Comm (a REST restart would leave the UI detached).
        await panel.sessionContext.restartKernel();
        result = {restarted: true, kernel: panel.sessionContext.session?.kernel?.id}; break;
      default: throw new Error(`Unsupported frontend operation: ${data.operation}`);
    }
    // Persist notebook mutations (executed / inserted cells) to disk so the
    // analysis history survives a supervisor or JupyterLab restart.
    // A failed save is reported to the caller instead of being silently swallowed:
    // "executed" and "persisted" are distinct outcomes.
    if (['execute_code', 'add_cell'].includes(data.operation)) {
      try {
        await panel.context.save();
        result.saved = true;
      } catch (saveErr) {
        result.saved = false;
        result.save_error = saveErr instanceof Error ? saveErr.message : String(saveErr);
      }
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
    let activePanel: NotebookPanel | null = null;
    let panelDisconnectors: Array<() => void> = [];
    let disconnectKernelStatus: (() => void) | null = null;

    const isTransitional = (status: unknown): boolean =>
      ['restarting', 'autorestarting', 'starting', 'connecting'].includes(String(status ?? ''));

    const teardown = (why: string, notify = false): void => {
      console.log(`[peaksmcp] teardown comm (${why})`);
      if (heartbeatTimer !== null) {
        window.clearInterval(heartbeatTimer);
        heartbeatTimer = null;
      }
      const current = comm;
      comm = null;
      connectedKernel = '';
      if (current) {
        if (notify) {
          try { current.send({type: 'frontend_closing'}); } catch { /* noop */ }
        }
        try { void current.close(); } catch { /* noop */ }
      }
    };

    const cleanupPanelBindings = (): void => {
      for (const disconnect of panelDisconnectors.splice(0)) {
        try { disconnect(); } catch { /* noop */ }
      }
      if (disconnectKernelStatus) {
        try { disconnectKernelStatus(); } catch { /* noop */ }
        disconnectKernelStatus = null;
      }
    };

    const publish = (panel: NotebookPanel, current: Kernel.IComm): void => {
      if (activePanel !== panel || panel.isDisposed || comm !== current) { return; }
      const cell = cellJSON(panel);
      // Cursor metadata only: outputs travel with the execute response and the
      // executed-cell pushes, never with the active-cell notification.
      const meta = {id: cell.id, index: cell.index, cell_type: cell.cell_type, source: cell.source};
      try { current.send({type: 'active_cell', cell: meta}); }
      catch { if (comm === current) { teardown('publish send failed'); } }
    };

    const connect = async (): Promise<void> => {
      const panel = activePanel;
      const kernel = panel?.sessionContext.session?.kernel;
      if (!panel || panel.isDisposed || !kernel || kernel.isDisposed) { return; }
      const status = String(kernel.status);
      console.log(`[peaksmcp] connect status=${status} connected=${connectedKernel === kernel.id && !!comm}`);
      if (isTransitional(status)) {
        teardown(`kernel ${status}`);
        return;
      }
      if (connectedKernel === kernel.id && comm) { return; }
      teardown('kernel changed');

      let newComm: Kernel.IComm;
      try {
        newComm = kernel.createComm(TARGET);
      } catch (error) {
        console.error('[peaksmcp] createComm failed:', String(error));
        return;
      }
      comm = newComm;
      connectedKernel = kernel.id;
      newComm.onMsg = msg => {
        if (activePanel === panel && comm === newComm) {
          void handle(panel, newComm, (msg.content.data as any) ?? {});
        }
      };
      newComm.onClose = () => {
        if (comm === newComm) { teardown('comm closed'); }
      };
      try {
        await newComm.open({kernel_id: kernel.id});
      } catch (error) {
        console.error('[peaksmcp] comm open failed:', String(error));
        if (comm === newComm) { teardown('open failed'); }
        return;
      }
      if (activePanel !== panel || panel.isDisposed || comm !== newComm) {
        try { void newComm.close(); } catch { /* noop */ }
        return;
      }
      console.log(`[peaksmcp] comm open ok for kernel ${kernel.id}`);
      heartbeatTimer = window.setInterval(() => {
        if (activePanel !== panel || comm !== newComm) { return; }
        try { newComm.send({type: 'heartbeat'}); }
        catch { if (comm === newComm) { teardown('heartbeat send failed'); } }
      }, 2000);
      publish(panel, newComm);
    };

    const bindKernelStatus = (panel: NotebookPanel): void => {
      if (disconnectKernelStatus) {
        try { disconnectKernelStatus(); } catch { /* noop */ }
        disconnectKernelStatus = null;
      }
      const kernel = panel.sessionContext.session?.kernel;
      if (!kernel || kernel.isDisposed) { return; }
      const onKernelStatus = (_kernel: any, status: any): void => {
        if (activePanel !== panel) { return; }
        if (isTransitional(status)) { teardown(`kernel ${String(status)}`); }
        else { void connect(); }
      };
      kernel.statusChanged.connect(onKernelStatus);
      disconnectKernelStatus = () => { kernel.statusChanged.disconnect(onKernelStatus); };
    };

    const attach = (panel: NotebookPanel | null): void => {
      if (panel === activePanel) {
        if (panel) { void connect(); }
        return;
      }
      cleanupPanelBindings();
      teardown('active notebook changed');
      activePanel = panel;
      if (!panel) { return; }
      // Initial positioning: when the notebook already has cells (e.g. restored
      // from a snapshot), work starts AFTER the existing cells — park the
      // active cell on the last one so read/execute continue from the end.
      try {
        const widgets = panel.content.widgets;
        if (widgets.length > 0) {
          panel.content.activeCellIndex = widgets.length - 1;
        }
      } catch { /* positioning is best-effort */ }

      const onKernelChanged = (): void => {
        if (activePanel !== panel) { return; }
        teardown('kernelChanged');
        bindKernelStatus(panel);
        void connect();
      };
      const onSessionStatus = (_context: any, status: any): void => {
        if (activePanel !== panel) { return; }
        if (isTransitional(status)) { teardown(`session ${String(status)}`); }
        else { void connect(); }
      };
      const onActiveCellChanged = (): void => {
        const current = comm;
        if (activePanel !== panel || !current) { return; }
        publish(panel, current);
      };
      const onDisposed = (): void => {
        if (activePanel !== panel) { return; }
        cleanupPanelBindings();
        activePanel = null;
        teardown('active notebook disposed', true);
      };

      panel.sessionContext.kernelChanged.connect(onKernelChanged);
      panelDisconnectors.push(() => panel.sessionContext.kernelChanged.disconnect(onKernelChanged));
      panel.sessionContext.statusChanged.connect(onSessionStatus);
      panelDisconnectors.push(() => panel.sessionContext.statusChanged.disconnect(onSessionStatus));
      panel.content.activeCellChanged.connect(onActiveCellChanged);
      panelDisconnectors.push(() => panel.content.activeCellChanged.disconnect(onActiveCellChanged));
      panel.disposed.connect(onDisposed);
      panelDisconnectors.push(() => panel.disposed.disconnect(onDisposed));
      bindKernelStatus(panel);
      void connect();
    };

    window.addEventListener('beforeunload', () => {
      cleanupPanelBindings();
      activePanel = null;
      teardown('frontend closing', true);
    }, {once: true});
    tracker.currentChanged.connect((_sender, panel) => { attach(panel); });
    if (tracker.currentWidget) { attach(tracker.currentWidget); }
  }
};
export default plugin;
