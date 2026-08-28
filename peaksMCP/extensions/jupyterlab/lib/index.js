import { Dialog, showDialog } from '@jupyterlab/apputils';
import { INotebookTracker, NotebookActions } from '@jupyterlab/notebook';
import { Widget } from '@lumino/widgets';
const TARGET = 'peaksMCP:frontend';
function escapeHtml(text) {
    return text.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
/**
 * Structured consent dialog (JupyterLab-native), replacing window.confirm.
 * Shows the operation, the exact code about to run, and any security notes
 * (e.g. figure-save consent) before the user decides.
 */
async function showConsentDialog(operation, details, targetCell) {
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
    const consentIssues = Array.isArray(scan.requires_explicit_consent) ? scan.requires_explicit_consent : [];
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
function cellJSON(panel) {
    const notebook = panel.content;
    const cell = notebook.activeCell;
    if (!cell) {
        return {};
    }
    return {
        id: cell.model.id,
        index: notebook.activeCellIndex,
        cell_type: cell.model.type,
        source: cell.model.sharedModel.getSource(),
        outputs: cell.model.type === 'code' ? cell.model.outputs?.toJSON() ?? [] : []
    };
}
async function handle(panel, comm, data) {
    if (data.type !== 'request') {
        return;
    }
    const request_id = data.request_id;
    const notebook = panel.content;
    // Push the live cell + its latest outputs so the backend cache (and therefore
    // the MCP notebook_read_active_cell_output tool) never serves stale output.
    const publishLive = (cellData) => {
        const cell = cellData ?? cellJSON(panel);
        try {
            comm.send({ type: 'active_cell', cell: cell, outputs: cell.outputs });
        }
        catch { /* noop */ }
    };
    try {
        let result = {};
        switch (data.operation) {
            case 'read_active_cell':
                result = cellJSON(panel);
                break;
            case 'read_notebook':
                result = { path: panel.context.path, active_index: notebook.activeCellIndex,
                    cells: Array.from({ length: notebook.widgets.length }, (_, i) => {
                        const cell = notebook.widgets[i];
                        return { id: cell.model.id, index: i, cell_type: cell.model.type, source: cell.model.sharedModel.getSource() };
                    }) };
                break;
            case 'move_cursor':
                notebook.activeCellIndex = data.direction === 'index' ? data.index : Math.max(0, Math.min(notebook.widgets.length - 1, notebook.activeCellIndex + (data.direction === 'previous' ? -1 : 1)));
                result = cellJSON(panel);
                break;
            case 'request_consent': {
                // Pass the target cell's current source so the consent dialog can show
                // exactly which cell will be deleted/overwritten and what it holds now.
                const cellInfo = data.details?.cell;
                let targetSource;
                if (cellInfo && typeof cellInfo.index === 'number') {
                    const targetWidget = notebook.widgets[cellInfo.index];
                    if (targetWidget) {
                        targetSource = targetWidget.model.sharedModel.getSource();
                    }
                }
                result = { approved: await showConsentDialog(data.requested_operation ?? 'notebook operation', data.details ?? {}, targetSource) };
                break;
            }
            case 'execute_code': {
                NotebookActions.insertBelow(notebook);
                const executed = notebook.activeCell; // the cell we are about to run
                if (!executed) {
                    throw new Error('could not create a code cell');
                }
                executed.model.sharedModel.setSource(data.code ?? '');
                // run() acts on the whole UI selection. Only this new cell is authorised.
                const executionSuccess = await NotebookActions.runCells(notebook, [executed], panel.sessionContext);
                // Keep reporting the executed cell even if the user moves the cursor.
                const executedJSON = () => ({
                    id: executed?.model.id, index: notebook.widgets.findIndex(w => w.model.id === executed?.model.id),
                    cell_type: executed?.model.type, source: executed?.model.sharedModel.getSource(),
                    execution_success: executionSuccess,
                    outputs: executed && executed.model.type === 'code' ? executed.model.outputs?.toJSON() ?? [] : [],
                });
                result = executedJSON();
                publishLive(result);
                // Matplotlib images can arrive at the outputs model after the cell
                // finishes; poll the executed cell and re-push so the backend cache
                // never serves stale (empty) output. outputs.changed is the long-tail
                // fallback for user-driven edits.
                (() => {
                    const deadline = Date.now() + 30000;
                    const poll = () => {
                        const latest = executedJSON();
                        if (latest.outputs.length > 0 || Date.now() > deadline) {
                            publishLive(latest);
                        }
                        else {
                            window.setTimeout(poll, 400);
                        }
                    };
                    poll();
                })();
                break;
            }
            case 'execute_active_cell': {
                // TOCTOU guard: the scanned/authorised source must match the live cell
                // right now.  If the user edited the cell after the scan, refuse and ask
                // the agent to re-run instead of executing unchecked code.
                const cell = notebook.activeCell;
                if (!cell) {
                    throw new Error('no active cell');
                }
                if (data.expected_id && cell.model.id !== data.expected_id) {
                    throw new Error('active cell changed since scan — please re-run');
                }
                if (typeof data.expected_source === 'string' && cell.model.sharedModel.getSource() !== data.expected_source) {
                    throw new Error('cell content changed since scan — please re-run');
                }
                const executionSuccess = await NotebookActions.runCells(notebook, [cell], panel.sessionContext);
                // Report the authorised cell, regardless of the current UI selection.
                const executedJSON = () => ({
                    id: cell.model.id, index: notebook.widgets.findIndex(w => w.model.id === cell.model.id),
                    cell_type: cell.model.type, source: cell.model.sharedModel.getSource(),
                    execution_success: executionSuccess,
                    outputs: cell.model.type === 'code' ? cell.model.outputs?.toJSON() ?? [] : [],
                });
                result = executedJSON();
                publishLive(result);
                const deadline = Date.now() + 30000;
                const poll = () => {
                    const latest = executedJSON();
                    if (latest.outputs.length > 0 || Date.now() > deadline) {
                        publishLive(latest);
                    }
                    else {
                        window.setTimeout(poll, 400);
                    }
                };
                poll();
                break;
            }
            case 'add_cell':
                NotebookActions.insertBelow(notebook);
                if (data.cell_type === 'markdown') {
                    NotebookActions.changeCellType(notebook, 'markdown');
                }
                else if (data.cell_type === 'raw') {
                    NotebookActions.changeCellType(notebook, 'raw');
                }
                notebook.activeCell?.model.sharedModel.setSource(data.source ?? '');
                result = cellJSON(panel);
                break;
            case 'delete_cell': {
                const index = data.index ?? notebook.activeCellIndex;
                if (!Number.isInteger(index) || index < 0 || index >= notebook.widgets.length) {
                    throw new Error('cell index is out of range');
                }
                const target = notebook.widgets[index];
                if (target.model.getMetadata('deletable') === false) {
                    throw new Error('target cell is not deletable');
                }
                const targetId = target.model.id;
                notebook.activeCellIndex = index;
                // deleteCells() deletes every selected cell, not just activeCellIndex.
                notebook.deselectAll();
                NotebookActions.deleteCells(notebook);
                result = { deleted: true, id: targetId, active_index: notebook.activeCellIndex };
                break;
            }
            case 'apply_patch':
                notebook.activeCellIndex = data.index;
                notebook.activeCell?.model.sharedModel.setSource(data.source ?? '');
                result = cellJSON(panel);
                break;
            case 'restart_kernel':
                // Frontend-initiated restart so JupyterLab reconnects the session and the
                // extension re-opens the Comm (a REST restart would leave the UI detached).
                await panel.sessionContext.restartKernel();
                result = { restarted: true, kernel: panel.sessionContext.session?.kernel?.id };
                break;
            default: throw new Error(`Unsupported frontend operation: ${data.operation}`);
        }
        // Persist notebook mutations (executed / inserted / deleted / patched cells)
        // to disk so the analysis history survives a supervisor or JupyterLab restart.
        if (['execute_code', 'execute_active_cell', 'add_cell', 'delete_cell', 'apply_patch'].includes(data.operation)) {
            try {
                await panel.context.save();
            }
            catch { /* save is best-effort */ }
        }
        comm.send({ request_id, ok: true, result });
    }
    catch (error) {
        comm.send({ request_id, ok: false, error: error instanceof Error ? error.message : String(error) });
    }
}
const plugin = {
    id: 'peaksmcp-jupyterlab:bridge', autoStart: true, requires: [INotebookTracker],
    activate: (_app, tracker) => {
        let comm = null;
        let connectedKernel = '';
        let heartbeatTimer = null;
        let activePanel = null;
        let panelDisconnectors = [];
        let disconnectKernelStatus = null;
        let disconnectOutputs = null;
        const isTransitional = (status) => ['restarting', 'autorestarting', 'starting', 'connecting'].includes(String(status ?? ''));
        const clearOutputBinding = () => {
            if (disconnectOutputs) {
                try {
                    disconnectOutputs();
                }
                catch { /* noop */ }
                disconnectOutputs = null;
            }
        };
        const teardown = (why, notify = false) => {
            console.log(`[peaksmcp] teardown comm (${why})`);
            clearOutputBinding();
            if (heartbeatTimer !== null) {
                window.clearInterval(heartbeatTimer);
                heartbeatTimer = null;
            }
            const current = comm;
            comm = null;
            connectedKernel = '';
            if (current) {
                if (notify) {
                    try {
                        current.send({ type: 'frontend_closing' });
                    }
                    catch { /* noop */ }
                }
                try {
                    void current.close();
                }
                catch { /* noop */ }
            }
        };
        const cleanupPanelBindings = () => {
            for (const disconnect of panelDisconnectors.splice(0)) {
                try {
                    disconnect();
                }
                catch { /* noop */ }
            }
            if (disconnectKernelStatus) {
                try {
                    disconnectKernelStatus();
                }
                catch { /* noop */ }
                disconnectKernelStatus = null;
            }
            clearOutputBinding();
        };
        const publish = (panel, current) => {
            if (activePanel !== panel || panel.isDisposed || comm !== current) {
                return;
            }
            const cell = cellJSON(panel);
            try {
                current.send({ type: 'active_cell', cell, outputs: cell.outputs ?? [] });
            }
            catch {
                if (comm === current) {
                    teardown('publish send failed');
                }
            }
        };
        const bindActiveOutputs = (panel, current) => {
            clearOutputBinding();
            if (activePanel !== panel || comm !== current) {
                return;
            }
            const active = panel.content.activeCell;
            const outputs = active && active.model.type === 'code' ? active.model.outputs : null;
            if (!outputs) {
                return;
            }
            const onOutputsChanged = () => { publish(panel, current); };
            outputs.changed.connect(onOutputsChanged);
            disconnectOutputs = () => { outputs.changed.disconnect(onOutputsChanged); };
        };
        const connect = async () => {
            const panel = activePanel;
            const kernel = panel?.sessionContext.session?.kernel;
            if (!panel || panel.isDisposed || !kernel || kernel.isDisposed) {
                return;
            }
            const status = String(kernel.status);
            console.log(`[peaksmcp] connect status=${status} connected=${connectedKernel === kernel.id && !!comm}`);
            if (isTransitional(status)) {
                teardown(`kernel ${status}`);
                return;
            }
            if (connectedKernel === kernel.id && comm) {
                return;
            }
            teardown('kernel changed');
            let newComm;
            try {
                newComm = kernel.createComm(TARGET);
            }
            catch (error) {
                console.error('[peaksmcp] createComm failed:', String(error));
                return;
            }
            comm = newComm;
            connectedKernel = kernel.id;
            newComm.onMsg = msg => {
                if (activePanel === panel && comm === newComm) {
                    void handle(panel, newComm, msg.content.data ?? {});
                }
            };
            newComm.onClose = () => {
                if (comm === newComm) {
                    teardown('comm closed');
                }
            };
            try {
                await newComm.open({ kernel_id: kernel.id });
            }
            catch (error) {
                console.error('[peaksmcp] comm open failed:', String(error));
                if (comm === newComm) {
                    teardown('open failed');
                }
                return;
            }
            if (activePanel !== panel || panel.isDisposed || comm !== newComm) {
                try {
                    void newComm.close();
                }
                catch { /* noop */ }
                return;
            }
            console.log(`[peaksmcp] comm open ok for kernel ${kernel.id}`);
            heartbeatTimer = window.setInterval(() => {
                if (activePanel !== panel || comm !== newComm) {
                    return;
                }
                try {
                    newComm.send({ type: 'heartbeat' });
                }
                catch {
                    if (comm === newComm) {
                        teardown('heartbeat send failed');
                    }
                }
            }, 2000);
            publish(panel, newComm);
            bindActiveOutputs(panel, newComm);
        };
        const bindKernelStatus = (panel) => {
            if (disconnectKernelStatus) {
                try {
                    disconnectKernelStatus();
                }
                catch { /* noop */ }
                disconnectKernelStatus = null;
            }
            const kernel = panel.sessionContext.session?.kernel;
            if (!kernel || kernel.isDisposed) {
                return;
            }
            const onKernelStatus = (_kernel, status) => {
                if (activePanel !== panel) {
                    return;
                }
                if (isTransitional(status)) {
                    teardown(`kernel ${String(status)}`);
                }
                else {
                    void connect();
                }
            };
            kernel.statusChanged.connect(onKernelStatus);
            disconnectKernelStatus = () => { kernel.statusChanged.disconnect(onKernelStatus); };
        };
        const attach = (panel) => {
            if (panel === activePanel) {
                if (panel) {
                    void connect();
                }
                return;
            }
            cleanupPanelBindings();
            teardown('active notebook changed');
            activePanel = panel;
            if (!panel) {
                return;
            }
            const onKernelChanged = () => {
                if (activePanel !== panel) {
                    return;
                }
                teardown('kernelChanged');
                bindKernelStatus(panel);
                void connect();
            };
            const onSessionStatus = (_context, status) => {
                if (activePanel !== panel) {
                    return;
                }
                if (isTransitional(status)) {
                    teardown(`session ${String(status)}`);
                }
                else {
                    void connect();
                }
            };
            const onActiveCellChanged = () => {
                const current = comm;
                if (activePanel !== panel || !current) {
                    return;
                }
                publish(panel, current);
                bindActiveOutputs(panel, current);
            };
            const onDisposed = () => {
                if (activePanel !== panel) {
                    return;
                }
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
        }, { once: true });
        tracker.currentChanged.connect((_sender, panel) => { attach(panel); });
        if (tracker.currentWidget) {
            attach(tracker.currentWidget);
        }
    }
};
export default plugin;
//# sourceMappingURL=index.js.map