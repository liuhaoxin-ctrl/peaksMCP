// Exercise the real Comm request handler without a browser or a live kernel.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const extension = path.resolve(__dirname, '../../peaksMCP/extensions/jupyterlab');
const ts = require(path.join(extension, 'node_modules/typescript'));
const source = fs.readFileSync(path.join(extension, 'src/index.ts'), 'utf8');
const outputStyle = fs.readFileSync(path.join(extension, 'style/index.css'), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;

test('verbose text outputs scroll without constraining inline figures', () => {
  assert.match(outputStyle, /data-mime-type='text\/plain'/);
  assert.match(outputStyle, /data-mime-type='application\/vnd\.jupyter\.stdout'/);
  assert.match(outputStyle, /data-mime-type='application\/vnd\.jupyter\.stderr'/);
  assert.match(outputStyle, /max-height:\s*min\(16rem, 36vh\)/);
  assert.match(outputStyle, /overflow-y:\s*auto/);
  assert.doesNotMatch(outputStyle, /image\/(?:png|jpeg|svg\+xml)/);
  assert.doesNotMatch(outputStyle, /jp-RenderedImage/);
});

function makeCell(id, source = 'print(1)', deletable = true, type = 'code') {
  let outputJSON = [{ output_type: 'stream', text: 'done' }];
  const outputListeners = new Set();
  return {
    model: {
      id, type,
      getMetadata: key => key === 'deletable' ? deletable : undefined,
      sharedModel: { getSource: () => source, setSource: value => { source = value; } },
      outputs: {
        toJSON: () => outputJSON,
        setJSON: value => { outputJSON = value; },
        changed: {
          connect: callback => outputListeners.add(callback),
          disconnect: callback => outputListeners.delete(callback),
          emit: () => { for (const callback of outputListeners) { callback(); } },
        },
      },
    },
  };
}

function harness({
  moveCursorDuringRun = false,
  executionSuccess = true,
  insertedCellType = 'code',
} = {}) {
  const selected = new Set(['a', 'b']);
  const notebook = {
    widgets: [makeCell('a'), makeCell('b'), makeCell('c')],
    activeCellIndex: 0,
    get activeCell() { return this.widgets[this.activeCellIndex] ?? null; },
    deselectAll() { selected.clear(); },
  };
  const runs = [], deleted = [], messages = [];
  let outputSnapshots = 0;
  let saves = 0;
  const actions = {
    run() { throw new Error('selection-based execution is not a single-cell operation'); },
    async runCells(_notebook, cells) {
      runs.push(...cells.map(cell => cell.model.id));
      if (moveCursorDuringRun) { notebook.activeCellIndex = notebook.widgets.length - 1; }
      return executionSuccess;
    },
    insertBelow() {
      const inserted = makeCell('inserted', 'print(1)', true, insertedCellType);
      const toJSON = inserted.model.outputs.toJSON;
      inserted.model.outputs.toJSON = () => { outputSnapshots++; return toJSON(); };
      notebook.widgets.splice(++notebook.activeCellIndex, 0, inserted);
      // Keep the old selection to verify the handler explicitly pins execution.
    },
    deleteCells() {
      const active = notebook.activeCell;
      notebook.widgets = notebook.widgets.filter(cell => {
        if (cell === active || selected.has(cell.model.id)) {
          deleted.push(cell.model.id);
          return false;
        }
        return true;
      });
    },
    changeCellType(_notebook, type) {
      notebook.activeCell.model.type = type;
    },
  };
  const modules = {
    '@jupyterlab/notebook': { NotebookActions: actions, INotebookTracker: {} },
    '@jupyterlab/apputils': {},
    '@lumino/widgets': {},
  };
  const context = {
    exports: {}, console,
    window: { setTimeout: (fn, ms) => setTimeout(fn, ms) },
    require(name) {
      if (!(name in modules)) { throw new Error(`unexpected import ${name}`); }
      return modules[name];
    },
  };
  vm.runInNewContext(
    `${compiled}\nexports.testHandle = handle; exports.testBoundedOutputs = boundedOutputs;`,
    context,
  );
  const panel = { content: notebook, sessionContext: {}, context: { async save() { saves++; } } };
  return {
    notebook, runs, deleted, selected, messages,
    boundOutputs(outputs, perImageLimit, totalLimit) {
      return context.exports.testBoundedOutputs(outputs, perImageLimit, totalLimit);
    },
    get saves() { return saves; },
    get outputSnapshots() { return outputSnapshots; },
    async request(operation, payload = {}) {
      await context.exports.testHandle(panel, { send: message => messages.push(message) }, {
        type: 'request', request_id: 'test', operation, ...payload,
      });
      return messages.findLast(message => message.request_id === 'test');
    },
  };
}

test('execute_code runs only the newly inserted cell', async () => {
  const h = harness();
  const reply = await h.request('execute_code', { code: 'answer = 42' });
  assert.equal(reply.ok, true);
  assert.deepEqual(h.runs, ['inserted']);
  assert.equal(reply.result.source, 'answer = 42');
  assert.equal(h.saves, 1);
});

test('execute_code forces the appended cell to code when defaultCell is markdown', async () => {
  const h = harness({ insertedCellType: 'markdown' });
  const reply = await h.request('execute_code', { code: 'answer = 42' });
  assert.equal(reply.ok, true);
  assert.deepEqual(h.runs, ['inserted']);
  assert.equal(reply.result.cell_type, 'code');
  assert.equal(h.notebook.widgets.find(cell => cell.model.id === 'inserted').model.type, 'code');
});

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

test('execution output is settled inside ONE reply, without Comm pushes', async () => {
  const h = harness({ moveCursorDuringRun: true });
  const reply = await h.request('execute_code', { code: 'print(1)' });
  const pushes = h.messages.filter(message => !message.request_id);
  assert.equal(pushes.length, 0);  // no repeated cell_output pushes any more
  assert.equal(reply.ok, true);
  assert.equal(reply.result.outputs[0].text, 'done');  // settled snapshot
  assert.equal(h.outputSnapshots, 1);
  assert.equal(h.notebook.activeCell.model.id, 'inserted');
});

test('settled reply waits for 200ms of output quiet', async () => {
  const h = harness();
  const started = Date.now();
  const reply = await h.request('execute_code', { code: 'print(1)' });
  assert.equal(reply.ok, true);
  assert.ok(Date.now() - started >= 180, 'reply returned before the 200ms quiet window');
  assert.equal(h.outputSnapshots, 1);
});

test('settled reply is capped at 2s even while output keeps changing', async () => {
  const h = harness();
  const started = Date.now();
  const pending = h.request('execute_code', { code: 'print(1)' });
  const timer = setInterval(() => {
    const executed = h.notebook.widgets.find(cell => cell.model.id === 'inserted');
    executed?.model.outputs.changed.emit();
  }, 80);
  try {
    const reply = await pending;
    const elapsed = Date.now() - started;
    assert.equal(reply.ok, true);
    assert.ok(elapsed >= 1800, `reply ignored the 2s cap: ${elapsed}ms`);
    assert.ok(elapsed < 2400, `reply exceeded the 2s cap: ${elapsed}ms`);
    assert.equal(h.outputSnapshots, 1);
  } finally {
    clearInterval(timer);
  }
});

test('output arriving after the settled reply is not pushed or resnapshotted', async () => {
  const h = harness();
  const reply = await h.request('execute_code', { code: 'print(1)' });
  const executed = h.notebook.widgets.find(cell => cell.model.id === 'inserted');
  executed.model.outputs.setJSON([
    {output_type: 'display_data', data: {'image/png': 'TEFURQ=='}},
  ]);
  executed.model.outputs.changed.emit();
  await sleep(40);
  assert.equal(reply.result.outputs[0].text, 'done');
  assert.equal(h.outputSnapshots, 1);
  assert.equal(h.messages.filter(message => !message.request_id).length, 0);
});

test('an image landing inside the quiet window is part of the settled reply', async () => {
  const h = harness({ moveCursorDuringRun: true });
  const pending = h.request('execute_code', { code: 'print(1)' });
  await sleep(80);  // inside the 200ms quiet window after kernel idle
  const executed = h.notebook.widgets.find(cell => cell.model.id === 'inserted');
  executed.model.outputs.setJSON([
    {output_type: 'display_data', data: {'image/png': 'QUJD'}},
  ]);
  executed.model.outputs.changed.emit();
  const reply = await pending;
  assert.equal(reply.ok, true);
  assert.equal(h.notebook.activeCell.model.id, 'inserted');
  assert.equal(reply.result.outputs[0].data['image/png'], 'QUJD');
  const pushes = h.messages.filter(message => !message.request_id);
  assert.equal(pushes.length, 0);
});

test('oversized images are removed before crossing the Comm', () => {
  const h = harness();
  const bounded = h.boundOutputs([
    {output_type: 'display_data', data: {'image/png': 'QUJDREVGRw=='}},
  ], 4, 8);
  assert.equal(bounded[0].data['image/png'], undefined);
  const omitted = bounded[0].data['application/vnd.peaksmcp.image-omitted+json'];
  assert.equal(omitted[0].decoded_bytes, 7);
  assert.equal(omitted[0].reason, 'per_image_limit');
});

test('execute_code reports failed execution separately from a successful Comm reply', async () => {
  const h = harness({ executionSuccess: false });
  const reply = await h.request('execute_code', { code: 'raise ValueError("failed")' });
  assert.equal(reply.ok, true);
  assert.equal(reply.result.execution_success, false);
});

test('execute_code reports the executed cell even if the active cell changes', async () => {
  const h = harness({ moveCursorDuringRun: true });
  const reply = await h.request('execute_code', { code: 'answer = 42' });
  assert.equal(reply.ok, true);
  assert.equal(reply.result.execution_success, true);
  assert.deepEqual(h.runs, ['inserted']);
  assert.equal(reply.result.id, 'inserted');
  assert.equal(h.notebook.activeCell.model.id, 'inserted');
});
