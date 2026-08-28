// Exercise the real Comm request handler without a browser or a live kernel.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const extension = path.resolve(__dirname, '../../peaksMCP/extensions/jupyterlab');
const ts = require(path.join(extension, 'node_modules/typescript'));
const source = fs.readFileSync(path.join(extension, 'src/index.ts'), 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;

function makeCell(id, source = 'print(1)', deletable = true) {
  return {
    model: {
      id, type: 'code',
      getMetadata: key => key === 'deletable' ? deletable : undefined,
      sharedModel: { getSource: () => source, setSource: value => { source = value; } },
      outputs: { toJSON: () => [{ output_type: 'stream', text: 'done' }] },
    },
  };
}

function harness({ moveCursorDuringRun = false, executionSuccess = true } = {}) {
  const selected = new Set(['a', 'b']);
  const notebook = {
    widgets: [makeCell('a'), makeCell('b'), makeCell('c')],
    activeCellIndex: 0,
    get activeCell() { return this.widgets[this.activeCellIndex] ?? null; },
    deselectAll() { selected.clear(); },
  };
  const runs = [], deleted = [], messages = [];
  let saves = 0;
  const actions = {
    run() { throw new Error('selection-based execution is not a single-cell operation'); },
    async runCells(_notebook, cells) {
      runs.push(...cells.map(cell => cell.model.id));
      if (moveCursorDuringRun) { notebook.activeCellIndex = notebook.widgets.length - 1; }
      return executionSuccess;
    },
    insertBelow() {
      notebook.widgets.splice(++notebook.activeCellIndex, 0, makeCell('inserted'));
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
  };
  const modules = {
    '@jupyterlab/notebook': { NotebookActions: actions, INotebookTracker: {} },
    '@jupyterlab/apputils': {},
    '@lumino/widgets': {},
  };
  const context = {
    exports: {}, console,
    window: { setTimeout: () => 0 },
    require(name) {
      if (!(name in modules)) { throw new Error(`unexpected import ${name}`); }
      return modules[name];
    },
  };
  vm.runInNewContext(`${compiled}\nexports.testHandle = handle;`, context);
  const panel = { content: notebook, sessionContext: {}, context: { async save() { saves++; } } };
  return {
    notebook, runs, deleted, selected,
    get saves() { return saves; },
    async request(operation, payload = {}) {
      await context.exports.testHandle(panel, { send: message => messages.push(message) }, {
        type: 'request', request_id: 'test', operation, ...payload,
      });
      return messages.findLast(message => message.request_id === 'test');
    },
  };
}

test('execute_active_cell runs only the approved cell in a multi-selection', async () => {
  const h = harness();
  const reply = await h.request('execute_active_cell', { expected_id: 'a', expected_source: 'print(1)' });
  assert.equal(reply.ok, true);
  assert.deepEqual(h.runs, ['a']);
  assert.equal(h.saves, 1);
});

test('execute_code runs only the newly inserted cell', async () => {
  const h = harness();
  const reply = await h.request('execute_code', { code: 'answer = 42' });
  assert.equal(reply.ok, true);
  assert.deepEqual(h.runs, ['inserted']);
  assert.equal(reply.result.source, 'answer = 42');
  assert.equal(h.saves, 1);
});

for (const operation of ['execute_active_cell', 'execute_code']) {
  test(`${operation} reports failed execution separately from a successful Comm reply`, async () => {
    const h = harness({ executionSuccess: false });
    const reply = await h.request(operation, { code: 'raise ValueError("failed")' });
    assert.equal(reply.ok, true);
    assert.equal(reply.result.execution_success, false);
  });

  test(`${operation} reports the executed cell even if the active cell changes`, async () => {
    const h = harness({ moveCursorDuringRun: true });
    const reply = await h.request(operation, { code: 'answer = 42' });
    const expected = operation === 'execute_code' ? 'inserted' : 'a';
    assert.equal(reply.ok, true);
    assert.equal(reply.result.execution_success, true);
    assert.deepEqual(h.runs, [expected]);
    assert.equal(reply.result.id, expected);
    assert.equal(h.notebook.activeCell.model.id, 'c');
  });
}

for (const payload of [{ expected_id: 'other' }, { expected_source: 'different source' }]) {
  test(`execution rejects a changed approved cell: ${JSON.stringify(payload)}`, async () => {
    const h = harness();
    assert.equal((await h.request('execute_active_cell', payload)).ok, false);
    assert.deepEqual(h.runs, []);
    assert.equal(h.saves, 0);
  });
}

for (const [index, expected] of [[1, 'b'], [null, 'a']]) {
  test(`delete_cell deletes only its target (${index}) despite a multi-selection`, async () => {
    const h = harness();
    const reply = await h.request('delete_cell', { index });
    assert.equal(reply.ok, true);
    assert.equal(reply.result.id, expected);
    assert.deepEqual(h.deleted, [expected]);
    assert.equal(h.notebook.widgets.length, 2);
    assert.equal(h.saves, 1);
  });
}

for (const index of [-1, 3, 0.5, '1']) {
  test(`delete_cell rejects invalid index ${JSON.stringify(index)} without deleting`, async () => {
    const h = harness();
    assert.equal((await h.request('delete_cell', { index })).ok, false);
    assert.deepEqual(h.deleted, []);
    assert.equal(h.saves, 0);
  });
}

test('delete_cell respects the target cell deletable metadata', async () => {
  const h = harness();
  h.notebook.widgets[0] = makeCell('a', 'print(1)', false);
  assert.equal((await h.request('delete_cell', { index: 0 })).ok, false);
  assert.deepEqual(h.deleted, []);
});
