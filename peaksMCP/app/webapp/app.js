const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
const pretty = x => JSON.stringify(x, null, 2);

function bind(selector, event, handler) {
  const el = $(selector);
  if (el) el.addEventListener(event, handler);
  return el;
}

// =====================
// UI Utilities
// =====================

function toast(message, type = 'info', duration = 4000) {
  const container = $('#toast-container');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  const icons = { success: '◉', error: '◉', warning: '◉', info: '◉' };
  const colors = {
    success: 'var(--accent)',
    error: 'var(--danger)',
    warning: 'var(--warning)',
    info: 'var(--info)'
  };
  const icon = document.createElement('span');
  icon.style.color = colors[type] || colors.info;
  icon.style.fontSize = '10px';
  icon.textContent = icons[type] || '◉';
  const text = document.createElement('span');
  text.textContent = String(message);
  el.append(icon, text);
  container.appendChild(el);

  el.addEventListener('click', () => {
    el.style.animation = 'toastOut 0.25s ease forwards';
    setTimeout(() => el.remove(), 250);
  });

  setTimeout(() => {
    if (el.parentNode) {
      el.style.animation = 'toastOut 0.25s ease forwards';
      setTimeout(() => el.remove(), 250);
    }
  }, duration);
}

function confirmAction(message, onConfirm) {
  $('#modal-body').textContent = message;
  $('#confirm-modal').classList.add('active');

  const btn = $('#modal-confirm');
  const newBtn = btn.cloneNode(true);
  btn.parentNode.replaceChild(newBtn, btn);

  newBtn.addEventListener('click', () => {
    $('#confirm-modal').classList.remove('active');
    onConfirm();
  });
}

bind('#modal-cancel', 'click', () => {
  $('#confirm-modal').classList.remove('active');
});

bind('#confirm-modal', 'click', (e) => {
  if (e.target === e.currentTarget) {
    $('#confirm-modal').classList.remove('active');
  }
});

// =====================
// Controls
// =====================

function setEnabled(btn, enabled) {
  if (!btn) return;
  btn.disabled = !enabled;
  btn.style.opacity = enabled ? '1' : '0.4';
  btn.style.cursor = enabled ? 'pointer' : 'not-allowed';
  btn.style.pointerEvents = enabled ? 'auto' : 'none';
}

function renderControls(s) {
  const mcpReady = (s.components?.mcp?.state) === 'ready';
  const lab = $('#open-lab');
  if (lab) {
    lab.href = s.notebook_open_url || s.notebook_url || '';
    const on = !!lab.href;
    lab.style.pointerEvents = on ? 'auto' : 'none';
    lab.style.opacity = on ? '1' : '0.35';
  }
  setEnabled($('#start-mcp'), !mcpReady);
  setEnabled($('#restart-mcp'), mcpReady);
  setEnabled($('#restart-kernel'), true);
  setEnabled($('#restart-all'), true);
}

// =====================
// Components
// =====================

function renderComponents(s) {
  $('#profile').textContent = `profile: ${s.profile}`;
  $('#state').textContent = (s.aggregate || s.status || '?').toUpperCase();

  const container = $('#components');
  const comps = s.components || {};
  const entries = Object.entries(comps);

  if (entries.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">◻</div>
        <div class="empty-title">No components</div>
        <div class="empty-desc">Awaiting MCP service handshake…</div>
      </div>`;
  } else {
    container.replaceChildren();
    const allowedStates = new Set(['ready', 'degraded', 'error', 'loading', 'unknown']);
    entries.forEach(([name, c]) => {
      const stateClass = allowedStates.has(c.state) ? c.state : 'unknown';
      const stateIcon = {
        ready: '●',
        degraded: '◐',
        error: '●',
        loading: '◐',
        unknown: '○'
      }[stateClass] || '○';
      const article = document.createElement('article');
      article.className = `component ${stateClass}`;
      const header = document.createElement('div');
      header.className = 'component-header';
      const icon = document.createElement('span');
      icon.className = 'component-icon';
      icon.textContent = stateIcon;
      const state = document.createElement('span');
      state.className = 'component-state';
      state.textContent = stateClass.toUpperCase();
      header.append(icon, state);
      const title = document.createElement('b');
      title.textContent = String(name);
      const detail = document.createElement('span');
      detail.textContent = String(c.detail || '');
      article.append(header, title, detail);
      container.appendChild(article);
    });
  }

  const agg = s.aggregate || s.status || 'unknown';
  const badge = $('#components-status');
  if (badge) {
    badge.className = `status-badge ${agg}`;
    badge.textContent = agg.toUpperCase();
  }

  renderControls(s);
}

async function refresh() {
  try {
    const s = await fetch('/api/status').then(r => r.json());
    $('#connection').classList.add('up');
    renderComponents(s);
  } catch (e) {
    $('#state').textContent = 'OFFLINE';
    $('#connection').classList.remove('up');
    toast('Connection lost — retrying…', 'error');
  }
}

// =====================
// Actions
// =====================

async function postAction(path, label, options = {}) {
  const { confirm: needConfirm, confirmMsg } = options;

  const run = async () => {
    $('#action-result').style.display = 'block';
    $('#action-result').textContent = `${label}…`;
    try {
      const resp = await fetch(path, { method: 'POST' });
      const r = await resp.json();
      if (!resp.ok) {
        $('#action-result').textContent = pretty(r);
        toast(`${label} failed`, 'error');
        refresh();
        return;
      }
      // Some endpoints answer 200 but carry an explicit failure (restart
      // ready:false, load loaded:false, snapshot errors).
      if (r && (r.ready === false || r.ok === false || r.loaded === false || r.error)) {
        $('#action-result').textContent = pretty(r);
        toast(`${label} failed (partial or unsuccessful)`, 'error');
        refresh();
        return;
      }
      $('#action-result').textContent = pretty(r);
      toast(`${label} completed`, 'success');
    } catch (e) {
      $('#action-result').textContent = String(e);
      toast(`${label} failed`, 'error');
    }
    refresh();
  };

  if (needConfirm) {
    confirmAction(confirmMsg || `Are you sure you want to ${label.toLowerCase()}?`, run);
  } else {
    run();
  }
}

bind('#start-mcp', 'click', () => postAction('/api/start-mcp', 'Start MCP'));
bind('#restart-mcp', 'click', () => postAction('/api/restart/mcp', 'Restart MCP'));
bind('#restart-kernel', 'click', () => postAction('/api/restart/kernel', 'Restart Kernel'));
bind('#restart-all', 'click', () => postAction('/api/restart/all', 'Restart Kernel + MCP', {
  confirm: true,
  confirmMsg: 'Restart the managed kernel and in-kernel MCP? This will interrupt active operations.'
}));

bind('#snapshot-button', 'click', () => postAction('/api/notebook/snapshot', 'Save snapshot'));

// =====================
// Conversion
// =====================

bind('#convert', 'submit', async e => {
  e.preventDefault();
  const btn = $('#convert button[type="submit"]');
  const original = btn.innerHTML;
  btn.innerHTML = '<span class="spinner"></span> Converting…';
  btn.disabled = true;

  $('#convert-result').style.display = 'block';
  $('#convert-result').textContent = 'Converting…';

  try {
    const body = {
      input: $('#pxt-input').value
    };
    const resp = await fetch('/api/convert', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body)
    });
    const r = await resp.json();

    $('#convert-result').textContent = pretty(r);
    if (!resp.ok) throw new Error(r.error || r.error_type || 'Conversion failed');
    // Unified Load control: only files whose output actually exists on disk are
    // loadable.  "skipped" may mean "already on disk" OR "CPU budget wait timed
    // out" (no output); cancelled items have no output either — the backend
    // reports output_exists precisely for each item.
    const loadable = Array.isArray(r.items) ? r.items.filter(x => x.output_exists) : [];
    renderLoadControl(loadable);
    const failures = Array.isArray(r.items) ? r.items.filter(x => x.status === 'failed').length : 0;
    toast(
      failures ? `Conversion completed with ${failures} failure(s)` : 'Conversion completed',
      failures ? 'warning' : 'success'
    );
  } catch (e) {
    $('#convert-result').textContent = String(e);
    toast('Conversion failed', 'error');
  } finally {
    btn.innerHTML = original;
    btn.disabled = false;
  }
});

function renderLoadControl(items) {
  const container = $('#load-actions');
  container.innerHTML = '';
  if (items.length === 0) return;
  const panel = document.createElement('div');
  panel.className = 'load-panel';
  const title = document.createElement('div');
  title.className = 'load-title';
  title.textContent = 'Load files into the notebook';
  const select = document.createElement('select');
  select.id = 'load-select';
  select.multiple = true;
  select.size = Math.min(6, items.length);
  items.forEach(item => {
    const option = document.createElement('option');
    option.value = String(item.output);
    const label = String(item.output).split('/').pop();
    option.textContent = item.status === 'skipped' ? `${label} (already on disk)` : label;    select.appendChild(option);
  });
  const buttons = document.createElement('div');
  buttons.className = 'load-buttons';
  const loadSelected = document.createElement('button');
  loadSelected.type = 'button';
  loadSelected.id = 'load-selected';
  loadSelected.className = 'button';
  loadSelected.textContent = 'Load selected';
  const loadAll = document.createElement('button');
  loadAll.type = 'button';
  loadAll.id = 'load-all';
  loadAll.className = 'button primary';
  loadAll.textContent = 'Load all';
  buttons.append(loadSelected, loadAll);
  panel.append(title, select, buttons);
  container.appendChild(panel);

  const run = async (paths) => {
    if (paths.length === 0) { toast('No files selected', 'warning'); return; }
    $('#convert-result').style.display = 'block';
    $('#convert-result').textContent = 'Loading into notebook…';
    try {
      const resp = await fetch('/api/notebook/load', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ paths })
      });
      const lr = await resp.json();
      $('#convert-result').textContent = pretty(lr);
      const results = lr.results ? Object.values(lr.results) : [];
      const failed = results.filter(v => !v).length;
      toast(failed === 0 ? `Loaded ${results.length} file(s)` : `${failed} file(s) failed`, failed === 0 ? 'success' : 'error');
    } catch (err) {
      $('#convert-result').textContent = String(err);
      toast('Load failed', 'error');
    }
  };
  loadSelected.addEventListener('click', () =>
    run([...select.selectedOptions].map(o => o.value)));
  loadAll.addEventListener('click', () =>
    run(items.map(item => item.output)));
}

bind('#pick-folder', 'click', async () => {
  const b = $('#pick-folder');
  const original = b.innerHTML;
  b.disabled = true;
  b.innerHTML = '<span class="spinner"></span> Selecting…';

  try {
    const r = await fetch('/api/choose-folder', { method: 'POST' }).then(x => x.json());
    if (r.ok) {
      $('#pxt-input').value = r.path;
      toast('Folder selected', 'success');
    } else {
      $('#convert-result').style.display = 'block';
      $('#convert-result').textContent = pretty(r);
      toast('Folder selection failed', 'error');
    }
  } catch (e) {
    $('#convert-result').style.display = 'block';
    $('#convert-result').textContent = String(e);
    toast('Folder selection failed', 'error');
  } finally {
    b.disabled = false;
    b.innerHTML = original;
  }
});

// =====================
// Init
// =====================

refresh();
setInterval(refresh, 5000);
