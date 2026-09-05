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
  const existing = [...container.children].find(t => t.dataset.msg === message && t.dataset.type === type);
  if (existing) {
    existing.remove();
  }

  const el = document.createElement('div');
  el.className = `toast ${type}`;
  el.dataset.msg = message;
  el.dataset.type = type;
  const icons = { success: '●', error: '●', warning: '●', info: '●' };
  const colors = {
    success: 'var(--accent)',
    error: 'var(--danger)',
    warning: 'var(--warning)',
    info: 'var(--info)'
  };
  const icon = document.createElement('span');
  icon.style.color = colors[type] || colors.info;
  icon.style.fontSize = '9px';
  icon.textContent = icons[type] || '●';
  const text = document.createElement('span');
  text.textContent = String(message);
  el.append(icon, text);
  container.appendChild(el);

  el.addEventListener('click', () => {
    el.style.animation = 'toastOut 0.2s ease forwards';
    setTimeout(() => el.remove(), 200);
  });

  setTimeout(() => {
    if (el.parentNode) {
      el.style.animation = 'toastOut 0.2s ease forwards';
      setTimeout(() => el.remove(), 200);
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
// Collapsible result panels
// =====================

function initResultToggles() {
  document.querySelectorAll('.result-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.dataset.target;
      const panel = document.getElementById(targetId);
      if (!panel) return;
      const isCollapsed = panel.classList.toggle('collapsed');
      btn.textContent = isCollapsed ? '+' : '−';
    });
  });
}

function showResultPanel(id) {
  const panel = document.getElementById(id);
  if (!panel) return;
  panel.style.display = 'flex';
  panel.classList.remove('collapsed');
  const btn = panel.querySelector('.result-toggle');
  if (btn) btn.textContent = '−';
}

function hideResultPanel(id) {
  const panel = document.getElementById(id);
  if (!panel) return;
  panel.style.display = 'none';
}

// =====================
// Controls
// =====================

function setEnabled(btn, enabled) {
  if (!btn) return;
  btn.disabled = !enabled;
  btn.style.opacity = enabled ? '1' : '0.3';
  btn.style.cursor = enabled ? 'pointer' : 'not-allowed';
  btn.style.pointerEvents = enabled ? 'auto' : 'none';
}

function renderControls(s) {
  const mcpReady = (s.components?.mcp?.state) === 'ready';
  const commReady = (s.components?.comm?.state) === 'ready';
  let jupyterState = s.jupyter_state;
  if (!jupyterState) {
    const cs = s.components?.jupyter?.state;
    jupyterState = cs === 'ready' ? 'running' : cs === 'starting' ? 'starting' : 'stopped';
  }
  // The stop/start buttons follow the *declared* group state: after clicking
  // Start Jupyter the group is "starting" and neither button is live until the
  // host reports "running" (HTTP up + managed kernel session exists).
  const jupyterReady = jupyterState === 'running';
  const hostUp = s.supervisor_running !== false;
  setEnabled($('#start-jupyter'), hostUp && jupyterState === 'stopped');
  setEnabled($('#stop-jupyter'), hostUp && jupyterReady);
  setEnabled($('#start-mcp'), hostUp && jupyterReady && !mcpReady);
  setEnabled($('#stop-mcp'), hostUp && mcpReady);
  const lab = $('#open-lab');
  if (lab) {
    lab.href = s.notebook_open_url || '';
    // Open-once semantics: usable only while Jupyter is ready AND no frontend
    // session is attached yet; once the Comm bridge connects, the managed
    // notebook is already open, so the button grays out (same pattern as
    // Start MCP after the MCP is started).
    const on = hostUp && jupyterReady && !commReady && !!lab.href;
    lab.style.pointerEvents = on ? 'auto' : 'none';
    lab.style.opacity = on ? '1' : '0.3';
  }
}

// =====================
// Components & Stats
// =====================

function renderComponents(s) {
  $('#profile').textContent = `profile: ${s.profile}`;
  $('#profile-display').textContent = `profile: ${s.profile}`;
  $('#state').textContent = (s.aggregate || s.status || '?').toUpperCase();

  const stateLarge = $('#state-large');
  const agg = s.aggregate || s.status || 'unknown';
  stateLarge.textContent = agg.toUpperCase();
  stateLarge.className = `state-large ${agg}`;

  const container = $('#components');
  const comps = s.components || {};
  const entries = Object.entries(comps);

  const total = entries.length;
  const ready = entries.filter(([, c]) => c.state === 'ready').length;
  const errors = entries.filter(([, c]) => c.state === 'error').length;
  $('#stat-total').textContent = total;
  $('#stat-ready').textContent = ready;
  $('#stat-error').textContent = errors;

  if (entries.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">◻</div>
        <div class="empty-title">No components</div>
        <div class="empty-desc">Awaiting MCP service handshake…</div>
      </div>`;
  } else {
    container.replaceChildren();
    const allowedStates = new Set(['ready', 'degraded', 'error', 'loading', 'starting', 'stopped', 'unknown']);
    entries.forEach(([name, c]) => {
      const stateClass = allowedStates.has(c.state) ? c.state : 'unknown';
      const stateIcon = {
        ready: '●',
        degraded: '◐',
        error: '●',
        loading: '◐',
        starting: '◐',
        stopped: '○',
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

  const badge = $('#components-status');
  if (badge) {
    badge.className = `status-badge ${agg}`;
    badge.textContent = agg.toUpperCase();
  }

  renderControls(s);
}

// =====================
// Refresh with lock & visibility awareness
// =====================

let refreshLock = false;

async function refresh() {
  if (refreshLock) return;
  refreshLock = true;
  try {
    const s = await fetch('/api/status').then(r => r.json());
    $('#connection').classList.add('up');
    $('#connection-display').classList.add('up');
    renderComponents(s);
  } catch (e) {
    $('#state').textContent = 'OFFLINE';
    $('#state-large').textContent = 'OFFLINE';
    $('#state-large').className = 'state-large error';
    $('#connection').classList.remove('up');
    $('#connection-display').classList.remove('up');
    toast('Connection lost — retrying…', 'error');
  } finally {
    refreshLock = false;
  }
}

// =====================
// Actions
// =====================

async function postAction(path, label, options = {}) {
  const { confirm: needConfirm, confirmMsg } = options;

  const run = async () => {
    const fail = (msg, detail) => {
      toast(detail ? `${msg}: ${detail}` : msg, 'error');
      refresh();
    };
    try {
      const resp = await fetch(path, { method: 'POST' });
      const r = await resp.json();
      if (!resp.ok) {
        fail(`${label} failed`, (r && (r.error || r.detail)) || `HTTP ${resp.status}`);
        return;
      }
      if (r && (r.ready === false || r.ok === false || r.loaded === false || r.error)) {
        fail(`${label} failed (partial or unsuccessful)`, r.error || r.detail);
        return;
      }
      toast(`${label} completed`, 'success');
    } catch (e) {
      toast(`${label} failed: ${e}`, 'error');
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
bind('#stop-mcp', 'click', () => postAction('/api/mcp/stop', 'Stop MCP'));
bind('#start-jupyter', 'click', () => postAction('/api/jupyter/start', 'Start Jupyter'));
bind('#stop-jupyter', 'click', () => postAction('/api/jupyter/stop', 'Stop Jupyter', {
  confirm: true,
  confirmMsg: 'Stop JupyterLab and its managed kernel? The dashboard stays up and can restart it.'
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

  showResultPanel('convert-result-panel');
  $('#convert-result').textContent = 'Converting…';

  try {
    const body = { input: $('#pxt-input').value };
    const resp = await fetch('/api/convert', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body)
    });
    const r = await resp.json();

    $('#convert-result').textContent = pretty(r);
    if (!resp.ok) throw new Error(r.error || r.error_type || 'Conversion failed');

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
    option.textContent = item.status === 'skipped' ? `${label} (already on disk)` : label;
    select.appendChild(option);
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
      showResultPanel('convert-result-panel');
      $('#convert-result').textContent = pretty(r);
      toast('Folder selection failed', 'error');
    }
  } catch (e) {
    showResultPanel('convert-result-panel');
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

initResultToggles();
refresh();
let refreshTimer = setInterval(refresh, 5000);

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    clearInterval(refreshTimer);
    refreshTimer = null;
  } else {
    refresh();
    // Re-create the poller only when none is running: recreating on every
    // return to the tab would accumulate untracked intervals that a later
    // hide could never clear (only the original id was remembered).
    if (refreshTimer === null) {
      refreshTimer = setInterval(refresh, 5000);
    }
  }
});