const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

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
  setEnabled($('#restart-mcp'), hostUp && jupyterReady);
  setEnabled($('#restart-kernel'), hostUp && jupyterReady);
  const lab = $('#open-lab');
  if (lab) {
    lab.href = s.notebook_open_url || '';
    // A connected Comm proves one frontend is open, but does not prove the
    // human can still see it. Keep the same control usable as a reopen action.
    const on = hostUp && jupyterReady && !!lab.href;
    lab.style.pointerEvents = on ? 'auto' : 'none';
    lab.style.opacity = on ? '1' : '0.3';
    const label = $('#open-lab-label');
    if (label) label.textContent = commReady ? 'Reopen Notebook' : 'Open Notebook';
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
      detail.className = 'component-detail';
      detail.textContent = String(c.detail || '');
      article.append(header, title, detail);
      if (c.last_error) {
        const error = document.createElement('span');
        error.className = 'component-error';
        error.textContent = `Last error: ${String(c.last_error)}`;
        article.append(error);
      }
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
// Recent operations (audit chains)
// =====================

function outcomeClass(outcome) {
  const cls = String(outcome || '').toLowerCase();
  const known = { executed: 'executed', saved: 'saved', ok: 'ok', blocked: 'blocked',
                  error: 'error', failed: 'failed', denied: 'denied' };
  return known[cls] || 'unknown';
}

function chainRow(chain, unattached = false) {
  const article = document.createElement('article');
  article.className = 'activity-chain' + (unattached ? ' unattached' : '');

  const head = document.createElement('div');
  head.className = 'chain-head';
  const time = document.createElement('span');
  time.className = 'chain-time';
  const last = chain.last_at || chain.timestamp || '';
  time.textContent = String(last).replace('T', ' ').slice(0, 19);
  const id = document.createElement('span');
  id.className = 'chain-id';
  id.textContent = unattached ? '(legacy event)' : chain.operation_id;
  head.append(time, id);
  (chain.tools || (chain.tool ? [chain.tool] : [])).forEach(tool => {
    const chip = document.createElement('code');
    chip.textContent = String(tool);
    head.append(chip);
  });
  const outcomes = chain.outcomes || (chain.outcome ? [chain.outcome] : []);
  outcomes.forEach(outcome => {
    const badge = document.createElement('span');
    badge.className = 'outcome ' + outcomeClass(outcome);
    badge.textContent = String(outcome);
    head.append(badge);
  });
  article.append(head);

  const meta = document.createElement('div');
  meta.className = 'chain-meta';
  const bits = [];
  if (chain.cell_ids && chain.cell_ids.length) bits.push(`cell ${chain.cell_ids.join(',')}`);
  if (chain.api_ids && chain.api_ids.length) bits.push(`api ${chain.api_ids.join(',')}`);
  if (chain.ticket_id) bits.push(`ticket ${chain.ticket_id}`);
  if (chain.sha256) bits.push(`sha ${String(chain.sha256).slice(0, 16)}`);
  if (chain.target_path) bits.push(`→ ${chain.target_path}`);
  const error = chain.events && chain.events[chain.events.length - 1];
  if (error && error.error) bits.push(`error: ${String(error.error).slice(0, 160)}`);
  meta.textContent = bits.join('  ·  ');
  if (bits.length) article.append(meta);

  if (chain.events && chain.events.length) {
    const events = document.createElement('div');
    events.className = 'chain-events';
    events.textContent = chain.events
      .map(e => `${String(e.timestamp || '').replace('T', ' ').slice(11, 19)} ${e.outcome}`)
      .join(' → ');
    article.append(events);
  }
  return article;
}

function renderActivity(payload) {
  const container = $('#activity');
  const badge = $('#activity-status');
  const chains = (payload && payload.chains) || [];
  const unattached = (payload && payload.unattached) || [];
  if (!container) return;
  container.replaceChildren();
  if (payload && payload.error) {
    const state = document.createElement('div');
    state.className = 'empty-state';
    state.innerHTML = `<div class="empty-title">Activity unavailable</div>`;
    const desc = document.createElement('div');
    desc.className = 'empty-desc';
    desc.textContent = String(payload.error);
    state.append(desc);
    container.append(state);
    if (badge) { badge.className = 'status-badge error'; badge.textContent = 'ERROR'; }
    return;
  }
  if (!chains.length && !unattached.length) {
    container.innerHTML = `
      <div class="empty-state">
        <div class="empty-title">No recent operations</div>
        <div class="empty-desc">Audit chains will appear here once tools are used.</div>
      </div>`;
  } else {
    chains.forEach(chain => container.append(chainRow(chain)));
    unattached.forEach(event => container.append(chainRow(event, true)));
  }
  if (badge) {
    badge.className = 'status-badge ' + (chains.length ? 'ready' : 'unknown');
    badge.textContent = chains.length ? `${chains.length} chains` : 'Empty';
  }
}

async function loadActivity() {
  try {
    const payload = await fetch('/api/activity/recent').then(r => r.json());
    renderActivity(payload);
  } catch (e) {
    renderActivity({ error: String(e) });
  }
}

// =====================
// Refresh with lock & visibility awareness
// =====================

let refreshLock = false;

async function refresh() {
  if (refreshLock) return;
  refreshLock = true;
  try {
    const [s, activity] = await Promise.all([
      fetch('/api/status').then(r => r.json()),
      fetch('/api/activity/recent').then(r => r.json()).catch(() => ({})),
    ]);
    $('#connection').classList.add('up');
    $('#connection-display').classList.add('up');
    renderComponents(s);
    renderActivity(activity);
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
bind('#restart-mcp', 'click', () => postAction('/api/restart/mcp', 'Restart MCP'));
bind('#start-jupyter', 'click', () => postAction('/api/jupyter/start', 'Start Jupyter'));
bind('#stop-jupyter', 'click', () => postAction('/api/jupyter/stop', 'Stop Jupyter', {
  confirm: true,
  confirmMsg: 'Stop JupyterLab and its managed kernel? The dashboard stays up and can restart it.'
}));
bind('#restart-kernel', 'click', () => postAction('/api/restart/kernel', 'Restart Kernel', {
  confirm: true,
  confirmMsg: 'Restart the managed kernel? In-memory analysis variables will be cleared.'
}));

bind('#snapshot-button', 'click', () => postAction('/api/notebook/snapshot', 'Save snapshot'));

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
