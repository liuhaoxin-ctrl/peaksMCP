const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
const pretty = x => JSON.stringify(x, null, 2);
let records = [];

// =====================
// UI Utilities
// =====================

function toast(message, type = 'info', duration = 4000) {
  const container = $('#toast-container');
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  const icons = { success: '◉', error: '◉', warning: '◉', info: '◉' };
  const colors = { 
    success: 'color: var(--accent)', 
    error: 'color: var(--danger)', 
    warning: 'color: var(--warning)', 
    info: 'color: var(--info)' 
  };
  el.innerHTML = `<span style="${colors[type] || colors.info}; font-size: 10px;">${icons[type] || '◉'}</span><span>${message}</span>`;
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

$('#modal-cancel').addEventListener('click', () => {
  $('#confirm-modal').classList.remove('active');
});

$('#confirm-modal').addEventListener('click', (e) => {
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
    container.innerHTML = entries.map(([name, c]) => {
      const stateClass = c.state || 'unknown';
      const stateIcon = {
        ready: '●',
        degraded: '◐',
        error: '●',
        loading: '◐',
        unknown: '○'
      }[stateClass] || '○';
      return `
        <article class="component ${stateClass}">
          <div class="component-header">
            <span class="component-icon">${stateIcon}</span>
            <span class="component-state">${c.state.toUpperCase()}</span>
          </div>
          <b>${name}</b>
          <span>${c.detail || ''}</span>
        </article>`;
    }).join('');
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
    renderComponents(s);
  } catch (e) {
    $('#state').textContent = 'OFFLINE';
    $('#connection').classList.remove('up');
    toast('Connection lost — retrying…', 'error');
  }
}

// =====================
// Images
// =====================

function renderImages(value) {
  const blocks = value?.result || [];
  const container = $('#images');
  const imgs = Array.isArray(blocks) ? blocks.filter(x => x.type === 'image') : [];

  if (imgs.length === 0) {
    container.style.display = 'none';
    return;
  }

  container.style.display = 'grid';
  container.innerHTML = imgs.map(x => 
    `<div class="image-wrapper">
      <img alt="Notebook output" src="data:${x.mimeType};base64,${x.data}">
    </div>`
  ).join('');
}

$('#read-output-button').addEventListener('click', async () => {
  $('#result').style.display = 'block';
  $('#result').textContent = 'Reading output…';
  try {
    const r = await fetch('/api/mcp/tool', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ name: 'notebook_read_active_cell_output', arguments: {} })
    }).then(x => x.json());
    $('#result').textContent = pretty(r);
    renderImages(r);
    toast('Output read', 'success');
  } catch (e) {
    $('#result').textContent = String(e);
    toast('Read failed: ' + e.message, 'error');
  }
});

// =====================
// Search
// =====================

$('#search').addEventListener('submit', async e => {
  e.preventDefault();
  const btn = $('#search button[type="submit"]');
  const original = btn.innerHTML;
  btn.innerHTML = '<span class="spinner"></span> Searching…';
  btn.disabled = true;

  $('#result').style.display = 'block';
  $('#result').textContent = 'Searching…';
  $('#images').style.display = 'none';

  try {
    const body = { name: 'peaks_search_api', arguments: { query: $('#query').value, limit: 8 } };
    const r = await fetch('/api/mcp/tool', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body)
    }).then(x => x.json());

    $('#result').textContent = pretty(r);
    renderImages(r);
    toast('Search completed', 'success');
  } catch (e) {
    $('#result').textContent = String(e);
    toast('Search failed: ' + e.message, 'error');
  } finally {
    btn.innerHTML = original;
    btn.disabled = false;
  }
});

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

$('#start-mcp').addEventListener('click', () => postAction('/api/start-mcp', 'Start MCP'));
$('#restart-mcp').addEventListener('click', () => postAction('/api/restart/mcp', 'Restart MCP'));
$('#restart-kernel').addEventListener('click', () => postAction('/api/restart/kernel', 'Restart Kernel'));
$('#restart-all').addEventListener('click', () => postAction('/api/restart/all', 'Restart All', { 
  confirm: true, 
  confirmMsg: 'Restart all services? This will interrupt any active operations.' 
}));

$('#snapshot-button').addEventListener('click', () => postAction('/api/notebook/snapshot', 'Save snapshot'));

$('#doctor-button').addEventListener('click', async () => {
  const status = $('#doctor-status');
  const pre = $('#doctor');
  status.className = 'status-badge running';
  status.textContent = 'Running';
  pre.textContent = 'Running diagnostics…';

  try {
    const r = await fetch('/api/doctor').then(r => r.json());
    pre.textContent = pretty(r);
    status.className = 'status-badge healthy';
    status.textContent = 'Healthy';
    toast('Doctor diagnostics completed', 'success');
  } catch (e) {
    pre.textContent = String(e);
    status.className = 'status-badge error';
    status.textContent = 'Failed';
    toast('Doctor failed: ' + e.message, 'error');
  }
});

// =====================
// Conversion
// =====================

$('#convert').addEventListener('submit', async e => {
  e.preventDefault();
  const btn = $('#convert button[type="submit"]');
  const original = btn.innerHTML;
  btn.innerHTML = '<span class="spinner"></span> Converting…';
  btn.disabled = true;

  $('#convert-result').style.display = 'block';
  $('#convert-result').textContent = 'Converting…';

  try {
    const body = {
      input: $('#pxt-input').value,
      output: $('#pxt-output').value || null,
      metadata: $('#pxt-metadata').value || null
    };
    const r = await fetch('/api/convert', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body)
    }).then(r => r.json());

    $('#convert-result').textContent = pretty(r);
    toast('Conversion completed', 'success');
  } catch (e) {
    $('#convert-result').textContent = String(e);
    toast('Conversion failed', 'error');
  } finally {
    btn.innerHTML = original;
    btn.disabled = false;
  }
});

$('#pick-folder').addEventListener('click', async () => {
  const b = $('#pick-folder');
  const original = b.innerHTML;
  b.disabled = true;
  b.innerHTML = '<span class="spinner"></span> Selecting…';

  try {
    const r = await fetch('/api/choose-folder', { method: 'POST' }).then(x => x.json());
    if (r.ok) {
      $('#pxt-input').value = r.path;
      $('#pxt-output').value = '';
      $('#pxt-metadata').focus();
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
// Logs
// =====================

function renderLogs() {
  const filter = $('#log-filter').value;
  const filtered = records.filter(x => !filter || x.component === filter);
  const pre = $('#logs');

  if (filtered.length === 0) {
    pre.textContent = filter ? 'No logs for this component.' : 'Awaiting log stream…';
    return;
  }

  pre.textContent = filtered.map(x => `[${x.component}] ${x.message}`).join('\n');
  pre.scrollTop = pre.scrollHeight;

  const values = new Set([...$('#log-filter').options].map(x => x.value));
  for (const c of new Set(records.map(x => x.component))) {
    if (!values.has(c)) $('#log-filter').add(new Option(c, c));
  }
}

$('#log-filter').addEventListener('change', renderLogs);

$('#clear-logs').addEventListener('click', () => {
  records = [];
  renderLogs();
  toast('Log buffer cleared', 'info');
});

$('#export-logs').addEventListener('click', () => {
  const text = records.map(r => `[${r.component}] ${r.message}`).join('\n');
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `peaksMCP-logs-${new Date().toISOString().slice(0,10)}.txt`;
  a.click();
  URL.revokeObjectURL(url);
  toast('Logs exported', 'success');
});

// =====================
// WebSocket
// =====================

const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/logs`);
ws.onopen = () => {
  $('#connection').classList.add('up');
  toast('WebSocket connected', 'success');
};
ws.onclose = () => {
  $('#connection').classList.remove('up');
  toast('WebSocket disconnected', 'warning');
};
ws.onmessage = e => {
  const d = JSON.parse(e.data);
  records = records.concat(d.logs || []).slice(-2000);
  renderLogs();
};

// =====================
// Init
// =====================

fetch('/api/profiles').then(r => r.json()).then(x => {
  $('#profiles').textContent = pretty(x);
}).catch(e => {
  $('#profiles').textContent = 'Failed to load profiles';
});

refresh();
setInterval(refresh, 5000);

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
    e.preventDefault();
    $('#query').focus();
  }
});
