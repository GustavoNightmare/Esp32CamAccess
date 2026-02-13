async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({}));
  return { ok: r.ok, status: r.status, json: j };
}

function fmtTs(ts) {
  if (!ts) return '—';
  const d = new Date(ts * 1000);
  return d.toLocaleString();
}

function setBadge(el, allowed) {
  el.classList.remove('ok', 'bad', 'wait');
  if (allowed === true) {
    el.textContent = 'ENTRA';
    el.classList.add('ok');
  } else if (allowed === false) {
    el.textContent = 'NO ENTRA';
    el.classList.add('bad');
  } else {
    el.textContent = '—';
    el.classList.add('wait');
  }
}

async function refreshStatus() {
  const { ok, json } = await fetchJSON('/api/status');
  if (!ok) return;

  document.getElementById('found').textContent = json.found_face ? 'Sí' : 'No';
  document.getElementById('name').textContent = json.name ?? '—';
  document.getElementById('cosine').textContent = (json.cosine ?? 0).toFixed(3);
  document.getElementById('detconf').textContent = (json.det_conf ?? 0).toFixed(2);
  document.getElementById('event').textContent = json.event_id ?? 0;
  document.getElementById('ts').textContent = fmtTs(json.ts);

  setBadge(document.getElementById('allowed'), json.allowed);
}

function renderPeople(people) {
  const el = document.getElementById('people');
  el.innerHTML = '';
  const entries = Object.entries(people || {});
  if (entries.length === 0) {
    el.textContent = 'DB vacía. Ve a "Enroll".';
    return;
  }

  for (const [name, count] of entries) {
    const div = document.createElement('div');
    div.className = 'pill';
    div.textContent = `${name} (${count})`;
    el.appendChild(div);
  }
}

async function loadDb() {
  const { ok, json } = await fetchJSON('/api/db');
  if (!ok) return;
  renderPeople(json.people);
}

async function reloadDb() {
  const { ok, json } = await fetchJSON('/api/reload_db', { method: 'POST' });
  if (!ok) return;
  renderPeople(json.people);
}

document.getElementById('reloadDb').addEventListener('click', reloadDb);

// Loop
loadDb();
setInterval(refreshStatus, 400);
