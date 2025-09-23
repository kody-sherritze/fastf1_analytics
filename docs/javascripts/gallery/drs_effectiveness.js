/* DRS Effectiveness — Interactive Renderer (Plotly)
   - Base-URL aware for MkDocs/GitHub Pages
   - Robust messages when data/Plotly missing or no windows found
   - Advanced parameters + overlay toggles

   Expects a container like:
   <div class="drs-widget"
        data-base="{{ base_url }}"
        data-index="{{ base_url }}/assets/data/drs/index.json"
        data-default-year="2025"
        data-default-event="italian_grand_prix"
        data-default-driver="VER"></div>
*/

/* global Plotly, FFA_Gallery */
(function () {
    // Minimal helpers; use global FFA_Gallery if present, otherwise fall back.
    const { el, debounce, clear } = window.FFA_Gallery || {
      el: (t, a = {}, c = []) => {
        const e = document.createElement(t);
        Object.entries(a).forEach(([k, v]) => {
          if (k === 'class') e.className = v;
          else if (k.startsWith('data-')) e.setAttribute(k, v);
          else e[k] = v;
        });
        [].concat(c).forEach(x => {
          if (x == null) return;
          if (typeof x === 'string') e.appendChild(document.createTextNode(x));
          else e.appendChild(x);
        });
        return e;
      },
      debounce: (fn, ms) => { let t = null; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; },
      clear: (n) => { while (n.firstChild) n.removeChild(n.firstChild); },
    };
  
    // Join a possibly-relative path against an mkdocs {{ base_url }} value.
    function joinUrl(base, path) {
      if (!path) return '';
      if (/^https?:\/\//i.test(path)) return path;   // already absolute
      base = (base || '').replace(/\/+$/, '');       // trim trailing slash
      const p = String(path);
      if (!base || base === '.' || base === './') return p.replace(/^\/+/, '');
      if (p.startsWith('/')) return base + p;        // base like '..', path '/assets/...'
      return base + '/' + p.replace(/^\/+/, '');
    }
  
    const DEFAULTS = {
      n_points: 200,
      accel_threshold_kmh_s: -8.0,
      sustain_sec: 0.30,
      min_open_ratio_on: 0.15,
      max_open_ratio_off: 0.02,
      show_activation: true,
      show_turns: true,
    };
  
    function msg(container, text) {
      clear(container);
      container.appendChild(el('div', { class: 'drs-msg', style: 'padding:8px;opacity:0.85;' }, [text]));
    }
  
    function uniqueId() { return 'drs-' + Math.random().toString(36).slice(2, 9); }
    async function fetchJSON(url) {
      const r = await fetch(url, { cache: 'no-store' });
      if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
      return r.json();
    }
  
    // ---------- Analysis helpers (JS mirror of Python) ----------
    function accelSeries(speedKmh, tSec) {
      const a = new Array(speedKmh.length).fill(0);
      for (let i = 1; i < speedKmh.length; i++) {
        const ds = speedKmh[i] - speedKmh[i - 1];
        const dt = Math.max(1e-6, tSec[i] - tSec[i - 1]);
        a[i] = ds / dt; // km/h/s
      }
      return a;
    }
  
    function brakeMask(accel, tSec, sustainSec, threshold) {
      const n = accel.length, mask = new Array(n).fill(false);
      let start = null;
      for (let i = 0; i < n; i++) {
        if (accel[i] < threshold) { if (start == null) start = i; }
        else if (start != null) {
          if (tSec[i] - tSec[start] >= sustainSec) for (let k = start; k < i; k++) mask[k] = true;
          start = null;
        }
      }
      if (start != null && (tSec[n - 1] - tSec[start] >= sustainSec)) for (let k = start; k < n; k++) mask[k] = true;
      return mask;
    }
  
    function segmentsFromBrakeMask(mask, minLen = 8) {
      const segs = []; let s = null;
      for (let i = 0; i < mask.length; i++) {
        if (!mask[i]) { if (s == null) s = i; }
        else if (s != null) { const e = i - 1; if (e - s + 1 >= minLen) segs.push({ start: s, end: e }); s = null; }
      }
      if (s != null) { const e = mask.length - 1; if (e - s + 1 >= minLen) segs.push({ start: s, end: e }); }
      return segs;
    }
  
    function selectDRSStraightIndices(lap, params) {
      const t = lap.t_s, v = lap.speed_kmh, d = lap.dist_m, drs = lap.drs || [];
      if (!t || !v || !d || t.length < 16) return null;
  
      const a = accelSeries(v, t);
      const brake = brakeMask(a, t, params.sustain_sec, params.accel_threshold_kmh_s);
      const segs = segmentsFromBrakeMask(brake, 8);
      if (!segs.length) return null;
  
      // Choose segment with strongest DRS presence (tie-break by length)
      let bestIdx = -1, bestScore = -Infinity;
      for (let i = 0; i < segs.length; i++) {
        const { start, end } = segs[i];
        let open = 0, total = 0;
        for (let k = start; k <= end; k++) { open += (drs[k] || 0); total += 1; }
        const ratio = total ? open / total : 0;
        const score = ratio * (end - start + 1);
        if (score > bestScore) { bestScore = score; bestIdx = i; }
      }
      if (bestIdx < 0) {
        bestIdx = segs.map((s, i) => [i, s.end - s.start]).sort((a,b)=>b[1]-a[1])[0][0];
      }
      const seg = segs[bestIdx];
      return [seg.start, seg.end];
    }
  
    function resampleNormalized(dist, speed, drs, i0, i1, nPoints) {
      const D = dist.slice(i0, i1 + 1), S = speed.slice(i0, i1 + 1), R = (drs || []).slice(i0, i1 + 1);
      const pairs = D.map((dd, i) => [dd, S[i], R[i] || 0]).sort((a, b) => a[0] - b[0]);
      const d = pairs.map(p => p[0]), s = pairs.map(p => p[1]), r = pairs.map(p => p[2]);
      if (!d.length) return { x: [], speed: [], drs: [] };
      const d0 = d[0], d1 = d[d.length - 1], span = Math.max(1e-6, d1 - d0);
      const x = new Array(nPoints), s_i = new Array(nPoints), r_i = new Array(nPoints);
      for (let j = 0; j < nPoints; j++) {
        const a = j / (nPoints - 1), target = d0 + a * span;
        let k = 1; while (k < d.length && d[k] < target) k++;
        const k0 = Math.max(0, k - 1), k1 = Math.min(d.length - 1, k);
        const dd = (d[k1] - d[k0]) > 1e-9 ? (target - d[k0]) / (d[k1] - d[k0]) : 0;
        s_i[j] = s[k0] + dd * (s[k1] - s[k0]);
        const rInterp = r[k0] + dd * (r[k1] - r[k0]); // soft interpolate for labeling
        r_i[j] = rInterp >= 0.5 ? 1 : 0;
        x[j] = a;
      }
      return { x, speed: s_i, drs: r_i, d0, d1 };
    }
  
    function timeFromResampled(spanMeters, speedKmhArr) {
      const n = speedKmhArr.length; if (n < 2) return NaN;
      const v_ms = speedKmhArr.map(v => v / 3.6);
      let t = 0, ds = spanMeters * (1 / (n - 1));
      for (let i = 1; i < n; i++) {
        const vMid = 0.5 * (v_ms[i] + v_ms[i - 1]);
        t += ds / Math.max(0.1, vMid);
      }
      return t;
    }
  
    function selectBestWindows(data, params) {
      let bestOn = null, bestOff = null;
      for (const lap of (data.laps || [])) {
        const idx = selectDRSStraightIndices(lap, params);
        if (!idx) continue;
        const [i0, i1] = idx;
        const res = resampleNormalized(lap.dist_m, lap.speed_kmh, lap.drs || [], i0, i1, params.n_points);
        if (!res.x.length) continue;
        const openRatio = res.drs.reduce((a, b) => a + (b ? 1 : 0), 0) / res.drs.length;
        const span = Math.max(1e-6, res.d1 - res.d0);
        const tSec = timeFromResampled(span, res.speed);
        if (!isFinite(tSec)) continue;
        if (openRatio >= params.min_open_ratio_on) {
          if (!bestOn || tSec < bestOn.t_sec) bestOn = { ...res, t_sec: tSec, open_ratio: openRatio };
        } else if (openRatio <= params.max_open_ratio_off) {
          if (!bestOff || tSec < bestOff.t_sec) bestOff = { ...res, t_sec: tSec, open_ratio: openRatio };
        }
      }
      return { bestOn, bestOff };
    }
  
    function cumulativeTimeGain(on, off) {
      const n = on.speed.length, span = Math.max(1e-6, on.d1 - on.d0), ds = span * (1 / (n - 1));
      const vOn = on.speed.map(v => v / 3.6), vOff = off.speed.map(v => v / 3.6);
      const cum = new Array(n).fill(0);
      for (let i = 1; i < n; i++) {
        const dOn = ds / Math.max(0.1, 0.5 * (vOn[i] + vOn[i - 1]));
        const dOff = ds / Math.max(0.1, 0.5 * (vOff[i] + vOff[i - 1]));
        cum[i] = cum[i - 1] + (dOff - dOn);
      }
      return cum;
    }
  
    function drawPlot(container, results, toggles) {
      if (typeof Plotly === 'undefined') {
        msg(container, 'Plotly failed to load. Check extra_javascript in mkdocs.yaml.');
        return;
      }
      const { bestOn, bestOff } = results;
      const traces = [];
      const layout = {
        margin: { l: 55, r: 55, t: 30, b: 45 },
        xaxis: { title: 'Normalized distance', range: [0, 1], showgrid: true, zeroline: false },
        yaxis: { title: 'Speed (km/h)', showgrid: true, zeroline: false },
        yaxis2: { title: 'ΔTime (s)', overlaying: 'y', side: 'right' },
        legend: { orientation: 'h', y: -0.2 },
        hovermode: 'x unified',
        showlegend: true,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
      };
  
      if (bestOn) traces.push({ x: bestOn.x, y: bestOn.speed, mode: 'lines', name: 'Fastest DRS ON', line: { width: 2.2 } });
      if (bestOff) traces.push({ x: bestOff.x, y: bestOff.speed, mode: 'lines', name: 'Fastest DRS OFF', line: { width: 2.2 } });
  
      if (bestOn && bestOff) {
        const cum = cumulativeTimeGain(bestOn, bestOff);
        traces.push({ x: bestOn.x, y: cum, mode: 'lines', name: 'Cumulative Δt', yaxis: 'y2', line: { width: 2.0, dash: 'dot' } });
      }
  
      if (!traces.length) {
        msg(container, 'No valid DRS windows found for the current selection/thresholds. Try another driver/event or relax thresholds in Advanced Parameters.');
        return;
      }
  
      if (toggles.show_activation && bestOn) {
        const kAct = bestOn.drs.findIndex(v => !!v);
        if (kAct >= 0) {
          traces.push({
            x: [bestOn.x[kAct], bestOn.x[kAct]],
            y: [0, Math.max(...bestOn.speed) * 1.05],
            mode: 'lines',
            name: 'Activation',
            line: { width: 1.4, dash: 'dash' },
            hoverinfo: 'skip'
          });
        }
      }
      if (toggles.show_turns && bestOn) {
        const s = bestOn.x[0], e = bestOn.x[bestOn.x.length - 1], ymax = Math.max(...bestOn.speed) * 1.05;
        traces.push({ x: [s, s], y: [0, ymax], mode: 'lines', name: 'Turn start', line: { width: 1.0, dash: 'dot' }, hoverinfo: 'skip' });
        traces.push({ x: [e, e], y: [0, ymax], mode: 'lines', name: 'Turn end', line: { width: 1.0, dash: 'dot' }, hoverinfo: 'skip' });
      }
  
      Plotly.react(container, traces, layout, { responsive: true, displayModeBar: false });
    }
  
    // ---------- UI ----------
    function buildControls(widget) {
      const controls = el('div', { class: 'drs-controls' });
  
      const row1 = el('div', { class: 'drs-row' });
      const yearSel = el('select', { class: 'drs-year' });
      const eventSel = el('select', { class: 'drs-event' });
      const driverSel = el('select', { class: 'drs-driver' });
      row1.appendChild(el('label', {}, ['Year ', yearSel]));
      row1.appendChild(el('label', {}, ['Event ', eventSel]));
      row1.appendChild(el('label', {}, ['Driver ', driverSel]));
  
      const row2 = el('div', { class: 'drs-row' });
      const toggleAct = el('input', { type: 'checkbox', class: 'drs-toggle-activation' }); toggleAct.checked = DEFAULTS.show_activation;
      const toggleTurns = el('input', { type: 'checkbox', class: 'drs-toggle-turns' }); toggleTurns.checked = DEFAULTS.show_turns;
      row2.appendChild(el('label', {}, [toggleAct, ' Activation line']));
      row2.appendChild(el('label', {}, [toggleTurns, ' Turn markers']));
  
      const adv = el('details', { class: 'drs-advanced' });
      const advSummary = el('summary', {}, ['Advanced parameters']);
      const advWrap = el('div', { class: 'drs-adv-wrap' });
      const npIn = el('input', { type: 'number', step: 10, min: 50, max: 2000, value: DEFAULTS.n_points, class: 'drs-npoints' });
      const accIn = el('input', { type: 'number', step: 0.1, value: DEFAULTS.accel_threshold_kmh_s, class: 'drs-accel' });
      const susIn = el('input', { type: 'number', step: 0.05, value: DEFAULTS.sustain_sec, class: 'drs-sustain' });
      const onIn  = el('input', { type: 'number', step: 0.01, min: 0, max: 1, value: DEFAULTS.min_open_ratio_on, class: 'drs-minon' });
      const offIn = el('input', { type: 'number', step: 0.01, min: 0, max: 1, value: DEFAULTS.max_open_ratio_off, class: 'drs-maxoff' });
  
      function row(label, input, suffix='') {
        const r = el('div', { class: 'drs-adv-row' });
        r.appendChild(el('label', {}, [label, ' ', input, suffix]));
        return r;
      }
      advWrap.appendChild(row('n_points', npIn));
      advWrap.appendChild(row('accel_threshold_kmh_s', accIn, ' (km/h/s)'));
      advWrap.appendChild(row('sustain_sec', susIn, ' (s)'));
      advWrap.appendChild(row('min_open_ratio_on', onIn));
      advWrap.appendChild(row('max_open_ratio_off', offIn));
      adv.appendChild(advSummary); adv.appendChild(advWrap);
  
      controls.appendChild(row1); controls.appendChild(row2); controls.appendChild(adv);
      return { controls, yearSel, eventSel, driverSel, toggleAct, toggleTurns, npIn, accIn, susIn, onIn, offIn };
    }
  
    async function initWidget(widget) {
      // Base URL coming from mkdocs {{ base_url }}
      const base = (widget.dataset.base || '').trim();
  
      // Index URL (base-aware)
      const indexAttr = widget.dataset.index || 'assets/data/drs/index.json';
      const indexUrl = joinUrl(base, indexAttr);
  
      const container = el('div', { id: uniqueId(), class: 'drs-plot', style: 'height:420px;border:1px solid transparent;' });
      widget.appendChild(container);
  
      const ctl = buildControls(widget);
      widget.insertBefore(ctl.controls, container);
  
      // Load index.json
      let index;
      try {
        index = await fetchJSON(indexUrl);
      } catch (e) {
        console.error(e);
        msg(container, 'Failed to load DRS index.json. Ensure docs/assets/data/drs/index.json exists and is served.');
        return;
      }
      const sessions = Array.isArray(index.sessions) ? index.sessions : [];
      if (!sessions.length) {
        msg(container, 'No DRS datasets found. Generate JSON with the exporter and commit them.');
        return;
      }
  
      // Group sessions by year
      const byYear = new Map();
      for (const s of sessions) {
        if (!byYear.has(s.year)) byYear.set(s.year, []);
        byYear.get(s.year).push(s);
      }
      const years = Array.from(byYear.keys()).sort((a,b)=>a-b);
      ctl.yearSel.innerHTML = years.map(y => `<option value="${y}">${y}</option>`).join('');
  
      function firstOr(arr, fallback=null) { return (arr && arr.length) ? arr[0] : fallback; }
  
      function fillEvents() {
        const y = parseInt(ctl.yearSel.value, 10);
        const evs = (byYear.get(y) || []);
        ctl.eventSel.innerHTML = evs.map(s => `<option value="${s.event_slug}">${s.event_name} (${s.session})</option>`).join('');
        if (!ctl.eventSel.value && evs.length) ctl.eventSel.value = evs[0].event_slug;
        fillDrivers();
      }
      function fillDrivers() {
        const y = parseInt(ctl.yearSel.value, 10);
        const slug = ctl.eventSel.value;
        const sess = sessions.find(s => s.event_slug === slug && s.year === y);
        const drivers = (sess && Array.isArray(sess.drivers)) ? sess.drivers : [];
        ctl.driverSel.innerHTML = drivers.map(d => `<option value="${d}">${d}</option>`).join('');
        if (!ctl.driverSel.value && drivers.length) ctl.driverSel.value = drivers[0];
      }
  
      // Initialize selectors with resilient defaults
      const attrYear = parseInt(widget.dataset.defaultYear || '', 10);
      ctl.yearSel.value = years.includes(attrYear) ? String(attrYear) : String(firstOr(years, ''));
      fillEvents();
  
      const attrEvent = (widget.dataset.defaultEvent || '').toLowerCase();
      if (attrEvent) {
        const yNow = parseInt(ctl.yearSel.value, 10);
        const evs = (byYear.get(yNow) || []);
        if (evs.some(s => s.event_slug === attrEvent)) ctl.eventSel.value = attrEvent;
      }
      fillDrivers();
  
      const attrDriver = (widget.dataset.defaultDriver || '').toUpperCase();
      if (attrDriver) {
        const opts = Array.from(ctl.driverSel.options).map(o => o.value.toUpperCase());
        if (opts.includes(attrDriver)) ctl.driverSel.value = attrDriver;
      }
  
      async function refresh() {
        try {
          const y = parseInt(ctl.yearSel.value, 10);
          const slug = ctl.eventSel.value;
          const sess = sessions.find(s => s.event_slug === slug && s.year === y);
          if (!sess) { msg(container, 'No session for current selection.'); return; }
  
          const drv = ctl.driverSel.value;
          const urlPath = (sess.files && sess.files[drv]) || null;
          if (!urlPath) { msg(container, 'No data file for selected driver.'); return; }
  
          const url = joinUrl(base, urlPath);
          let data;
          try {
            data = await fetchJSON(url);
          } catch (e) {
            console.error(e);
            msg(container, `Failed to load data: ${url}`);
            return;
          }
  
          const params = {
            n_points: parseInt(ctl.npIn.value, 10) || DEFAULTS.n_points,
            accel_threshold_kmh_s: parseFloat(ctl.accIn.value) || DEFAULTS.accel_threshold_kmh_s,
            sustain_sec: parseFloat(ctl.susIn.value) || DEFAULTS.sustain_sec,
            min_open_ratio_on: parseFloat(ctl.onIn.value) || DEFAULTS.min_open_ratio_on,
            max_open_ratio_off: parseFloat(ctl.offIn.value) || DEFAULTS.max_open_ratio_off,
          };
          const toggles = { show_activation: ctl.toggleAct.checked, show_turns: ctl.toggleTurns.checked };
          const results = selectBestWindows(data, params);
          drawPlot(container, results, toggles);
        } catch (err) {
          console.error(err);
          msg(container, 'Unexpected error rendering DRS plot. See console for details.');
        }
      }
  
      const refreshDebounced = debounce(refresh, 80);
      ctl.yearSel.addEventListener('change', refresh);
      ctl.eventSel.addEventListener('change', refresh);
      ctl.driverSel.addEventListener('change', refresh);
      ctl.toggleAct.addEventListener('change', refreshDebounced);
      ctl.toggleTurns.addEventListener('change', refreshDebounced);
      ctl.npIn.addEventListener('input', refreshDebounced);
      ctl.accIn.addEventListener('input', refreshDebounced);
      ctl.susIn.addEventListener('input', refreshDebounced);
      ctl.onIn.addEventListener('input', refreshDebounced);
      ctl.offIn.addEventListener('input', refreshDebounced);
  
      await refresh();
    }
  
    function bootstrap() {
      document.querySelectorAll('.drs-widget').forEach(w =>
        initWidget(w).catch(err => {
          console.error('DRS widget failed:', err);
          w.appendChild(document.createTextNode('Interactive DRS widget failed to load.'));
        })
      );
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bootstrap);
    else bootstrap();
  })();
  