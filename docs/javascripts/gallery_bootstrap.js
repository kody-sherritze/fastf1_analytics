/* Gallery Bootstrap Helpers (shared across widgets)
   Place at: docs/javascripts/gallery_bootstrap.js

   Exposes a small utility API on window.FFA_Gallery:
     - el(tag, attrs, children)       → DOM helper
     - clear(node)                    → remove all children
     - debounce(fn, ms)               → debounce wrapper
     - joinUrl(base, path)            → base-URL aware path join (MkDocs {{ base_url }})
     - setOptions(select, items, ...) → populate <select> options
     - first(arr, fallback)           → first item or fallback
     - formatDuration(sec)            → "0.123 s" style
     - formatNumber(n, digits)        → number with fixed digits
     - onChange(el, fn, debounceMs)   → attach change/input listeners
*/

(function () {
    if (window.FFA_Gallery) return; // Don't redefine if already loaded
  
    function el(tag, attrs = {}, children = []) {
      const e = document.createElement(tag);
      if (attrs && typeof attrs === 'object') {
        Object.entries(attrs).forEach(([k, v]) => {
          if (k === 'class') {
            e.className = v;
          } else if (k === 'style' && typeof v === 'object') {
            Object.assign(e.style, v);
          } else if (k.startsWith('data-')) {
            e.setAttribute(k, v);
          } else if (k === 'html') {
            e.innerHTML = v;
          } else {
            try { e[k] = v; } catch { e.setAttribute(k, v); }
          }
        });
      }
      const kids = Array.isArray(children) ? children : [children];
      for (const c of kids) {
        if (c == null) continue;
        if (typeof c === 'string') e.appendChild(document.createTextNode(c));
        else e.appendChild(c);
      }
      return e;
    }
  
    function clear(node) {
      while (node && node.firstChild) node.removeChild(node.firstChild);
    }
  
    function debounce(fn, ms) {
      let t = null;
      return (...args) => {
        if (t) clearTimeout(t);
        t = setTimeout(() => fn(...args), ms);
      };
    }
  
    // Join a possibly-relative path against an mkdocs {{ base_url }} value.
    function joinUrl(base, path) {
      if (!path) return '';
      if (/^https?:\/\//i.test(path)) return path;   // absolute URL stays as-is
      base = (base || '').replace(/\/+$/, '');       // trim trailing slash from base
      const p = String(path);
      if (!base || base === '.' || base === './') return p.replace(/^\/+/, '');
      if (p.startsWith('/')) return base + p;        // base like '..', path '/assets/...'
      return base + '/' + p.replace(/^\/+/, '');
    }
  
    function setOptions(select, items, getValue = x => x, getLabel = x => String(x), selectedValue = null) {
      const opts = (items || []).map(item => {
        const val = getValue(item);
        const lbl = getLabel(item);
        return `<option value="${String(val)}">${String(lbl)}</option>`;
      }).join('');
      select.innerHTML = opts;
      if (selectedValue != null) {
        const sv = String(selectedValue);
        const has = Array.from(select.options).some(o => String(o.value) === sv);
        if (has) select.value = sv;
      }
    }
  
    function first(arr, fallback = null) {
      return (Array.isArray(arr) && arr.length) ? arr[0] : fallback;
    }
  
    function formatDuration(sec) {
      if (sec == null || !isFinite(sec)) return '–';
      return `${Number(sec).toFixed(3)} s`;
    }
  
    function formatNumber(n, digits = 2) {
      if (n == null || !isFinite(n)) return '–';
      return Number(n).toFixed(digits);
    }
  
    function onChange(elm, fn, debounceMs = 0) {
      const handler = debounceMs > 0 ? debounce(fn, debounceMs) : fn;
      const types = ['change', 'input'];
      types.forEach(t => elm.addEventListener(t, handler));
      return () => types.forEach(t => elm.removeEventListener(t, handler));
    }
  
    window.FFA_Gallery = {
      el,
      clear,
      debounce,
      joinUrl,
      setOptions,
      first,
      formatDuration,
      formatNumber,
      onChange,
    };
  })();
  