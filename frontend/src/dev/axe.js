/**
 * Development-only accessibility auditing with axe-core.
 *
 * Never reaches production: `main.jsx` imports this inside an `import.meta.env.DEV`
 * branch, which Vite replaces with `false` when building, so Rollup removes the whole
 * import. Confirm with `npm run build && npm run smoke` -- the smoke test asserts the
 * built bundle contains no axe.
 *
 * What it checks: WCAG 2.0/2.1 level A and AA rules against the live DOM, re-run on
 * every route change (debounced, because React re-renders constantly in dev).
 */

const RULE_TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa', 'best-practice'];

/** Every route in the app, for window.__axeScan(). */
const ROUTES = ['#/', '#/trip', '#/map', '#/forecast', '#/replay', '#/model', '#/system'];

const IMPACT_STYLE = {
  critical: 'color:#A3231B;font-weight:700',
  serious: 'color:#D4581F;font-weight:700',
  moderate: 'color:#B87D04',
  minor: 'color:#50605A',
};

let timer = null;
let lastSignature = '';

async function run(axe) {
  try {
    const results = await axe.run(document, {
      runOnly: { type: 'tag', values: RULE_TAGS },
      resultTypes: ['violations'],
    });

    const violations = results.violations || [];
    // Don't re-log an identical result set on every re-render.
    const signature = violations.map((v) => `${v.id}:${v.nodes.length}`).join('|');
    if (signature === lastSignature) return;
    lastSignature = signature;

    if (violations.length === 0) {
      console.log(
        `%caxe: no violations%c  (${RULE_TAGS.join(', ')}) on ${location.hash || '#/'}`,
        'color:#1B7A4B;font-weight:700',
        'color:inherit',
      );
      return;
    }

    console.groupCollapsed(
      `%caxe: ${violations.length} violation type(s)%c on ${location.hash || '#/'}`,
      'color:#A3231B;font-weight:700',
      'color:inherit',
    );
    violations.forEach((violation) => {
      console.groupCollapsed(
        `%c${violation.impact}%c  ${violation.id} — ${violation.help} (${violation.nodes.length} node${violation.nodes.length > 1 ? 's' : ''})`,
        IMPACT_STYLE[violation.impact] || '',
        'color:inherit',
      );
      console.log(violation.helpUrl);
      violation.nodes.forEach((node) => {
        console.log(node.target.join(' '), node.failureSummary);
      });
      console.groupEnd();
    });
    console.groupEnd();
  } catch (error) {
    console.warn('axe run failed:', error);
  }
}

function schedule(axe) {
  clearTimeout(timer);
  // React in StrictMode double-renders and the map mutates the DOM continuously;
  // a debounce keeps axe from running dozens of times per navigation.
  timer = setTimeout(() => run(axe), 1200);
}

export async function start() {
  const axe = (await import('axe-core')).default;

  schedule(axe);
  window.addEventListener('hashchange', () => {
    lastSignature = ''; // a new route deserves a fresh report
    schedule(axe);
  });

  // Expose a manual trigger, for checking a state you had to click into.
  window.__axe = () => {
    lastSignature = '';
    return run(axe);
  };

  /**
   * Sweep every route and return a one-line summary per page.
   * Paste `await window.__axeScan()` into the console for a whole-site report.
   */
  window.__axeScan = async (routes = ROUTES) => {
    const report = [];
    for (const route of routes) {
      window.location.hash = route;
      // Let React render, Leaflet settle and Recharts finish its entry pass.
      await new Promise((resolve) => setTimeout(resolve, 2200));
      const results = await axe.run(document, {
        runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'] },
        resultTypes: ['violations'],
      });
      report.push(
        `${route.padEnd(11)} ${
          results.violations.length
            ? results.violations
                .map((v) => `${v.id}(${v.impact} x${v.nodes.length})`)
                .join(', ')
            : 'CLEAN'
        }`,
      );
    }
    const text = report.join(String.fromCharCode(10));
    console.log(text);
    return text;
  };
  console.log(
    '%caxe-core active%c — dev only. Call window.__axe() to re-check after interacting.',
    'color:#1F5C47;font-weight:700',
    'color:inherit',
  );
}
