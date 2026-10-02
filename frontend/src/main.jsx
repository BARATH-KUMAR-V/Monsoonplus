import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { HashRouter } from 'react-router-dom';

// Public Sans, self-hosted. These imports pull the woff2 into the bundle, so the font
// is served from our own origin -- no Google Fonts request, no CDN, nothing external.
import '@fontsource-variable/public-sans';
import '@fontsource-variable/public-sans/wght-italic.css';

import App from './App.jsx';
import { StoreProvider } from './data/store.jsx';
import { ConsumerProvider } from './data/consumer.jsx';
import './styles/app.css';

/*
 * HashRouter rather than BrowserRouter: the built site is deployed as static files
 * (Vercel / Netlify / GitHub Pages / even opened from disk), and a hash route needs
 * no server-side rewrite rule to survive a page refresh on /forecast.
 */
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <HashRouter>
      <StoreProvider>
        <ConsumerProvider>
          <App />
        </ConsumerProvider>
      </StoreProvider>
    </HashRouter>
  </StrictMode>,
);

/*
 * Accessibility checking in development only.
 *
 * axe-core runs against the live DOM and reports WCAG violations to the console. The
 * import is inside an `import.meta.env.DEV` branch, which Vite statically replaces with
 * `false` for production builds -- so Rollup drops this whole block and axe-core never
 * reaches the deployed bundle.
 *
 * Run `npm run dev`, open the browser console, and navigate the pages. A clean run
 * prints "axe: no violations" per route.
 */
if (import.meta.env.DEV) {
  import('./dev/axe.js').then((module) => module.start()).catch(() => {
    /* axe is a development nicety; never let it break the app */
  });
}
