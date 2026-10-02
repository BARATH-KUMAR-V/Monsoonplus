/**
 * App shell: sidebar navigation, top bar with the global mode selector, and the
 * always-visible LIVE / REPLAY / DATASET badge plus "last updated".
 *
 * The badge is deliberately prominent. The single worst outcome in a demo is an
 * evaluator thinking synthetic data is measured, so the mode is on screen at all
 * times and says when it has silently fallen back.
 */
import { useEffect, useRef, useState } from 'react';
import { Link, NavLink } from 'react-router-dom';
import { MODES, useStore } from '@/data/store';
import { useApp } from '@/data/consumer';
import { Segmented } from '@/components/ui';
import { hhmm } from '@/lib/format';

const COMMUTER_NAV = [
  { to: '/', label: 'Home', end: true, icon: 'home' },
  { to: '/trip', label: 'Plan a trip', icon: 'route' },
  { to: '/weather', label: 'Weather', icon: 'cloud' },
  { to: '/map', label: 'Map', icon: 'map' },
  { to: '/alerts', label: 'Alerts', icon: 'bell' },
  { to: '/places', label: 'Saved places', icon: 'star' },
  { to: '/report', label: 'Report a hazard', icon: 'warn' },
  { to: '/ask', label: 'Ask', icon: 'chat' },
  { to: '/settings', label: 'Settings', icon: 'gear' },
];

const NAV = [
  { to: '/', label: 'Overview', end: true, icon: 'home' },
  { to: '/trip', label: 'Smart Trip', icon: 'route' },
  { to: '/map', label: 'Live Map', icon: 'map' },
  { to: '/forecast', label: 'Forecast', icon: 'cloud' },
  { to: '/replay', label: 'Event Replay', icon: 'play' },
  { to: '/model', label: 'Model Lab', icon: 'lab' },
  { to: '/system', label: 'Data & System', icon: 'gear' },
];

function Icon({ name }) {
  const common = { width: 16, height: 16, viewBox: '0 0 16 16', fill: 'none', 'aria-hidden': true };
  const stroke = { stroke: 'currentColor', strokeWidth: 1.6, strokeLinecap: 'round', strokeLinejoin: 'round' };
  switch (name) {
    case 'home':
      return <svg {...common}><path d="M2 7l6-5 6 5v6a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1z" {...stroke} /></svg>;
    case 'route':
      return <svg {...common}><circle cx="3.5" cy="12.5" r="1.8" {...stroke} /><circle cx="12.5" cy="3.5" r="1.8" {...stroke} /><path d="M5 11.5c4 0 2-7 6-7" {...stroke} /></svg>;
    case 'map':
      return <svg {...common}><path d="M1.5 4 6 2.2 10 4l4.5-1.8v9.6L10 13.8 6 12 1.5 13.8z" {...stroke} /><path d="M6 2.2V12M10 4v9.8" {...stroke} /></svg>;
    case 'cloud':
      return <svg {...common}><path d="M4.5 11a2.8 2.8 0 0 1 .3-5.6 3.6 3.6 0 0 1 6.8.8A2.5 2.5 0 0 1 11.3 11z" {...stroke} /><path d="M6 13.4 5.4 15M9 13.4 8.4 15" {...stroke} /></svg>;
    case 'play':
      return <svg {...common}><circle cx="8" cy="8" r="6.2" {...stroke} /><path d="M6.6 5.6 10.4 8l-3.8 2.4z" {...stroke} /></svg>;
    case 'lab':
      return <svg {...common}><path d="M6.4 2v4L3 12.4A1 1 0 0 0 3.9 14h8.2a1 1 0 0 0 .9-1.6L9.6 6V2" {...stroke} /><path d="M5.4 2h5.2" {...stroke} /></svg>;
    case 'star':
      return <svg {...common}><path d="M8 1.8l1.9 3.9 4.3.6-3.1 3 .7 4.3L8 11.6l-3.8 2 .7-4.3-3.1-3 4.3-.6z" {...stroke} /></svg>;
    case 'warn':
      return <svg {...common}><path d="M8 2.2 14.8 13.5H1.2z" {...stroke} /><path d="M8 6.6v3.2M8 11.6h.01" {...stroke} /></svg>;
    case 'chat':
      return <svg {...common}><path d="M2 3.5h12v8H7l-3.4 2.6V11.5H2z" {...stroke} /></svg>;
    case 'bell':
      return <svg {...common}><path d="M3.5 11.5h9L11.4 10V7a3.4 3.4 0 0 0-6.8 0v3z" {...stroke} /><path d="M6.6 13.6a1.5 1.5 0 0 0 2.8 0" {...stroke} /></svg>;
    case 'gear':
      return <svg {...common}><circle cx="8" cy="8" r="2.2" {...stroke} /><path d="M8 1.6v1.8M8 12.6v1.8M14.4 8h-1.8M3.4 8H1.6M12.5 3.5l-1.3 1.3M4.8 11.2l-1.3 1.3M12.5 12.5l-1.3-1.3M4.8 4.8 3.5 3.5" {...stroke} /></svg>;
    default:
      return null;
  }
}

/** Which place the weather pages are pointed at, always visible in the consumer view. */
function PlaceTag() {
  const { focus, weather } = useApp();
  const tone = weather.state === 'ready' ? 'ok' : weather.state === 'error' ? 'bad' : 'warn';
  return (
    <span className="tag" title="The place these pages are showing">
      <span className={`dot ${tone}`} aria-hidden="true" />
      <strong style={{ color: 'var(--ink)' }}>{focus?.name || 'Chennai'}</strong>
    </span>
  );
}

function ModeBadge() {
  const { effectiveMode } = useStore();
  const tone = effectiveMode.tone === 'ok' ? 'ok' : 'warn';
  return (
    <span className="tag" title="Which data this screen is showing right now">
      <span className={`dot ${tone}`} aria-hidden="true" />
      <strong style={{ color: 'var(--ink)' }}>{effectiveMode.label}</strong>
    </span>
  );
}

export default function Shell({ children }) {
  const { mode, setMode, highContrast, setHighContrast, status, viewMode, setViewMode } = useStore();
  const commuter = viewMode === 'commuter';
  const nav = commuter ? COMMUTER_NAV : NAV;
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [updated, setUpdated] = useState(hhmm());
  const menuButtonRef = useRef(null);

  useEffect(() => {
    const timer = setInterval(() => setUpdated(hhmm()), 60000);
    return () => clearInterval(timer);
  }, []);

  // Close the mobile drawer on Escape and return focus where it came from.
  useEffect(() => {
    if (!sidebarOpen) return undefined;
    const onKey = (event) => {
      if (event.key === 'Escape') {
        setSidebarOpen(false);
        menuButtonRef.current?.focus();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [sidebarOpen]);

  return (
    <div className="app">
      <a className="skip-link" href="#main">
        Skip to main content
      </a>

      <nav
        className={`sidebar ${sidebarOpen ? 'open' : ''}`}
        aria-label="Primary"
        id="sidebar"
      >
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M4 10c0-2.6 4-7 4-7s4 4.4 4 7a4 4 0 1 1-8 0z" fill="currentColor" />
            </svg>
          </span>
          <div>
            <div className="brand-name">MonsoonPlus</div>
            <div className="brand-sub">{commuter ? 'Weather and safer routes' : 'Research view'}</div>
          </div>
        </div>

        <div className="nav">
          {nav.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              onClick={() => setSidebarOpen(false)}
            >
              <span className="nav-ic">
                <Icon name={item.icon} />
              </span>
              {item.label}
            </NavLink>
          ))}
        </div>

        <div className="sidebar-foot">
          {commuter ? (
            <>
              <div>
                Planning aid, not a safety system. Never use it to decide whether to cross flood water.
              </div>
              <div style={{ marginTop: 6 }}>
                <Link to="/about" onClick={() => setSidebarOpen(false)}>About and data sources</Link>
              </div>
            </>
          ) : (
            <>
              <div>Prototype - not a freedom-to-operate or safety system.</div>
              <div style={{ marginTop: 6 }}>
                Status:{' '}
                <strong>
                  {status === 'ready' ? 'model data loaded' : status === 'loading' ? 'loading' : 'data error'}
                </strong>
              </div>
              <button type="button" className="btn small" style={{ marginTop: 8 }} onClick={() => setViewMode('commuter')}>
                Switch to simple view
              </button>
            </>
          )}
        </div>
      </nav>

      {sidebarOpen && (
        <div
          className="drawer-backdrop"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      <main id="main">
        <div className="topbar">
          <div className="topbar-group">
            <button
              type="button"
              className="menu-btn"
              ref={menuButtonRef}
              onClick={() => setSidebarOpen((open) => !open)}
              aria-expanded={sidebarOpen}
              aria-controls="sidebar"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
                <path d="M2 4h12M2 8h12M2 12h12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
              </svg>
              Menu
            </button>

            {!commuter && (
              <Segmented
                label="Data mode"
                value={mode}
                onChange={setMode}
                options={MODES.map((m) => ({ value: m.id, label: m.label, title: m.blurb }))}
              />
            )}
          </div>

          <div className="topbar-group">
            {commuter ? <PlaceTag /> : <ModeBadge />}
            <span className="tag" title="Clock time this page last refreshed">
              Updated {updated}
            </span>
            <button
              type="button"
              className="btn ghost small"
              aria-pressed={highContrast}
              onClick={() => setHighContrast((v) => !v)}
              title="Increase contrast for low-vision readers"
            >
              {highContrast ? 'High contrast on' : 'High contrast'}
            </button>
          </div>
        </div>

        {children}
      </main>
    </div>
  );
}
