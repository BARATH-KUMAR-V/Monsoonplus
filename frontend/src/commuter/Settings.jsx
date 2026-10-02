/**
 * Settings: who you are, how you want to be told, and the way into the developer view.
 */
import { Link } from 'react-router-dom';
import { useApp } from '@/data/consumer';
import { useStore } from '@/data/store';
import { PROFILES } from '@/engine/profiles';
import { REPORT_TTL_HOURS } from '@/data/consumer';

export default function Settings() {
  const {
    profileId, setProfileId, profile, alertsEnabled, setAlertsEnabled,
    voiceEnabled, setVoiceEnabled, bigText, setBigText, reports, places,
  } = useApp();
  const { highContrast, setHighContrast, setViewMode } = useStore();

  const notificationsSupported = typeof window !== 'undefined' && 'Notification' in window;

  const enableAlerts = async () => {
    if (notificationsSupported && Notification.permission === 'default') await Notification.requestPermission();
    setAlertsEnabled(true);
  };

  return (
    <div className="stack cm-page">
      <h1 className="cm-h1">Settings</h1>

      <section className="card pad">
        <h2 className="card-title">How you travel</h2>
        <p className="cm-muted">This decides how routes are scored and which warnings come first.</p>
        <div className="settings-grid" role="radiogroup" aria-label="How you travel">
          {PROFILES.map((item) => (
            <button
              key={item.id}
              type="button"
              role="radio"
              aria-checked={profileId === item.id}
              className={`settings-opt ${profileId === item.id ? 'on' : ''}`}
              onClick={() => setProfileId(item.id)}
            >
              <b>{item.label}</b>
              <i>{item.blurb}</i>
            </button>
          ))}
        </div>
        <div className="note" style={{ marginTop: 12 }}>
          As a <b>{profile.label.toLowerCase()}</b> traveller, routes are weighted{' '}
          {Object.entries(profile.weights)
            .sort((a, b) => b[1] - a[1])
            .map(([key, value]) => `${key} ${Math.round(value * 100)}%`)
            .join(', ')}
          .
        </div>
      </section>

      <section className="card pad">
        <h2 className="card-title">Alerts</h2>
        <p className="cm-muted">Get told when conditions turn bad on a place or route you have saved.</p>
        <div className="inline" style={{ marginTop: 8 }}>
          {alertsEnabled ? (
            <>
              <span className="chip smooth">Alerts on</span>
              <button type="button" className="btn ghost small" onClick={() => setAlertsEnabled(false)}>Turn off</button>
            </>
          ) : (
            <button type="button" className="btn cm-primary" onClick={enableAlerts}>Turn on alerts</button>
          )}
        </div>
        <p className="cm-muted small" style={{ marginTop: 8 }}>
          Alerts arrive while MonsoonPlus is open in a browser tab. Getting them when the app is closed needs a server,
          which this build does not have.
        </p>
      </section>

      <section className="card pad">
        <h2 className="card-title">Reading and accessibility</h2>
        <div className="toggles">
          <label className="toggle">
            <input type="checkbox" checked={bigText} onChange={(e) => setBigText(e.target.checked)} />
            <span><b>Larger text</b><i>Increases text size across the app</i></span>
          </label>
          <label className="toggle">
            <input type="checkbox" checked={highContrast} onChange={(e) => setHighContrast(e.target.checked)} />
            <span><b>Higher contrast</b><i>Stronger colours and borders</i></span>
          </label>
          <label className="toggle">
            <input type="checkbox" checked={voiceEnabled} onChange={(e) => setVoiceEnabled(e.target.checked)} />
            <span><b>Read warnings aloud</b><i>Speaks severe weather warnings when they appear</i></span>
          </label>
        </div>
      </section>

      <section className="card pad">
        <h2 className="card-title">Your data</h2>
        <p className="cm-muted">
          Everything MonsoonPlus knows about you stays in this browser: {places.length} saved place
          {places.length === 1 ? '' : 's'}, {reports.length} active report{reports.length === 1 ? '' : 's'}, and your
          travel mode. Nothing is sent to a MonsoonPlus server, because there is not one. Reports expire after{' '}
          {REPORT_TTL_HOURS} hours.
        </p>
        <p className="cm-muted small">
          Place searches, routes and forecasts are requested from OpenStreetMap, OSRM and Open-Meteo as you use the app,
          so those services see the coordinates you look up.
        </p>
      </section>

      <section className="card pad">
        <h2 className="card-title">Developer view</h2>
        <p className="cm-muted">
          The trained model, its evaluation, the data sources and the Dataset / Live / Log modes. Built for reviewers
          rather than travellers.
        </p>
        <div className="inline">
          <button type="button" className="btn" onClick={() => setViewMode('developer')}>Open developer view</button>
          <Link className="btn ghost" to="/about">About this app</Link>
        </div>
      </section>
    </div>
  );
}
