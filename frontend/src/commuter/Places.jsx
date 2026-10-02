/**
 * Saved places. Home, Work and College get a role so the rest of the app can offer
 * them as one-tap destinations and show their conditions side by side.
 */
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useApp } from '@/data/consumer';
import PlaceInput from './PlaceInput';
import WeatherIcon from './WeatherIcon';
import { describeCode, fetchWeather, nextRain } from '@/services/weather';
import SourceBadge from './SourceBadge';

const ROLES = [
  { id: 'home', label: 'Home' },
  { id: 'work', label: 'Work' },
  { id: 'college', label: 'College' },
  { id: 'custom', label: 'Other place' },
];

/** Each saved place loads its own conditions, and fails on its own. */
function PlaceCard({ place, onRemove }) {
  const [state, setState] = useState({ status: 'loading', data: null });

  useEffect(() => {
    let cancelled = false;
    fetchWeather(place.lat, place.lon, { days: 2 })
      .then((data) => !cancelled && setState({ status: 'ready', data }))
      .catch(() => !cancelled && setState({ status: 'error', data: null }));
    return () => { cancelled = true; };
  }, [place.lat, place.lon]);

  const current = state.data?.current;
  const soon = state.data ? nextRain(state.data.hourly) : null;

  return (
    <article className="place-card">
      <header>
        <div>
          <span className="place-role">{ROLES.find((r) => r.id === place.role)?.label || 'Saved'}</span>
          <h3>{place.name}</h3>
          {place.context && <p className="cm-muted">{place.context}</p>}
        </div>
        {current && (
          <div className="place-wx">
            <WeatherIcon name={describeCode(current.weather_code).icon} size={32} night={!current.is_day} />
            <b>{Math.round(current.temperature_2m)}°</b>
          </div>
        )}
      </header>

      {state.status === 'loading' && <p className="cm-muted">Checking conditions…</p>}
      {state.status === 'error' && <p className="cm-muted">Conditions could not be loaded for this place.</p>}
      {state.status === 'ready' && (
        <p className="cm-big">
          {soon?.willRain
            ? soon.startsInMinutes === 0
              ? 'Rain starting now.'
              : `Rain likely in about ${soon.startsInMinutes} minutes.`
            : 'No rain expected in the next 3 hours.'}
        </p>
      )}

      <div className="inline">
        <Link className="btn ghost small" to="/trip">Plan a trip here</Link>
        <button type="button" className="btn ghost small" onClick={() => onRemove(place.id)}>Remove</button>
        {state.status === 'ready' && <SourceBadge source="official" at={state.data.fetchedAt} />}
      </div>
    </article>
  );
}

export default function Places() {
  const { places, savePlace, removePlace } = useApp();
  const [adding, setAdding] = useState(null);
  const [role, setRole] = useState('home');

  return (
    <div className="stack cm-page">
      <h1 className="cm-h1">Saved places</h1>
      <p className="cm-muted">Save the places you travel between, and the app can check them for you in one tap.</p>

      <section className="card pad">
        <h2 className="card-title">Add a place</h2>
        <div className="seg" role="group" aria-label="What kind of place" style={{ marginBottom: 12 }}>
          {ROLES.map((r) => (
            <button key={r.id} type="button" aria-pressed={role === r.id} onClick={() => setRole(r.id)}>{r.label}</button>
          ))}
        </div>
        <PlaceInput label="Place" value={adding} onChange={setAdding} placeholder="Search for the place" />
        <button
          type="button"
          className="btn cm-primary"
          style={{ marginTop: 12 }}
          disabled={!adding}
          onClick={() => { savePlace(adding, role); setAdding(null); }}
        >
          Save as {ROLES.find((r) => r.id === role).label}
        </button>
      </section>

      {places.length === 0 ? (
        <p className="cm-muted">Nothing saved yet.</p>
      ) : (
        <div className="place-cards">
          {places.map((place) => <PlaceCard key={place.id} place={place} onRemove={removePlace} />)}
        </div>
      )}
    </div>
  );
}
