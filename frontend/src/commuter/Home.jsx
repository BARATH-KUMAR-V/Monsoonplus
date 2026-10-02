/**
 * Home. The answer to "what is it doing out there, and can I go?" in the first screen,
 * before any scrolling.
 *
 * Flooding is a conditional hazard, not the subject of the app: on a dry day this page
 * is a weather and travel summary, and the flood line only appears when the forecast
 * puts it there.
 */
import { useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useApp } from '@/data/consumer';
import StatusBanner from './StatusBanner';
import WeatherIcon from './WeatherIcon';
import SourceBadge from './SourceBadge';
import PlaceInput from './PlaceInput';
import { describeCode, nextRain, rainWord } from '@/services/weather';
import { historyAt } from '@/engine/flood';
import { Loading } from '@/components/ui';

function Action({ to, icon, label, sub }) {
  const glyphs = {
    weather: <path d="M6.5 17a4.2 4.2 0 0 1 .5-8.4 5.4 5.4 0 0 1 10.2 1.2A3.8 3.8 0 0 1 17 17z" />,
    map: <path d="M2.5 6 9 3.5l6 2.5 6.5-2.5v14L15 20l-6-2.5L2.5 20zM9 3.5v14M15 6v14" />,
    report: <path d="M12 3.5 22 20.5H2zM12 10v4.5M12 17.5h.01" />,
    alerts: <path d="M5.5 17h13L16.8 15V11a4.8 4.8 0 0 0-9.6 0v4zM9.6 20a2.5 2.5 0 0 0 4.8 0" />,
    ask: <path d="M3.5 5.5h17v11h-9l-5 4v-4h-3zM9 10.5a3 3 0 0 1 5.6 1.4c0 1.6-2.2 1.9-2.2 3.1M12 17.6h.01" />,
  };
  return (
    <Link className="action" to={to}>
      <span className="action-ic" aria-hidden="true">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
          {glyphs[icon]}
        </svg>
      </span>
      <b>{label}</b>
      <i>{sub}</i>
    </Link>
  );
}

export default function Home() {
  const { weather, air, severe, focus, setFocus, profile, home, work, reports, located, locate } = useApp();
  const navigate = useNavigate();
  const [to, setTo] = useState(null);

  const data = weather.data;
  const current = data?.current;
  const soon = useMemo(() => (data ? nextRain(data.hourly) : null), [data]);

  // Flooding only enters the conversation when rain and a flood-prone area coincide.
  const floodWatch = useMemo(() => {
    if (!data || !focus) return null;
    const peak = Math.max(current?.precipitation ?? 0, soon?.peakMm ?? 0);
    if (peak < 4) return null;
    const area = historyAt(focus.lat, focus.lon);
    if (area.weight < 0.4) return null;
    return { area: area.name, peak };
  }, [data, focus, current, soon]);

  const activeReports = reports.length;

  const headline = () => {
    if (severe.length) return { tier: severe[0].severity === 'severe' ? 'critical' : 'warning', title: severe[0].title, detail: severe[0].detail };
    if (floodWatch) return { tier: 'warning', title: `Heavy rain around ${floodWatch.area}.`, detail: 'This area is prone to waterlogging. Allow extra time and avoid low roads.' };
    if (soon?.willRain) return { tier: 'warning', title: soon.startsInMinutes === 0 ? 'Rain is starting now.' : `Rain likely in about ${soon.startsInMinutes} minutes.`, detail: `Expect around ${soon.total.toFixed(1)} mm over the next three hours.` };
    return { tier: 'ok', title: 'Nothing to worry about right now.', detail: 'No storms, heavy rain or flooding expected in the next few hours.' };
  };

  const trips = [
    home && work && { label: `${home.name} to ${work.name}`, from: home, to: work },
    home && work && { label: `${work.name} to ${home.name}`, from: work, to: home },
  ].filter(Boolean);

  return (
    <div className="stack cm-page">
      {weather.state === 'loading' && <Loading label="Checking the weather" rows={3} />}

      {weather.state === 'error' && (
        <StatusBanner tier="warning" title="Weather could not be loaded." detail={weather.error} />
      )}

      {weather.state === 'ready' && current && (
        <>
          <section className="card pad hero">
            <div className="hero-top">
              <div>
                <div className="hero-place">
                  {focus.name}
                  <button type="button" className="link-btn" onClick={() => locate()}>
                    {located.state === 'locating' ? 'finding you…' : 'use my location'}
                  </button>
                </div>
                <div className="hero-temp">{Math.round(current.temperature_2m)}°</div>
                <div className="hero-cond">
                  {describeCode(current.weather_code).text} · feels like {Math.round(current.apparent_temperature)}°
                </div>
              </div>
              <span className="hero-icon" aria-hidden="true">
                <WeatherIcon name={describeCode(current.weather_code).icon} size={72} night={!current.is_day} />
              </span>
            </div>
            <div className="hero-strip">
              <span>Rain next 3 h <b>{soon ? `${Math.round(soon.chance)}%` : '—'}</b></span>
              <span>Wind <b>{Math.round(current.wind_speed_10m)} km/h</b></span>
              <span>Humidity <b>{Math.round(current.relative_humidity_2m)}%</b></span>
              {air.data?.us_aqi != null && <span>Air <b>{Math.round(air.data.us_aqi)}</b></span>}
            </div>
            <Link className="hero-more" to="/weather">See the full forecast</Link>
          </section>

          <StatusBanner {...headline()}>
            <SourceBadge source="official" basis="Open-Meteo forecast" at={data.fetchedAt} />
          </StatusBanner>

          {severe.slice(1).map((condition) => (
            <StatusBanner key={condition.title} tier="warning" title={condition.title} detail={condition.detail} />
          ))}
        </>
      )}

      <section className="card pad">
        <h2 className="card-title">Going somewhere?</h2>
        <PlaceInput label="Destination" value={to} onChange={setTo} placeholder="Where to?" allowMyLocation={false} />
        <button
          type="button"
          className="btn cm-primary"
          style={{ marginTop: 12 }}
          onClick={() => navigate('/trip')}
        >
          Plan the trip
        </button>
        {trips.length > 0 && (
          <div className="quick-row" style={{ marginTop: 12 }}>
            {trips.map((trip) => (
              <Link key={trip.label} className="btn ghost small" to="/trip">{trip.label}</Link>
            ))}
          </div>
        )}
        {!home && (
          <p className="cm-muted" style={{ marginTop: 10 }}>
            <Link to="/places">Save your home and work</Link> and we can check your usual trips in one tap.
          </p>
        )}
      </section>

      <div className="actions">
        <Action to="/weather" icon="weather" label="Weather" sub="Hourly and 14 days" />
        <Action to="/map" icon="map" label="Map" sub="Rain, wind and hazards" />
        <Action to="/report" icon="report" label="Report" sub="Flooding or a blocked road" />
        <Action to="/ask" icon="ask" label="Ask" sub="Questions about your trip" />
      </div>

      {activeReports > 0 && (
        <section className="card pad">
          <h2 className="card-title">Reported nearby in the last few hours</h2>
          <ul className="cm-list">
            {reports.slice(0, 4).map((report) => (
              <li key={report.id}>
                <span className={`cm-dot ${report.severity === 'severe' ? 'jammed' : 'slow'}`} aria-hidden="true" />
                <div>
                  <div className="cm-strong">{report.label}</div>
                  <div className="cm-muted">
                    {report.placeName || 'Reported location'} · {new Date(report.at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}
                  </div>
                </div>
                <SourceBadge source="community" />
              </li>
            ))}
          </ul>
          <Link className="btn ghost small" to="/map" style={{ marginTop: 10 }}>See them on the map</Link>
        </section>
      )}

      <p className="note">
        You are set up as a <b>{profile.label.toLowerCase()}</b> traveller, so warnings are ordered around{' '}
        {profile.cares[0].toLowerCase()}. <Link to="/settings">Change this</Link>.
      </p>
    </div>
  );
}
