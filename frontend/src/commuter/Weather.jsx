/**
 * The weather page. On most days this is the page people actually open: it is a full
 * weather app, and the monsoon-specific parts only come forward when conditions call
 * for them.
 */
import { useMemo, useState } from 'react';
import { useApp } from '@/data/consumer';
import PlaceInput from './PlaceInput';
import StatusBanner from './StatusBanner';
import SourceBadge, { SourceLine } from './SourceBadge';
import WeatherIcon, { WindArrow } from './WeatherIcon';
import { RainChart, TempChart, DayRange } from './charts';
import { aqiBand, compass, describeCode, nextRain, rainWord, uvBand } from '@/services/weather';
import { Loading } from '@/components/ui';
import { relativeTime } from '@/lib/format';

const hhmm = (iso) => (iso ? new Date(iso).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }) : '--');
const hourLabel = (date) => date.toLocaleTimeString([], { hour: 'numeric' }).replace(' ', '');

function Detail({ label, value, note, children }) {
  return (
    <div className="wx-detail">
      <span className="wx-detail-label">{label}</span>
      <span className="wx-detail-value">{value}{children}</span>
      {note && <span className="wx-detail-note">{note}</span>}
    </div>
  );
}

export default function Weather() {
  const { focus, setFocus, weather, air, severe, anomalies, reloadWeather, savePlace } = useApp();
  const [days, setDays] = useState(7);

  const data = weather.data;
  const current = data?.current;

  const upcoming = useMemo(() => {
    if (!data?.hourly) return [];
    const now = Date.now() - 1800000;
    return data.hourly.filter((h) => h.date >= now).slice(0, 48);
  }, [data]);

  const soon = useMemo(() => (data ? nextRain(data.hourly) : null), [data]);

  const rainRows = upcoming.slice(0, 24).map((h) => ({
    label: hourLabel(h.date),
    value: h.precipitation ?? 0,
    extra: h.precipitation_probability != null ? `${Math.round(h.precipitation_probability)}% chance` : null,
  }));
  const tempRows = upcoming.slice(0, 24).map((h) => ({ label: hourLabel(h.date), value: h.temperature_2m }));

  const daily = (data?.daily || []).slice(0, days);
  const tempLo = daily.length ? Math.floor(Math.min(...daily.map((d) => d.temperature_2m_min))) : 0;
  const tempHi = daily.length ? Math.ceil(Math.max(...daily.map((d) => d.temperature_2m_max))) : 1;

  const uv = uvBand(current?.uv_index ?? upcoming[0]?.uv_index);
  const aq = aqiBand(air.data?.us_aqi);
  const today = daily[0];

  return (
    <div className="stack cm-page wide">
      <div className="wx-head">
        <h1 className="cm-h1">Weather</h1>
        <div className="wx-place">
          <PlaceInput label="Place" value={focus} onChange={(p) => p && setFocus(p)} placeholder="Search a city or area" />
        </div>
      </div>

      {weather.state === 'loading' && <Loading label="Loading the forecast" rows={4} />}

      {weather.state === 'error' && (
        <>
          <StatusBanner tier="warning" title="The forecast could not be loaded." detail={weather.error} />
          <button type="button" className="btn" onClick={reloadWeather}>Try again</button>
        </>
      )}

      {weather.state === 'ready' && current && (
        <>
          {severe.map((condition) => (
            <StatusBanner
              key={condition.title}
              tier={condition.severity === 'severe' ? 'critical' : 'warning'}
              title={condition.title}
              detail={condition.detail}
            >
              <SourceBadge source="official" basis="threshold over the Open-Meteo forecast" />
            </StatusBanner>
          ))}

          {anomalies.map((anomaly) => (
            <div key={anomaly.title} className="note anomaly">
              <b>{anomaly.title}.</b> {anomaly.detail}{' '}
              <SourceBadge source="estimate" basis="statistical comparison against the past four weeks here" />
            </div>
          ))}

          {/* ---- now ---- */}
          <section className="card pad wx-now">
            <div className="wx-now-main">
              <span className="wx-icon" aria-hidden="true">
                <WeatherIcon name={describeCode(current.weather_code).icon} size={64} night={!current.is_day} />
              </span>
              <div>
                <div className="wx-temp">{Math.round(current.temperature_2m)}°</div>
                <div className="wx-cond">{describeCode(current.weather_code).text}</div>
                <div className="cm-muted">
                  Feels like {Math.round(current.apparent_temperature)}°
                  {today && ` · today ${Math.round(today.temperature_2m_min)}° to ${Math.round(today.temperature_2m_max)}°`}
                </div>
              </div>
            </div>

            <p className="wx-rainline">
              {soon?.willRain
                ? soon.startsInMinutes === 0
                  ? 'Rain is starting about now.'
                  : `Rain likely in about ${soon.startsInMinutes} minutes.`
                : (current.precipitation ?? 0) > 0.2
                  ? `Raining now, ${rainWord(current.precipitation)}.`
                  : 'No rain expected in the next 3 hours.'}
              {soon && <span className="cm-muted"> Chance in the next 3 hours: {Math.round(soon.chance)}%.</span>}
            </p>

            <div className="wx-details">
              <Detail label="Humidity" value={`${Math.round(current.relative_humidity_2m)}%`} />
              <Detail label="Wind" value={`${Math.round(current.wind_speed_10m)} km/h`} note={`from the ${compass(current.wind_direction_10m)}${current.wind_gusts_10m ? `, gusts ${Math.round(current.wind_gusts_10m)}` : ''}`}>
                <span className="wx-arrow"><WindArrow degrees={current.wind_direction_10m} size={15} /></span>
              </Detail>
              <Detail label="Pressure" value={`${Math.round(current.pressure_msl)} hPa`} />
              <Detail label="Cloud cover" value={`${Math.round(current.cloud_cover)}%`} />
              <Detail label="Visibility" value={upcoming[0]?.visibility != null ? `${(upcoming[0].visibility / 1000).toFixed(1)} km` : 'not available'} />
              <Detail label="UV index" value={uv ? `${Math.round(current.uv_index ?? upcoming[0]?.uv_index ?? 0)} · ${uv.label}` : 'not available'} note={uv?.advice} />
              <Detail label="Sunrise" value={hhmm(today?.sunrise)} />
              <Detail label="Sunset" value={hhmm(today?.sunset)} />
              <Detail
                label="Air quality"
                value={air.state === 'loading' ? 'loading…' : aq ? `${Math.round(air.data.us_aqi)} · ${aq.label}` : 'not available'}
                note={aq?.advice || (air.state === 'error' ? 'The air quality service did not respond.' : null)}
              />
            </div>

            <SourceLine source="official" basis="Open-Meteo" at={data.fetchedAt} />
          </section>

          {/* ---- next 48 hours ---- */}
          <section className="card pad">
            <h2 className="card-title">Next 24 hours</h2>
            <RainChart rows={rainRows} label="Rain expected" />
            <TempChart rows={tempRows} label="Temperature" />
            <details className="wx-table">
              <summary>See it as a table</summary>
              <div className="table-scroll" tabIndex={0} role="region" aria-label="Hourly forecast table">
                <table>
                  <thead>
                    <tr><th>Time</th><th>Temp</th><th>Rain</th><th>Chance</th><th>Wind</th></tr>
                  </thead>
                  <tbody>
                    {upcoming.slice(0, 24).map((h) => (
                      <tr key={h.time}>
                        <td>{hourLabel(h.date)}</td>
                        <td>{Math.round(h.temperature_2m)}°</td>
                        <td>{(h.precipitation ?? 0).toFixed(1)} mm</td>
                        <td>{Math.round(h.precipitation_probability ?? 0)}%</td>
                        <td>{Math.round(h.wind_speed_10m)} km/h</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
            <SourceLine source="official" basis="Open-Meteo hourly forecast" at={data.fetchedAt} />
          </section>

          {/* ---- the days ahead ---- */}
          <section className="card pad">
            <div className="card-head">
              <h2 className="card-title">The days ahead</h2>
              <div className="seg" role="group" aria-label="How many days">
                {[7, 14].map((n) => (
                  <button key={n} type="button" aria-pressed={days === n} onClick={() => setDays(n)}>{n} days</button>
                ))}
              </div>
            </div>
            <ul className="days">
              {daily.map((day, index) => (
                <li key={day.time}>
                  <span className="day-name">{index === 0 ? 'Today' : day.date.toLocaleDateString([], { weekday: 'short', day: 'numeric', month: 'short' })}</span>
                  <span className="day-icon" aria-hidden="true"><WeatherIcon name={describeCode(day.weather_code).icon} size={22} /></span>
                  <span className="day-rain">
                    {(day.precipitation_sum ?? 0) > 0.1 ? `${day.precipitation_sum.toFixed(1)} mm` : '—'}
                    <i>{Math.round(day.precipitation_probability_max ?? 0)}%</i>
                  </span>
                  <span className="day-lo">{Math.round(day.temperature_2m_min)}°</span>
                  <DayRange day={day.time} min={day.temperature_2m_min} max={day.temperature_2m_max} lo={tempLo} hi={tempHi} />
                  <span className="day-hi">{Math.round(day.temperature_2m_max)}°</span>
                </li>
              ))}
            </ul>
            <SourceLine source="official" basis="Open-Meteo daily forecast" at={data.fetchedAt} />
          </section>

          <div className="inline wrap">
            <button type="button" className="btn ghost" onClick={() => savePlace(focus, 'custom')}>Save this place</button>
            <button type="button" className="btn ghost" onClick={reloadWeather}>Refresh</button>
            <span className="cm-muted">Last updated {relativeTime(data.fetchedAt)}</span>
          </div>

          <p className="note">
            Forecasts come from Open-Meteo, a free public weather service. MonsoonPlus does not issue official weather
            warnings. For those, check the India Meteorological Department.
          </p>
        </>
      )}
    </div>
  );
}
