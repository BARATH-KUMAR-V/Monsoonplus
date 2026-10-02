/**
 * Page 4 - Forecast.
 *
 * The rain slider sweeps the trained model across its exported rainfall grid. The
 * hourly timeline and the alerts come from Open-Meteo when the collector is running,
 * and from a clearly-labelled generated series when it is not.
 */
import { useEffect, useMemo, useState } from 'react';
import {
  Area,
  Bar as RBar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useStore } from '@/data/store';
import {
  Banner,
  Card,
  Chip,
  EmptyState,
  PageHead,
  Provenance,
  Stat,
  StatusChip,
  TableScroll,
} from '@/components/ui';
import { SEGMENTS, THRESHOLDS } from '@/config/network';
import { rainBucket, rainRegime } from '@/lib/model';
import { ALERT_RULES, buildAlerts, historicalLikeToday } from '@/lib/alerts';
import { downloadText, kmh, mm, toCsv } from '@/lib/format';

const CHART_GRID = 'var(--line)';
const AXIS_STYLE = { fontSize: 11, fill: 'var(--muted)' };

/** Hourly weather: live from the collector if present, generated otherwise. */
function useHourlyWeather() {
  const { mode, live, liveApiUrl } = useStore();
  const [data, setData] = useState({ state: 'loading', hourly: [], source: null });

  useEffect(() => {
    let cancelled = false;

    async function load() {
      // Prefer whatever the live collector already fetched.
      if (mode === 'live' && live.payload?.weather?.hourly) {
        if (!cancelled) {
          setData({
            state: 'ready',
            hourly: live.payload.weather.hourly,
            source: live.payload.weather.source,
          });
        }
        return;
      }
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 4000);
        const response = await fetch(`${liveApiUrl}/api/weather?hours=24`, {
          signal: controller.signal,
        });
        clearTimeout(timer);
        if (!response.ok) throw new Error('unavailable');
        const payload = await response.json();
        if (!cancelled) {
          setData({
            state: 'ready',
            hourly: payload.hourly || [],
            source: payload.source,
          });
        }
      } catch {
        // The backend is optional. Generate a labelled series so the page still works.
        if (!cancelled) setData({ state: 'ready', hourly: generatedHourly(), source: 'heuristic_fallback' });
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [mode, live.payload, liveApiUrl]);

  return data;
}

/** Deterministic generated 24-hour series. Labelled heuristic_fallback everywhere. */
function generatedHourly() {
  const start = new Date();
  start.setMinutes(0, 0, 0);
  return Array.from({ length: 24 }, (_, index) => {
    const when = new Date(start.getTime() + index * 3600000);
    const hour = when.getHours();
    // Chennai's northeast-monsoon rain peaks late afternoon into the evening.
    const storm = Math.exp(-0.5 * ((hour - 18) / 4.5) ** 2);
    const precipitation = Math.max(0, storm * 46 - 1.5 + (index % 3) * 1.4);
    return {
      time: when.toISOString().slice(0, 16),
      precipitation: Number(precipitation.toFixed(1)),
      temperature: Number((29 - 4 * Math.sin(((hour - 5) / 24) * 2 * Math.PI)).toFixed(1)),
      wind: Number((9 + 0.2 * precipitation).toFixed(1)),
      humidity: Math.round(Math.min(100, 68 + 0.7 * precipitation)),
      visibility: Math.round(Math.max(500, 10000 - 230 * precipitation)),
    };
  });
}

function sourceLabel(source) {
  switch (source) {
    case 'open_meteo_forecast':
      return { text: 'Open-Meteo forecast (live)', tone: 'rain', measured: true };
    case 'open_meteo_archive':
      return { text: 'Open-Meteo ERA5 archive', tone: 'rain', measured: true };
    case 'heuristic_fallback':
    default:
      return { text: 'Generated - no live weather connection', tone: 'slow', measured: false };
  }
}

export default function Forecast() {
  const { rain, setRain, trend, setTrend, predictor, statesAt, states, chennai } = useStore();
  const weather = useHourlyWeather();
  const [aqi, setAqi] = useState(null);

  const source = sourceLabel(weather.source);

  // Air quality badge: optional, and labelled when it is demo data.
  useEffect(() => {
    let cancelled = false;
    fetch('https://air-quality-api.open-meteo.com/v1/air-quality?latitude=13.08&longitude=80.27&current=us_aqi,pm2_5,pm10&timezone=Asia%2FKolkata')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('aqi'))))
      .then((payload) => {
        if (!cancelled && payload?.current) {
          setAqi({ ...payload.current, measured: true });
        }
      })
      .catch(() => {
        if (!cancelled) setAqi({ us_aqi: 78, pm2_5: 23.4, pm10: 52.1, measured: false });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const horizonStates = useMemo(
    () => ({
      now: statesAt(0),
      t15: statesAt(15),
      t30: statesAt(30),
      t60: statesAt(60),
    }),
    [statesAt],
  );

  const kpis = useMemo(() => {
    const mean = (list) => (list.length ? list.reduce((a, s) => a + s.speed, 0) / list.length : 0);
    return [
      { label: 'Now', value: mean(horizonStates.now) },
      { label: 'In 15 min', value: mean(horizonStates.t15) },
      { label: 'In 30 min', value: mean(horizonStates.t30) },
      { label: 'In 60 min', value: mean(horizonStates.t60) },
    ];
  }, [horizonStates]);

  // Uncertainty band chart across the rainfall sweep.
  const sweep = useMemo(() => {
    if (!predictor) return [];
    return predictor.levels.map((level) => {
      const atLevel = predictor.allStates(level, 30);
      const mean = atLevel.reduce((a, s) => a + s.speed, 0) / atLevel.length;
      const halfWidth =
        atLevel.reduce((a, s) => a + s.halfWidth, 0) / Math.max(atLevel.length, 1);
      return {
        rain: level,
        speed: Number(mean.toFixed(2)),
        lower: Number((mean - halfWidth).toFixed(2)),
        bandWidth: Number((halfWidth * 2).toFixed(2)),
      };
    });
  }, [predictor]);

  // Which roads cross into each status as rainfall rises.
  const whatIfTable = useMemo(() => {
    if (!predictor) return [];
    const thresholds = [10, 25, 40, 55, 70, 85];
    return SEGMENTS.map((segment) => {
      const row = { segment: segment.name, id: segment.id };
      thresholds.forEach((level) => {
        const state = predictor.stateFor(segment.id, level, 30);
        row[`r${level}`] = state ? state.level.key : null;
      });
      // First rainfall level at which this road stops being Smooth.
      row.breakPoint =
        thresholds.find((level) => {
          const state = predictor.stateFor(segment.id, level, 30);
          return state && state.level.index >= 1;
        }) ?? null;
      return row;
    }).sort((a, b) => (a.breakPoint ?? 999) - (b.breakPoint ?? 999));
  }, [predictor]);

  // Scatter: rainfall against the model's predicted slowdown. Justifies "weather matters".
  const rainVsSlowdown = useMemo(() => {
    if (!predictor) return [];
    const points = [];
    predictor.levels.forEach((level) => {
      predictor.allStates(level, 30).forEach((state) => {
        points.push({
          rain: Number(state.rain.toFixed(1)),
          slowdown: Number(
            (state.segment.free_flow_kmh - state.speed).toFixed(2),
          ),
          road: state.segment.name,
        });
      });
    });
    return points;
  }, [predictor]);

  // A text equivalent of the scatter, for screen readers and for anyone who would
  // rather read the numbers than squint at 100 dots.
  const scatterSummary = useMemo(() => {
    if (!predictor) return [];
    return predictor.levels.map((level) => {
      const states = predictor.allStates(level, 30);
      const slowdown =
        states.reduce((a, s) => a + (s.segment.free_flow_kmh - s.speed), 0) / states.length;
      return { rain: level, slowdown };
    });
  }, [predictor]);

  // Scenario battle.
  const scenarios = useMemo(() => {
    if (!predictor) return null;
    const build = (rainValue) => {
      const list = predictor.allStates(rainValue, 30);
      return {
        rain: rainValue,
        mean: list.reduce((a, s) => a + s.speed, 0) / list.length,
        disrupted: list.filter((s) => s.level.index >= 1).length,
        jammed: list.filter((s) => s.level.index === 3).length,
        flood: list.filter((s) => s.flood.index >= 1).length,
        worst: [...list].sort((a, b) => b.impact - a.impact)[0],
      };
    };
    return { normal: build(0), monsoon: build(75) };
  }, [predictor]);

  // "What if it weren't raining?" -- a counterfactual sensitivity read, not a second
  // model. It takes the SAME trained model's own rain=0 row from the exported grid
  // (forecast_grid always samples rain=0 - see predictor.levels) and compares it to
  // the model's prediction at the currently selected rain level. The whole point is
  // that both numbers come out of monsoonplus; nothing here is invented.
  const whatIfDry = useMemo(() => {
    if (!predictor) return null;
    const withRain = predictor.allStates(rain, 30);
    const dry = predictor.allStates(0, 30);
    const rows = SEGMENTS.map((segment) => {
      const wet = withRain.find((s) => s.segmentId === segment.id);
      const clear = dry.find((s) => s.segmentId === segment.id);
      if (!wet || !clear) return null;
      return {
        id: segment.id,
        name: segment.name,
        wetSpeed: wet.speed,
        drySpeed: clear.speed,
        delta: clear.speed - wet.speed,
        wetLevel: wet.level,
      };
    }).filter(Boolean);
    const sorted = [...rows].sort((a, b) => b.delta - a.delta);
    const totalDelta = rows.reduce((a, r) => a + r.delta, 0) / Math.max(rows.length, 1);
    return { rows: sorted, meanDelta: totalDelta };
  }, [predictor, rain]);

  const alerts = useMemo(
    () =>
      buildAlerts({
        hourly: weather.hourly,
        states,
        rainNow: rain,
        trend,
      }),
    [weather.hourly, states, rain, trend],
  );

  const likeToday = useMemo(
    () =>
      historicalLikeToday({
        historical: chennai?.historical_baseline,
        states,
        rainNow: rain,
      }),
    [chennai, states, rain],
  );

  const exportCsv = () => {
    const rows = [
      ['time', 'precipitation_mm_h', 'temperature_c', 'wind_kmh', 'humidity_pct', 'visibility_m'],
      ...weather.hourly.map((h) => [
        h.time,
        h.precipitation,
        h.temperature,
        h.wind,
        h.humidity,
        h.visibility,
      ]),
    ];
    downloadText(
      `monsoonplus-forecast-${new Date().toISOString().slice(0, 10)}.csv`,
      `# monsoonplus hourly weather export\n# source: ${weather.source}\n${toCsv(rows)}`,
      'text/csv;charset=utf-8',
    );
  };

  if (!predictor) {
    return <EmptyState title="No model data">The forecast grid did not load.</EmptyState>;
  }

  return (
    <>
      <PageHead
        title="Forecast"
        caption="How hard the rain has to fall before each road gives way, and what the next 24 hours look like."
        actions={
          <button type="button" className="btn ghost small" onClick={exportCsv}>
            Export CSV
          </button>
        }
      />

      <Banner tier={alerts[0].tier === 'critical' ? 'critical' : alerts[0].tier === 'warning' ? 'warning' : 'info'} tag="ALERT">
        {alerts[0].title}
      </Banner>

      <Card
        title="Rainfall scenario"
        subtitle="Drives the model sweep below"
        actions={
          <>
            <Chip tone="rain">{rainRegime(rain)}</Chip>
            <Chip tone="neutral">{rainBucket(rain)}</Chip>
          </>
        }
      >
        <label className="field-label" htmlFor="forecast-rain">
          Rain intensity: <strong>{mm(rain)}</strong>
        </label>
        <input
          id="forecast-rain"
          type="range"
          min="0"
          max="100"
          value={rain}
          onChange={(event) => setRain(Number(event.target.value))}
        />
        <div className="ticks">
          <span>0 Dry</span>
          <span>25</span>
          <span>50</span>
          <span>75</span>
          <span>100 Extreme</span>
        </div>
        <div className="inline" style={{ marginTop: 10 }}>
          <span className="field-label">Trend</span>
          {['rising', 'steady', 'easing'].map((option) => (
            <button
              key={option}
              type="button"
              className={`btn ${trend === option ? '' : 'ghost'} small`}
              aria-pressed={trend === option}
              onClick={() => setTrend(option)}
            >
              {option[0].toUpperCase() + option.slice(1)}
            </button>
          ))}
        </div>
      </Card>

      <div className="grid g-4" style={{ marginTop: 14 }}>
        {kpis.map((kpi) => (
          <Stat key={kpi.label} label={`Network average ${kpi.label}`} value={kmh(kpi.value)} />
        ))}
      </div>

      {whatIfDry && (
        <Card
          title="What if it weren't raining?"
          subtitle={`Same model, two rainfall inputs: ${mm(rain)} now versus 0 mm at t+30`}
          actions={<Chip tone="neutral">Model estimate</Chip>}
          style={{ marginTop: 14 }}
        >
          <p style={{ margin: 0, fontWeight: 650 }}>
            At today's rainfall, the network loses{' '}
            <strong>{whatIfDry.meanDelta.toFixed(1)} km/h</strong> on average compared to a dry
            scenario, by the model's own t+30 prediction.
          </p>
          <TableScroll label="Data table, scroll horizontally">
            <table className="data" style={{ marginTop: 10 }}>
              <thead>
                <tr>
                  <th>Road</th>
                  <th className="num">Speed now (rain)</th>
                  <th className="num">Speed if dry</th>
                  <th className="num">Rain costs</th>
                  <th>Status (rain)</th>
                </tr>
              </thead>
              <tbody>
                {whatIfDry.rows.slice(0, 6).map((row) => (
                  <tr key={row.id}>
                    <td>{row.name}</td>
                    <td className="num">{kmh(row.wetSpeed)}</td>
                    <td className="num">{kmh(row.drySpeed)}</td>
                    <td className="num">
                      <strong>{row.delta >= 0 ? '-' : '+'}{Math.abs(row.delta).toFixed(1)} km/h</strong>
                    </td>
                    <td>
                      <StatusChip status={row.wetLevel.key} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </TableScroll>
          <Provenance>
            Both columns are the trained model's own t+30 output, run at two different rainfall
            inputs - rain as selected above, and rain pinned to 0. The gap is the model's learned
            sensitivity to rainfall, not a separate rule.
          </Provenance>
        </Card>
      )}

      <div className="grid g-2" style={{ marginTop: 14 }}>
        <Card
          title="Speed against rainfall"
          subtitle="Network mean at t+30, with the model's confidence band"
        >
          <div style={{ height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={sweep} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
                <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 3" />
                <XAxis dataKey="rain" tick={AXIS_STYLE} unit=" mm" stroke={CHART_GRID} />
                <YAxis tick={AXIS_STYLE} unit=" km/h" stroke={CHART_GRID} />
                <RTooltip
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid var(--line)' }}
                  formatter={(value, name) => [value, name === 'bandWidth' ? 'band width' : name]}
                />
                <Area
                  dataKey="lower"
                  stackId="band"
                  stroke="none"
                  fill="transparent"
                  isAnimationActive={false}
                />
                <Area
                  dataKey="bandWidth"
                  stackId="band"
                  stroke="none"
                  fill="var(--primary)"
                  fillOpacity={0.16}
                  name="confidence band"
                  isAnimationActive={false}
                />
                <Line
                  dataKey="speed"
                  stroke="var(--primary)"
                  strokeWidth={2.5}
                  dot={{ r: 3 }}
                  name="mean speed"
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <Provenance>
            Each point is the trained model run at that rainfall level. The band is the measured
            per-regime test error from <code>ml/reports/eval_synthetic_chennai.json</code>, widened
            with horizon - not a drawn-on guess.
          </Provenance>
        </Card>

        <Card
          title="Rain against predicted slowdown"
          subtitle="Every road at every sampled rainfall level"
        >
          {/* Recharts gives every scatter symbol role="img" with no accessible name,
              which is 100 unlabelled images to a screen reader and tells it nothing.
              The chart is hidden from assistive tech and replaced by the equivalent
              figures in text below -- the same approach used for the maps. */}
          <div style={{ height: 260 }} aria-hidden="true">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
                <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="rain"
                  name="rain"
                  unit=" mm/h"
                  tick={AXIS_STYLE}
                  stroke={CHART_GRID}
                />
                <YAxis
                  type="number"
                  dataKey="slowdown"
                  name="slowdown"
                  unit=" km/h"
                  tick={AXIS_STYLE}
                  stroke={CHART_GRID}
                />
                <RTooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid var(--line)' }}
                />
                <Scatter data={rainVsSlowdown} fill="var(--rain)" fillOpacity={0.55} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <p className="visually-hidden">
            Scatter plot of rainfall against predicted slowdown. Summarised:{' '}
            {scatterSummary.map((point) => (
              <span key={point.rain}>
                at {point.rain} millimetres per hour the mean slowdown below free-flow speed
                is {point.slowdown.toFixed(1)} kilometres per hour;{' '}
              </span>
            ))}
            Slowdown rises monotonically with rainfall across all {SEGMENTS.length} roads.
          </p>
          <Provenance>
            This is the "weather matters" claim as a picture: slowdown below free-flow rises with
            rainfall, and the spread at each level is the per-road sensitivity difference.
          </Provenance>
        </Card>
      </div>

      <h2 className="section-title">Next 24 hours</h2>
      <Card
        title="Hourly weather"
        subtitle={source.text}
        actions={<Chip tone={source.measured ? 'rain' : 'slow'}>{source.measured ? 'Measured' : 'Generated'}</Chip>}
      >
        {weather.state !== 'ready' ? (
          <EmptyState title="Loading weather" />
        ) : (
          <>
            {/* A horizontally scrolling region must be reachable by keyboard, or a
                keyboard-only user cannot see past the first few hours. tabIndex={0}
                plus a group role and label is what axe's scrollable-region-focusable
                rule asks for. */}
            <div
              className="inline"
              style={{ gap: 8, flexWrap: 'nowrap', overflowX: 'auto', paddingBottom: 6 }}
              tabIndex={0}
              role="group"
              aria-label="Hourly weather for the next 24 hours, scroll horizontally"
            >
              {weather.hourly.slice(0, 24).map((hour) => (
                <div
                  key={hour.time}
                  style={{
                    flex: '0 0 104px',
                    padding: '10px',
                    border: '1px solid var(--line)',
                    borderRadius: 'var(--radius-sm)',
                    background: hour.precipitation > 15 ? 'var(--rain-tint)' : 'var(--surface-sunk)',
                  }}
                >
                  <div className="field-label">{hour.time.slice(11, 16)}</div>
                  <div style={{ fontWeight: 850, fontSize: 15 }}>{hour.precipitation} mm</div>
                  <div className="small muted">
                    {hour.temperature}&deg;C
                    <br />
                    {hour.wind} km/h
                    <br />
                    {hour.humidity}% hum
                    <br />
                    {(hour.visibility / 1000).toFixed(1)} km vis
                  </div>
                </div>
              ))}
            </div>
            {!source.measured && (
              <Provenance>
                <strong>Generated series.</strong> No live weather connection, so this is a
                plausible northeast-monsoon day rather than a real forecast. Start the collector
                (README section 6) to replace it with Open-Meteo data - no API key needed.
              </Provenance>
            )}
          </>
        )}
      </Card>

      <div className="grid g-3" style={{ marginTop: 14 }}>
        <Card title="Active alerts" subtitle="Plain threshold rules">
          <div className="rows">
            {alerts.map((alert, index) => (
              <div key={index} className="row">
                <div className="row-top">
                  <div>
                    <div className="row-name">{alert.title}</div>
                    <div className="row-meta">{alert.detail}</div>
                  </div>
                  <Chip
                    tone={
                      alert.tier === 'critical' ? 'jammed' : alert.tier === 'warning' ? 'slow' : 'neutral'
                    }
                  >
                    {alert.rule || 'info'}
                  </Chip>
                </div>
              </div>
            ))}
          </div>
          <details style={{ marginTop: 10 }}>
            <summary className="small muted" style={{ cursor: 'pointer' }}>
              The full rule set
            </summary>
            <ul className="small muted" style={{ paddingLeft: 18, marginTop: 6 }}>
              {ALERT_RULES.map((rule) => (
                <li key={rule.id}>
                  <strong>{rule.id}</strong>: {rule.text}
                </li>
              ))}
            </ul>
          </details>
          <Provenance>
            These are <strong>if-this-then-that thresholds</strong> over the forecast, not a
            trained alert model. Only the traffic speeds they quote come from the network.
          </Provenance>
        </Card>

        <Card title="On days like today" subtitle="Historical baseline">
          {likeToday.available ? (
            <>
              <p style={{ margin: 0, fontWeight: 650 }}>{likeToday.text}</p>
              <Provenance>Derived from grouped historical data in {likeToday.source}.</Provenance>
            </>
          ) : (
            <EmptyState title="Not enough history">{likeToday.text}</EmptyState>
          )}
        </Card>

        <Card title="Air quality" subtitle="Open-Meteo air-quality API">
          {aqi ? (
            <>
              <div className="big">{aqi.us_aqi ?? '--'}</div>
              <p className="small muted" style={{ marginTop: 2 }}>
                US AQI - PM2.5 {aqi.pm2_5 ?? '--'}, PM10 {aqi.pm10 ?? '--'}
              </p>
              <div className="inline" style={{ marginTop: 8 }}>
                <Chip tone={aqi.measured ? 'rain' : 'slow'}>
                  {aqi.measured ? 'Measured' : 'Demo values'}
                </Chip>
                <Chip tone="neutral">Pollen: not published for Chennai</Chip>
              </div>
              {!aqi.measured && (
                <Provenance>
                  <strong>Demo values.</strong> The air-quality endpoint was unreachable.
                </Provenance>
              )}
            </>
          ) : (
            <EmptyState title="Loading air quality" />
          )}
        </Card>
      </div>

      <h2 className="section-title">Where each road breaks</h2>
      <Card subtitle="Status at t+30 as rainfall rises. Sorted by which road gives way first.">
        <TableScroll label="Data table, scroll horizontally">
          <table className="data">
            <thead>
              <tr>
                <th>Road</th>
                {[10, 25, 40, 55, 70, 85].map((level) => (
                  <th key={level}>{level} mm/h</th>
                ))}
                <th>Gives way at</th>
              </tr>
            </thead>
            <tbody>
              {whatIfTable.map((row) => (
                <tr key={row.id}>
                  <td>{row.segment}</td>
                  {[10, 25, 40, 55, 70, 85].map((level) => (
                    <td key={level}>
                      {row[`r${level}`] ? <StatusChip status={row[`r${level}`]} /> : '--'}
                    </td>
                  ))}
                  <td>
                    <strong>{row.breakPoint ? `${row.breakPoint} mm/h` : 'holds throughout'}</strong>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableScroll>
        <Provenance>
          "Gives way" means the model's t+30 prediction drops below{' '}
          {Math.round(THRESHOLDS.congestion_levels[0].min_speed_ratio * 100)}% of the road's
          free-flow speed.
        </Provenance>
      </Card>

      {scenarios && (
        <>
          <h2 className="section-title">Scenario battle</h2>
          <div className="grid g-2">
            {[
              { key: 'normal', title: 'Normal evening', detail: scenarios.normal, tone: 'smooth' },
              { key: 'monsoon', title: 'Heavy monsoon', detail: scenarios.monsoon, tone: 'jammed' },
            ].map((scenario) => (
              <Card
                key={scenario.key}
                title={scenario.title}
                subtitle={`Rain ${scenario.detail.rain} mm/h, t+30`}
              >
                <div className="big">{kmh(scenario.detail.mean)}</div>
                <p className="small muted" style={{ marginTop: 2 }}>
                  network average speed
                </p>
                <div style={{ marginTop: 10 }}>
                  <div className="kv">
                    <span className="muted">Roads disrupted</span>
                    <b>
                      {scenario.detail.disrupted} of {SEGMENTS.length}
                    </b>
                  </div>
                  <div className="kv">
                    <span className="muted">Roads jammed</span>
                    <b>{scenario.detail.jammed}</b>
                  </div>
                  <div className="kv">
                    <span className="muted">Flood-risk roads</span>
                    <b>{scenario.detail.flood}</b>
                  </div>
                  <div className="kv">
                    <span className="muted">Worst corridor</span>
                    <b>{scenario.detail.worst.segment.name}</b>
                  </div>
                </div>
              </Card>
            ))}
          </div>
          <Card title="The difference" style={{ marginTop: 14 }}>
            <p style={{ margin: 0, fontWeight: 650 }}>
              Going from a dry evening to 75 mm/h costs the network{' '}
              {(scenarios.normal.mean - scenarios.monsoon.mean).toFixed(1)} km/h on average and
              pushes {scenarios.monsoon.disrupted - scenarios.normal.disrupted} more roads into
              disruption.
            </p>
          </Card>
        </>
      )}
    </>
  );
}
