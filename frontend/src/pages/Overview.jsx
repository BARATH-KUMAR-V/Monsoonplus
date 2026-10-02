/**
 * Page 1 - Overview. The rain-day briefing.
 *
 * Everything on this page is computed from the current rain state against the trained
 * model's exported grid. Nothing is hardcoded: move the rain slider and all four KPIs,
 * the priority ranking, the alerts and the map recolour together.
 */
import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { useStore } from '@/data/store';
import NetworkMap from '@/components/NetworkMap';
import {
  Banner,
  Bar,
  Card,
  Chip,
  EmptyState,
  FreshnessDot,
  PageHead,
  Provenance,
  Stat,
  StatusChip,
} from '@/components/ui';
import { ATTRIBUTIONS, monsoonRegime } from '@/config/network';
import { impactBand, rainBucket, rainRegime } from '@/lib/model';
import { buildAlerts } from '@/lib/alerts';
import { kmh, mm, relativeTime } from '@/lib/format';

function RainControl({ rain, setRain, trend, setTrend }) {
  return (
    <Card
      title="Rain right now"
      subtitle="Drives every number on this page"
      actions={<Chip tone="rain">{rainRegime(rain)}</Chip>}
    >
      <label htmlFor="rain-slider" className="field-label">
        Rainfall intensity: <strong>{mm(rain)}</strong> ({rainBucket(rain)})
      </label>
      <input
        id="rain-slider"
        type="range"
        min="0"
        max="100"
        step="1"
        value={rain}
        onChange={(event) => setRain(Number(event.target.value))}
        aria-valuetext={`${rain} millimetres per hour, ${rainBucket(rain)}`}
      />
      <div className="ticks">
        <span>0</span>
        <span>25</span>
        <span>50</span>
        <span>75</span>
        <span>100 mm/h</span>
      </div>
      <div className="inline" style={{ marginTop: 10 }}>
        <span className="field-label" style={{ marginRight: 4 }}>
          Trend
        </span>
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
  );
}

function DataHealth() {
  const { chennai, mode, live, satelliteEnabled } = useStore();

  const generatedAt = chennai?.generated_at;
  const ageMinutes = generatedAt
    ? (Date.now() - new Date(generatedAt).getTime()) / 60000
    : null;

  const sources = [
    {
      name: 'Traffic',
      detail:
        mode === 'live'
          ? live.state === 'ready'
            ? 'TomTom via local collector'
            : 'collector unreachable'
          : 'Synthetic Chennai window (precomputed)',
      ok: mode === 'live' ? live.state === 'ready' : true,
      age: mode === 'live' ? (live.fetchedAt ? (Date.now() - new Date(live.fetchedAt).getTime()) / 60000 : null) : 0,
    },
    {
      name: 'Weather',
      detail:
        mode === 'live'
          ? live.state === 'ready'
            ? 'Open-Meteo (no key needed)'
            : 'collector unreachable'
          : 'Synthetic rainfall sweep (precomputed)',
      ok: mode === 'live' ? live.state === 'ready' : true,
      age: mode === 'live' ? (live.fetchedAt ? (Date.now() - new Date(live.fetchedAt).getTime()) / 60000 : null) : 0,
    },
    {
      name: 'Satellite',
      detail: satelliteEnabled
        ? 'Synthetic NDWI (no Earth Engine export configured)'
        : 'switched off by the user',
      ok: satelliteEnabled,
      age: satelliteEnabled ? 0 : null,
    },
  ];

  return (
    <Card
      title="Data health"
      subtitle={
        generatedAt
          ? `Dataset exported ${relativeTime(generatedAt)}`
          : 'Freshness per input source'
      }
    >
      <div className="stack">
        {sources.map((source) => (
          <div key={source.name} className="spread" style={{ gap: 8 }}>
            <div>
              <div style={{ fontWeight: 750, fontSize: 12.5 }}>{source.name}</div>
              <div className="small muted">{source.detail}</div>
            </div>
            <FreshnessDot ageMinutes={source.age} ok={source.ok} />
          </div>
        ))}
      </div>

      <div style={{ marginTop: 14, paddingTop: 12, borderTop: '1px solid var(--line)' }}>
        <div className="field-label">Attribution</div>
        <ul style={{ margin: '6px 0 0', paddingLeft: 16, fontSize: 11, color: 'var(--muted)' }}>
          {ATTRIBUTIONS.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      </div>
      {ageMinutes != null && ageMinutes > 60 * 24 * 7 && (
        <Provenance>
          This dataset export is over a week old. Re-run{' '}
          <code>python -m ml.export.export_predictions</code> to refresh it.
        </Provenance>
      )}
    </Card>
  );
}

function MonsoonBadge() {
  const now = new Date();
  const regime = monsoonRegime(now.getMonth() + 1);
  const isPeak = regime.label.includes('peak');
  return (
    <Card title="Monsoon regime" subtitle={`${now.toLocaleString('en-IN', { month: 'long' })} in Chennai`}>
      <div className="inline">
        <Chip tone={isPeak ? 'flood' : 'rain'}>{regime.label}</Chip>
      </div>
      <p className="small muted" style={{ marginTop: 10, marginBottom: 0 }}>
        {regime.note}
      </p>
      <Provenance>
        Chennai is driven by the <strong>northeast</strong> monsoon (October to December),
        unlike most of India. Onset October, peak November, withdrawal December.
      </Provenance>
    </Card>
  );
}

export default function Overview() {
  const { rain, setRain, trend, setTrend, states, statesAt, savedRoutes, removeRoute, chennai } =
    useStore();
  const [layer] = useState('combined');

  const future = useMemo(() => statesAt(60), [statesAt]);

  const kpis = useMemo(() => {
    if (!states.length) return null;
    const meanSpeed = states.reduce((a, s) => a + s.speed, 0) / states.length;
    const meanFuture = future.length
      ? future.reduce((a, s) => a + s.speed, 0) / future.length
      : meanSpeed;
    const disrupted = states.filter((s) => s.level.index >= 1).length;
    const floodRoads = states.filter((s) => s.flood.index >= 1).length;
    const worst = [...states].sort((a, b) => b.impact - a.impact)[0];
    return { meanSpeed, meanFuture, disrupted, floodRoads, worst };
  }, [states, future]);

  const ranked = useMemo(() => [...states].sort((a, b) => b.impact - a.impact), [states]);

  const alerts = useMemo(
    () =>
      buildAlerts({
        hourly: [],
        states,
        rainNow: rain,
        trend,
      }),
    [states, rain, trend],
  );

  if (!states.length) {
    return <EmptyState title="No model data">The forecast grid did not load.</EmptyState>;
  }

  const topAlert = alerts[0];
  const bannerTier =
    topAlert.tier === 'critical' ? 'critical' : topAlert.tier === 'warning' ? 'warning' : 'info';

  return (
    <>
      <PageHead
        title="Overview"
        caption="What the monsoon is doing to Chennai's roads right now, and which roads to watch."
      />

      <Banner tier={bannerTier} tag={topAlert.tier.toUpperCase()}>
        {topAlert.title}
      </Banner>

      <div className="grid g-4">
        <Stat
          label="Average speed"
          value={kmh(kpis.meanSpeed)}
          note={`${kmh(kpis.meanFuture)} forecast in 60 min`}
        />
        <Stat
          label="Roads disrupted"
          value={`${kpis.disrupted} of ${states.length}`}
          note="At Slow or worse"
          tone={kpis.disrupted > states.length / 2 ? 'jammed' : kpis.disrupted ? 'slow' : 'smooth'}
        />
        <Stat
          label="Flood-risk roads"
          value={`${kpis.floodRoads}`}
          note="Moderate or High flood signal"
          tone={kpis.floodRoads > 2 ? 'jammed' : kpis.floodRoads ? 'slow' : 'smooth'}
        />
        <Stat
          label="Worst corridor"
          value={kpis.worst.segment.name.split(' - ')[0]}
          note={`Impact score ${kpis.worst.impact} - ${kpis.worst.level.label}`}
          tone={impactBand(kpis.worst.impact)}
        />
      </div>

      <div className="grid g-wide" style={{ marginTop: 14 }}>
        <Card title="City at a glance" subtitle="Coloured by Monsoon Impact Score">
          <NetworkMap states={states} layer={layer} height={380} />
          <Provenance>
            Impact Score = 0.40 x traffic + 0.25 x rain + 0.20 x flood + 0.15 x deterioration.
            Weights live in <code>config/segments.json</code>; speeds are model outputs.
          </Provenance>
        </Card>

        <div className="stack">
          <RainControl rain={rain} setRain={setRain} trend={trend} setTrend={setTrend} />
          <MonsoonBadge />
        </div>
      </div>

      <h2 className="section-title">
        Priority roads
        <Link to="/map" className="btn ghost small">
          Open the map
        </Link>
      </h2>
      <Card>
        <div className="rows">
          {ranked.map((state) => (
            <div key={state.segmentId} className="row">
              <div className="row-top">
                <div>
                  <div className="row-name">{state.segment.name}</div>
                  <div className="row-meta">
                    {kmh(state.speed)} now - {kmh(state.predicted['t+60'])} in 60 min - {mm(state.rain)}
                  </div>
                </div>
                <div className="inline">
                  <StatusChip status={state.level.key} />
                  <strong className="tnum" style={{ fontSize: 15 }}>
                    {state.impact}
                  </strong>
                </div>
              </div>
              <Bar value={state.impact} tone={impactBand(state.impact)} />
              <div className="row-foot">
                <span>Flood {state.flood.label}</span>
                <span>{state.segment.note}</span>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <div className="grid g-3" style={{ marginTop: 14 }}>
        <Card
          title="My routes"
          subtitle={savedRoutes.length ? `${savedRoutes.length} saved` : 'None saved yet'}
        >
          {savedRoutes.length === 0 ? (
            <EmptyState title="No saved routes">
              Plan a trip on Smart Trip and save it to see it here.
            </EmptyState>
          ) : (
            <div className="rows">
              {savedRoutes.map((route, index) => (
                <div key={`${route.from}-${route.to}`} className="row">
                  <div className="row-top">
                    <div>
                      <div className="row-name">
                        {route.fromName} to {route.toName}
                      </div>
                      <div className="row-meta">{route.via}</div>
                    </div>
                    <button
                      type="button"
                      className="btn ghost small"
                      onClick={() => removeRoute(index)}
                      aria-label={`Remove saved route ${route.fromName} to ${route.toName}`}
                    >
                      Remove
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>

        <Card title="Alert log" subtitle="Threshold rules, not a model">
          <div className="rows">
            {alerts.map((alert, index) => (
              <div key={`${alert.rule}-${index}`} className="row">
                <div className="row-top">
                  <div>
                    <div className="row-name">{alert.title}</div>
                    <div className="row-meta">{alert.detail}</div>
                  </div>
                  <Chip tone={alert.tier === 'critical' ? 'jammed' : alert.tier === 'warning' ? 'slow' : 'neutral'}>
                    {alert.rule || 'info'}
                  </Chip>
                </div>
              </div>
            ))}
          </div>
        </Card>

        <DataHealth />
      </div>

      <Provenance>
        <strong>What you are looking at:</strong> {chennai?.label}. Speeds are forecasts from a
        trimodal gated-fusion graph network, not measurements. This is a planning aid, not a
        safety system - never use it to judge whether a flooded road is passable.
      </Provenance>
    </>
  );
}
