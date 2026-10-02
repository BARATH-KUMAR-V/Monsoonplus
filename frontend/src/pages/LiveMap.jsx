/**
 * Page 3 - Live Map.
 *
 * The time slider, the five layers, the satellite wetness overlay, the compare mode
 * and the auto-play all read the same exported forecast grid, so nothing here can
 * disagree with the Overview or the Forecast page.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from '@/data/store';
import NetworkMap, { LAYERS } from '@/components/NetworkMap';
import {
  Banner,
  Bar,
  Card,
  Chip,
  EmptyState,
  PageHead,
  Provenance,
  Segmented,
  StatusChip,
  TableScroll,
} from '@/components/ui';
import { impactBand } from '@/lib/model';
import { kmh, mm, offsetLabel } from '@/lib/format';

const STEPS = [0, 15, 30, 45, 60];

/** "What's normal for this road at this time?" from the exported baseline table. */
function useHistoricalBaseline(segmentId) {
  const { chennai } = useStore();
  return useMemo(() => {
    const baseline = chennai?.historical_baseline;
    if (!baseline?.per_segment?.[segmentId]) return null;
    const series = baseline.per_segment[segmentId];
    const now = new Date();
    const bucket = Math.floor((now.getHours() * 60 + now.getMinutes()) / baseline.bucket_minutes);
    return {
      typical: series[Math.min(bucket, series.length - 1)],
      series,
      label: baseline.label,
      bucketMinutes: baseline.bucket_minutes,
    };
  }, [chennai, segmentId]);
}

function SegmentPanel({ state, onClose, onCompare, comparing }) {
  const baseline = useHistoricalBaseline(state.segmentId);
  const anomaly = baseline ? state.speed - baseline.typical : null;

  return (
    <Card
      title={state.segment.name}
      subtitle={`${state.segment.note} - free flow ${kmh(state.segment.free_flow_kmh)}`}
      actions={
        <>
          <button type="button" className="btn ghost small" onClick={onCompare}>
            {comparing ? 'Remove from compare' : 'Add to compare'}
          </button>
          {onClose && (
            <button type="button" className="btn ghost small" onClick={onClose}>
              Close
            </button>
          )}
        </>
      }
    >
      <div className="inline" style={{ marginBottom: 10 }}>
        <StatusChip status={state.level.key} />
        <Chip tone="rain">Rain {mm(state.rain)}</Chip>
        <Chip tone={state.flood.index === 2 ? 'jammed' : state.flood.index === 1 ? 'slow' : 'neutral'}>
          Flood {state.flood.label}
        </Chip>
      </div>

      <div className="big">{kmh(state.speed)}</div>
      <p className="small muted" style={{ marginTop: 2 }}>
        Confidence band {Math.round(state.band[0])} to {Math.round(state.band[1])} km/h
        {' - '}
        {state.confidence.toLowerCase()} confidence
      </p>

      <div style={{ marginTop: 12 }}>
        <div className="field-label">Monsoon Impact Score</div>
        <div className="spread">
          <Bar value={state.impact} tone={impactBand(state.impact)} />
          <strong className="tnum" style={{ minWidth: 28, textAlign: 'right' }}>
            {state.impact}
          </strong>
        </div>
      </div>

      <div style={{ marginTop: 12 }}>
        <div className="kv">
          <span className="muted">Now</span>
          <b>{kmh(state.speedNow)}</b>
        </div>
        <div className="kv">
          <span className="muted">In 15 min</span>
          <b>{kmh(state.predicted['t+15'])}</b>
        </div>
        <div className="kv">
          <span className="muted">In 30 min</span>
          <b>{kmh(state.predicted['t+30'])}</b>
        </div>
        <div className="kv">
          <span className="muted">In 60 min</span>
          <b>{kmh(state.predicted['t+60'])}</b>
        </div>
      </div>

      {baseline && (
        <div style={{ marginTop: 12 }}>
          <div className="field-label">What is normal here at this time?</div>
          <p className="small" style={{ margin: '4px 0 0' }}>
            Typically <strong>{kmh(baseline.typical)}</strong>.{' '}
            {anomaly != null && Math.abs(anomaly) > 1.5 ? (
              <span style={{ color: anomaly < 0 ? 'var(--jammed-text)' : 'var(--smooth-text)' }}>
                Currently {Math.abs(anomaly).toFixed(1)} km/h {anomaly < 0 ? 'slower' : 'faster'} than usual.
              </span>
            ) : (
              <span>Currently about normal.</span>
            )}
          </p>
          <Provenance>
            Baseline is the per-road, per-time-of-day average over the training split of{' '}
            {baseline.label}. Same table the historical-average baseline uses in Model Lab.
          </Provenance>
        </div>
      )}

      <div style={{ marginTop: 12 }}>
        <div className="field-label">Deterioration over the hour</div>
        <p className="small" style={{ margin: '4px 0 0' }}>
          {state.speedIn60 < state.speedNow - 1
            ? `Expected to lose ${(state.speedNow - state.speedIn60).toFixed(1)} km/h over the next hour.`
            : state.speedIn60 > state.speedNow + 1
              ? `Expected to recover ${(state.speedIn60 - state.speedNow).toFixed(1)} km/h over the next hour.`
              : 'Expected to hold roughly steady over the next hour.'}
        </p>
      </div>
    </Card>
  );
}

export default function LiveMap() {
  const { rain, setRain, states, statesAt, satelliteEnabled } = useStore();
  const [layer, setLayer] = useState('traffic');
  const [offset, setOffset] = useState(0);
  const [selectedId, setSelectedId] = useState('r2');
  const [compareIds, setCompareIds] = useState([]);
  const [playing, setPlaying] = useState(false);
  const [whatIfEarlier, setWhatIfEarlier] = useState(15);
  const playTimer = useRef(null);

  const framed = useMemo(() => statesAt(offset), [statesAt, offset]);
  const selected = framed.find((s) => s.segmentId === selectedId) || framed[0];

  // Auto-play: step Now -> +60 and back to the start.
  useEffect(() => {
    if (!playing) {
      if (playTimer.current) clearInterval(playTimer.current);
      return undefined;
    }
    playTimer.current = setInterval(() => {
      setOffset((current) => {
        const index = STEPS.indexOf(current);
        return STEPS[(index + 1) % STEPS.length];
      });
    }, 1400);
    return () => clearInterval(playTimer.current);
  }, [playing]);

  const toggleCompare = useCallback((segmentId) => {
    setCompareIds((current) =>
      current.includes(segmentId)
        ? current.filter((id) => id !== segmentId)
        : [...current, segmentId].slice(-4),
    );
  }, []);

  const headline = useMemo(() => {
    if (!framed.length) return '';
    const disrupted = framed.filter((s) => s.level.index >= 1).length;
    const mean = framed.reduce((a, s) => a + s.speed, 0) / framed.length;
    const when = offset === 0 ? 'Right now' : `In ${offset} minutes`;
    if (disrupted === 0) return `${when}: the whole network is running smoothly at about ${kmh(mean)}.`;
    const worst = [...framed].sort((a, b) => b.impact - a.impact)[0];
    return `${when}: ${disrupted} of ${framed.length} roads disrupted, averaging ${kmh(
      mean,
    )}. ${worst.segment.name} is worst hit.`;
  }, [framed, offset]);

  // What-if: compare leaving now against leaving `whatIfEarlier` minutes earlier.
  const whatIf = useMemo(() => {
    if (!selected) return null;
    const earlierStates = statesAt(Math.max(0, offset - whatIfEarlier));
    const earlier = earlierStates.find((s) => s.segmentId === selected.segmentId);
    if (!earlier) return null;
    const speedGain = earlier.speed - selected.speed;
    const lengthKm = 3.5;
    const minutesNow = (lengthKm / Math.max(selected.speed, 1)) * 60;
    const minutesEarlier = (lengthKm / Math.max(earlier.speed, 1)) * 60;
    return {
      speedGain,
      minutesSaved: minutesNow - minutesEarlier,
      earlierSpeed: earlier.speed,
    };
  }, [selected, statesAt, offset, whatIfEarlier]);

  if (!states.length) {
    return <EmptyState title="No model data">The forecast grid did not load.</EmptyState>;
  }

  const compareStates = framed.filter((s) => compareIds.includes(s.segmentId));

  return (
    <>
      <PageHead
        title="Live Map"
        caption="Scrub forward an hour and watch the network change. Click any road for its detail panel."
        actions={
          <Segmented
            label="Map layer"
            value={layer}
            onChange={setLayer}
            options={LAYERS.map((l) => ({ value: l.id, label: l.label }))}
          />
        }
      />

      <Banner tier={framed.some((s) => s.level.index === 3) ? 'warning' : 'info'} tag={offsetLabel(offset).toUpperCase()}>
        {headline}
      </Banner>

      <div className="grid g-wide">
        <div className="stack">
          <Card style={{ padding: 0 }}>
            <NetworkMap
              states={framed}
              layer={layer}
              onSelect={setSelectedId}
              selectedId={selectedId}
              height={450}
              showWetness={satelliteEnabled && (layer === 'flood' || layer === 'combined')}
              topLeft={
                <span>
                  {offsetLabel(offset)} - {mm(rain)}
                </span>
              }
            />
          </Card>

          <Card title="Time" subtitle="Now through the next hour">
            <div className="spread" style={{ marginBottom: 8 }}>
              <div className="inline">
                <button
                  type="button"
                  className="btn small"
                  onClick={() => setPlaying((p) => !p)}
                  aria-pressed={playing}
                >
                  {playing ? 'Pause' : 'Play next hour'}
                </button>
                <button
                  type="button"
                  className="btn ghost small"
                  onClick={() => {
                    setPlaying(false);
                    setOffset(0);
                  }}
                >
                  Reset
                </button>
              </div>
              <strong>{offsetLabel(offset)}</strong>
            </div>
            <input
              type="range"
              min="0"
              max="60"
              step="15"
              value={offset}
              onChange={(event) => {
                setPlaying(false);
                setOffset(Number(event.target.value));
              }}
              aria-label="Forecast time offset in minutes"
              aria-valuetext={offsetLabel(offset)}
            />
            <div className="ticks">
              {STEPS.map((step) => (
                <span key={step}>{step === 0 ? 'Now' : `+${step}`}</span>
              ))}
            </div>

            <div style={{ marginTop: 14 }}>
              <label className="field-label" htmlFor="map-rain">
                Rainfall {mm(rain)}
              </label>
              <input
                id="map-rain"
                type="range"
                min="0"
                max="100"
                value={rain}
                onChange={(event) => setRain(Number(event.target.value))}
              />
            </div>
            <Provenance>
              +15, +30 and +60 are direct model outputs. +45 is interpolated between the +30 and
              +60 predictions - arithmetic on model outputs, not a separate forecast.
            </Provenance>
          </Card>
        </div>

        <div className="stack">
          {selected && (
            <SegmentPanel
              state={selected}
              comparing={compareIds.includes(selected.segmentId)}
              onCompare={() => toggleCompare(selected.segmentId)}
            />
          )}

          <Card title="If you left earlier" subtitle="Same road, earlier departure">
            <div className="inline" style={{ marginBottom: 10 }}>
              {[15, 30, 45].map((value) => (
                <button
                  key={value}
                  type="button"
                  className={`btn ${whatIfEarlier === value ? '' : 'ghost'} small`}
                  aria-pressed={whatIfEarlier === value}
                  onClick={() => setWhatIfEarlier(value)}
                >
                  {value} min earlier
                </button>
              ))}
            </div>
            {whatIf && selected ? (
              <p style={{ margin: 0, fontWeight: 650 }}>
                {offset - whatIfEarlier < 0
                  ? `Already at the earliest point in the window - move the time slider forward to compare.`
                  : whatIf.minutesSaved > 0.3
                    ? `Leaving ${whatIfEarlier} min earlier, ${selected.segment.name} runs at ${kmh(
                        whatIf.earlierSpeed,
                      )} instead of ${kmh(selected.speed)} - about ${whatIf.minutesSaved.toFixed(
                        1,
                      )} min saved over a 3.5 km stretch.`
                    : `Leaving ${whatIfEarlier} min earlier makes no meaningful difference on ${selected.segment.name}.`}
              </p>
            ) : (
              <EmptyState title="Select a road" />
            )}
          </Card>
        </div>
      </div>

      <h2 className="section-title">
        Compare roads
        {compareIds.length > 0 && (
          <button type="button" className="btn ghost small" onClick={() => setCompareIds([])}>
            Clear
          </button>
        )}
      </h2>
      <Card>
        {compareStates.length === 0 ? (
          <EmptyState title="Nothing to compare yet">
            Click a road on the map, then press "Add to compare". Up to four at a time.
          </EmptyState>
        ) : (
          <TableScroll label="Data table, scroll horizontally">
            <table className="data">
              <caption className="visually-hidden">Comparison of selected roads</caption>
              <thead>
                <tr>
                  <th>Road</th>
                  <th>Status</th>
                  <th className="num">Now</th>
                  <th className="num">+15</th>
                  <th className="num">+30</th>
                  <th className="num">+60</th>
                  <th className="num">Rain</th>
                  <th>Flood</th>
                  <th className="num">Impact</th>
                </tr>
              </thead>
              <tbody>
                {compareStates.map((state) => (
                  <tr key={state.segmentId}>
                    <td>{state.segment.name}</td>
                    <td>
                      <StatusChip status={state.level.key} />
                    </td>
                    <td className="num">{Math.round(state.speedNow)}</td>
                    <td className="num">{Math.round(state.predicted['t+15'])}</td>
                    <td className="num">{Math.round(state.predicted['t+30'])}</td>
                    <td className="num">{Math.round(state.predicted['t+60'])}</td>
                    <td className="num">{state.rain.toFixed(1)}</td>
                    <td>{state.flood.label}</td>
                    <td className="num">{state.impact}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </TableScroll>
        )}
      </Card>

      <h2 className="section-title">All roads at {offsetLabel(offset)}</h2>
      <Card>
        <div className="rows">
          {[...framed]
            .sort((a, b) => b.impact - a.impact)
            .map((state) => (
              <button
                key={state.segmentId}
                type="button"
                className={`row ${selectedId === state.segmentId ? 'selected' : ''}`}
                aria-pressed={selectedId === state.segmentId}
                onClick={() => setSelectedId(state.segmentId)}
              >
                <div className="row-top">
                  <div>
                    <div className="row-name">{state.segment.name}</div>
                    <div className="row-meta">
                      {kmh(state.speed)} - rain {mm(state.rain)} - flood {state.flood.label}
                    </div>
                  </div>
                  <div className="inline">
                    <StatusChip status={state.level.key} />
                    <strong className="tnum">{state.impact}</strong>
                  </div>
                </div>
                <Bar value={state.impact} tone={impactBand(state.impact)} />
              </button>
            ))}
        </div>
        <Provenance>
          This list carries the same information as the map above. The map is an enhancement -
          every number on it is reachable by keyboard here.
        </Provenance>
      </Card>
    </>
  );
}
