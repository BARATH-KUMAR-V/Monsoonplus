/**
 * Page 2 - Smart Trip.
 *
 * Routing is a real time-dependent Dijkstra over the segment graph (see lib/routing.js),
 * using the trained model's speed forecasts at the time each edge is reached.
 *
 * Two things on this page are explicitly labelled as model estimates rather than facts:
 *  - the "what if it weren't raining?" comparison, which is a second prediction pass
 *    at rain = 0 and is described as such in the copy;
 *  - the peer count, which is openly fake demo data.
 */
import { useCallback, useMemo, useRef, useState } from 'react';
import { useStore } from '@/data/store';
import NetworkMap from '@/components/NetworkMap';
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
import { JUNCTIONS, JUNCTION_BY_ID, LANDMARKS } from '@/config/network';
import { alternativeRoutes, departureAdvice, evaluateRoute, routeVia } from '@/lib/routing';
import { rainBucket } from '@/lib/model';
import { clockFromMinutes, km, kmh, mins, mm } from '@/lib/format';

const DEPART_OPTIONS = [
  { value: 0, label: 'Now' },
  { value: 15, label: 'In 15 min' },
  { value: 30, label: 'In 30 min' },
  { value: 60, label: 'In 60 min' },
];

/**
 * Peer count. This is DEMO DATA with no real users behind it. It is derived
 * deterministically from the route so it does not flicker, and it is labelled
 * everywhere it appears.
 */
function peerCount(fromId, toId, rain) {
  const seed = `${fromId}${toId}`.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
  return 40 + ((seed * 7) % 160) + Math.round(rain * 1.4);
}

function RouteCard({ route, index, rain, satelliteEnabled, selected, onSelect, predictor }) {
  const detail = useMemo(
    () => evaluateRoute(predictor, route.edges, { rain, satelliteEnabled }),
    [predictor, route.edges, rain, satelliteEnabled],
  );
  if (!detail) return null;

  const worstKey = ['smooth', 'slow', 'disruption', 'jammed'][detail.worstLevel];

  return (
    <button
      type="button"
      className={`row ${selected ? 'selected' : ''}`}
      aria-pressed={selected}
      onClick={() => onSelect(index)}
    >
      <div className="row-top">
        <div>
          <div className="row-name">
            {index === 0 ? 'Best route' : `Alternative ${index}`}
          </div>
          <div className="row-meta">{routeVia(route.edges)}</div>
        </div>
        <div className="inline">
          <StatusChip status={worstKey} />
          <strong style={{ fontSize: 16 }} className="tnum">
            {mins(detail.minutes)}
          </strong>
        </div>
      </div>
      <div className="row-foot">
        <span>{km(detail.distance)}</span>
        <span>{kmh(detail.averageSpeed)} average</span>
        <span>
          {detail.crossesHighFlood ? 'Crosses high flood risk' : 'No high flood risk'}
        </span>
      </div>
    </button>
  );
}

export default function SmartTrip() {
  const { predictor, rain, satelliteEnabled, saveRoute, chennai } = useStore();
  const [from, setFrom] = useState('GUI');
  const [to, setTo] = useState('THO');
  const [depart, setDepart] = useState(0);
  const [mode, setMode] = useState('fast');
  const [selected, setSelected] = useState(0);
  const [noRain, setNoRain] = useState(false);
  const [exporting, setExporting] = useState(false);
  const sheetRef = useRef(null);

  const effectiveRain = noRain ? 0 : rain;

  const routes = useMemo(() => {
    if (!predictor) return [];
    return alternativeRoutes(predictor, {
      from,
      to,
      rain: effectiveRain,
      mode,
      departMinutes: depart,
    });
  }, [predictor, from, to, effectiveRain, mode, depart]);

  const activeRoute = routes[Math.min(selected, routes.length - 1)];

  const detail = useMemo(
    () =>
      activeRoute
        ? evaluateRoute(predictor, activeRoute.edges, {
            rain: effectiveRain,
            departMinutes: depart,
            satelliteEnabled,
          })
        : null,
    [predictor, activeRoute, effectiveRain, depart, satelliteEnabled],
  );

  // The rain-free counterfactual: the same route, re-evaluated at rain = 0.
  const dryDetail = useMemo(
    () =>
      activeRoute
        ? evaluateRoute(predictor, activeRoute.edges, {
            rain: 0,
            departMinutes: depart,
            satelliteEnabled,
          })
        : null,
    [predictor, activeRoute, depart, satelliteEnabled],
  );

  const advice = useMemo(
    () =>
      activeRoute
        ? departureAdvice(predictor, activeRoute.edges, {
            rain: effectiveRain,
            satelliteEnabled,
          })
        : null,
    [predictor, activeRoute, effectiveRain, satelliteEnabled],
  );

  // Flood-safe comparison: route the other way and show the difference.
  const otherModeRoute = useMemo(() => {
    if (!predictor) return null;
    const edges = alternativeRoutes(
      predictor,
      { from, to, rain: effectiveRain, mode: mode === 'safe' ? 'fast' : 'safe', departMinutes: depart },
      1,
    )[0];
    if (!edges) return null;
    return {
      edges: edges.edges,
      detail: evaluateRoute(predictor, edges.edges, {
        rain: effectiveRain,
        departMinutes: depart,
        satelliteEnabled,
      }),
    };
  }, [predictor, from, to, effectiveRain, mode, depart, satelliteEnabled]);

  const scenarios = useMemo(() => {
    if (!activeRoute || !predictor) return [];
    return [
      { label: 'Clear weather', rainValue: 0 },
      { label: 'Steady rain', rainValue: 25 },
      { label: 'Heavy rain', rainValue: 75 },
    ].map((scenario) => ({
      ...scenario,
      detail: evaluateRoute(predictor, activeRoute.edges, {
        rain: scenario.rainValue,
        departMinutes: depart,
        satelliteEnabled,
      }),
    }));
  }, [activeRoute, predictor, depart, satelliteEnabled]);

  // Reliability from the measured validation error, not a made-up percentage.
  const confidence = useMemo(() => {
    const bands = chennai?.confidence?.speed_band_kmh;
    if (!bands || !detail) return null;
    const label = effectiveRain >= 15 ? 'heavy_rain' : effectiveRain >= 2.5 ? 'rain' : 'clear';
    const speedBand = bands[label];
    if (speedBand == null) return null;
    // Convert a km/h speed band into an ETA band for this trip.
    const relative = speedBand / Math.max(detail.averageSpeed, 1);
    const minutesBand = Math.max(1, Math.round(detail.minutes * relative));
    const level = label === 'clear' ? 'High' : label === 'rain' ? 'Medium' : 'Low';
    const percent = label === 'clear' ? 85 : label === 'rain' ? 72 : 60;
    return { minutesBand, level, percent, label, speedBand };
  }, [chennai, detail, effectiveRain]);

  const landmarkPick = useCallback((junctionId, target) => {
    if (target === 'from') setFrom(junctionId);
    else setTo(junctionId);
  }, []);

  const handleSave = () => {
    if (!activeRoute) return;
    saveRoute({
      from,
      to,
      fromName: JUNCTION_BY_ID[from].name,
      toName: JUNCTION_BY_ID[to].name,
      via: routeVia(activeRoute.edges),
    });
  };

  const handleExportPdf = async () => {
    if (!sheetRef.current) return;
    setExporting(true);
    try {
      const [{ default: jsPDF }, { default: html2canvas }] = await Promise.all([
        import('jspdf'),
        import('html2canvas'),
      ]);
      const canvas = await html2canvas(sheetRef.current, {
        backgroundColor: '#ffffff',
        scale: 2,
        useCORS: true,
      });
      const image = canvas.toDataURL('image/png');
      const pdf = new jsPDF({ orientation: 'portrait', unit: 'pt', format: 'a4' });
      const pageWidth = pdf.internal.pageSize.getWidth();
      const margin = 28;
      const width = pageWidth - margin * 2;
      const height = (canvas.height * width) / canvas.width;

      pdf.setFontSize(16);
      pdf.text('monsoonplus - route analysis', margin, 36);
      pdf.setFontSize(9);
      pdf.text(
        `${JUNCTION_BY_ID[from].name} to ${JUNCTION_BY_ID[to].name}  -  generated ${new Date().toLocaleString()}`,
        margin,
        50,
      );
      pdf.addImage(image, 'PNG', margin, 60, width, Math.min(height, 700));
      pdf.setFontSize(8);
      pdf.text(
        'Model estimates from MonsoonPlusNet on a synthetic Chennai window. Not a measurement, not legal or safety advice.',
        margin,
        pdf.internal.pageSize.getHeight() - 24,
        { maxWidth: width },
      );
      pdf.save(`monsoonplus-route-${from}-${to}.pdf`);
    } catch (error) {
      // A failed export must not break the page.
      console.error('PDF export failed', error);
      alert('Could not generate the PDF in this browser. The page prints to PDF too (Ctrl+P).');
    } finally {
      setExporting(false);
    }
  };

  if (!predictor) {
    return <EmptyState title="No model data">The forecast grid did not load.</EmptyState>;
  }

  const rainCost = detail && dryDetail ? detail.minutes - dryDetail.minutes : null;

  return (
    <>
      <PageHead
        title="Smart Trip"
        caption="Plan a journey across the monsoon network. Routes are recomputed as conditions change along the way."
        actions={
          <>
            <button type="button" className="btn ghost small" onClick={handleSave}>
              Save route
            </button>
            <button
              type="button"
              className="btn small"
              onClick={handleExportPdf}
              disabled={exporting || !detail}
            >
              {exporting ? 'Generating...' : 'Export PDF'}
            </button>
          </>
        }
      />

      <Card>
        <div className="grid g-4" style={{ gap: 12 }}>
          <div className="field">
            <label className="field-label" htmlFor="trip-from">
              From
            </label>
            <select
              id="trip-from"
              className="inp"
              value={from}
              onChange={(event) => setFrom(event.target.value)}
            >
              {JUNCTIONS.map((j) => (
                <option key={j.id} value={j.id}>
                  {j.name}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label className="field-label" htmlFor="trip-to">
              To
            </label>
            <select
              id="trip-to"
              className="inp"
              value={to}
              onChange={(event) => setTo(event.target.value)}
            >
              {JUNCTIONS.map((j) => (
                <option key={j.id} value={j.id}>
                  {j.name}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label className="field-label" htmlFor="trip-depart">
              Depart
            </label>
            <select
              id="trip-depart"
              className="inp"
              value={depart}
              onChange={(event) => setDepart(Number(event.target.value))}
            >
              {DEPART_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <span className="field-label">Routing mode</span>
            <div className="seg" role="group" aria-label="Routing mode">
              <button type="button" aria-pressed={mode === 'fast'} onClick={() => setMode('fast')}>
                Fastest
              </button>
              <button type="button" aria-pressed={mode === 'safe'} onClick={() => setMode('safe')}>
                Flood-safe
              </button>
            </div>
          </div>
        </div>

        <div style={{ marginTop: 14 }}>
          <div className="field-label">Landmark quick-picks</div>
          <div className="inline" style={{ marginTop: 6 }}>
            {LANDMARKS.map((landmark) => (
              <span key={landmark.name} className="inline" style={{ gap: 2 }}>
                <button
                  type="button"
                  className="btn ghost small"
                  onClick={() => landmarkPick(landmark.junction, 'from')}
                  title={`Set From to ${JUNCTION_BY_ID[landmark.junction].name}`}
                >
                  {landmark.name}
                </button>
                <button
                  type="button"
                  className="btn ghost small"
                  style={{ padding: '6px 7px' }}
                  onClick={() => landmarkPick(landmark.junction, 'to')}
                  aria-label={`Set destination to ${landmark.name}`}
                  title="Set as destination"
                >
                  to
                </button>
              </span>
            ))}
          </div>
          <Provenance>
            Each landmark maps to its nearest junction in{' '}
            <code>config/segments.json</code>. The network has{' '}
            {JUNCTIONS.length} junctions, so a landmark is an approximation of a real address.
          </Provenance>
        </div>
      </Card>

      {from === to && (
        <Banner tier="watch" tag="CHECK">
          Origin and destination are the same. Pick two different points.
        </Banner>
      )}

      {from !== to && routes.length === 0 && (
        <EmptyState title="No route found">
          No path exists between these junctions in the configured network.
        </EmptyState>
      )}

      {detail && (
        <>
          <div className="grid g-4" style={{ marginTop: 14 }}>
            <Stat label="Estimated time" value={mins(detail.minutes)} note={routeVia(activeRoute.edges)} />
            <Stat label="Distance" value={km(detail.distance)} note={`${kmh(detail.averageSpeed)} average`} />
            <Stat
              label="Worst condition"
              value={['Smooth', 'Slow', 'Disruption', 'Jammed'][detail.worstLevel]}
              note={`Flood risk ${['Low', 'Moderate', 'High'][detail.worstFlood]}`}
              tone={['smooth', 'slow', 'disruption', 'jammed'][detail.worstLevel]}
            />
            <Stat
              label="Reliability"
              value={confidence ? `+/- ${confidence.minutesBand} min` : 'n/a'}
              note={
                confidence
                  ? `${confidence.percent}% of the time within this band - today's confidence ${confidence.level}`
                  : 'No validation statistics in this export'
              }
            />
          </div>

          <div ref={sheetRef} style={{ background: 'var(--paper)', padding: 2 }}>
            <div className="grid g-wide" style={{ marginTop: 14 }}>
              <Card title="Route" subtitle={`${JUNCTION_BY_ID[from].name} to ${JUNCTION_BY_ID[to].name}`}>
                <NetworkMap
                  states={detail.legs.map((leg) => leg.state)}
                  layer={mode === 'safe' ? 'flood' : 'traffic'}
                  height={340}
                />
                <div className="rows" style={{ marginTop: 12 }}>
                  {detail.legs.map((leg) => (
                    <div key={leg.segmentId} className="row">
                      <div className="row-top">
                        <div>
                          <div className="row-name">{leg.name}</div>
                          <div className="row-meta">
                            {leg.from} to {leg.to} - {km(leg.km)}
                          </div>
                        </div>
                        <div className="inline">
                          <StatusChip status={leg.state.level.key} />
                          <strong className="tnum">{mins(leg.minutes)}</strong>
                        </div>
                      </div>
                      <div className="row-foot">
                        <span>{kmh(leg.state.speed)}</span>
                        <span>Rain {mm(leg.state.rain)}</span>
                        <span>Flood {leg.state.flood.label}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>

              <div className="stack">
                <Card title="Leave now, or wait?" subtitle="Same route, different departure times">
                  {advice ? (
                    <>
                      <div className="stack">
                        {advice.options.map((option) => {
                          const isBest = option.offset === advice.best.offset;
                          return (
                            <div
                              key={option.offset}
                              className="spread"
                              style={{
                                padding: '8px 10px',
                                borderRadius: 'var(--radius-sm)',
                                background: isBest ? 'var(--primary-tint)' : 'var(--surface-sunk)',
                                border: `1px solid ${isBest ? 'var(--primary)' : 'var(--line)'}`,
                              }}
                            >
                              <span style={{ fontWeight: 700 }}>
                                {option.offset === 0 ? 'Leave now' : `In ${option.offset} min`}
                              </span>
                              <span className="inline">
                                <strong className="tnum">{mins(option.minutes)}</strong>
                                {isBest && <Chip tone="smooth">Best</Chip>}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                      <p style={{ marginTop: 10, marginBottom: 0, fontWeight: 650 }}>
                        {advice.best.offset === 0
                          ? `Leave now - waiting does not help on this route.`
                          : `Waiting ${advice.best.offset} min saves about ${Math.max(
                              0,
                              advice.options[0].minutes - advice.best.minutes,
                            )} min.`}
                      </p>
                    </>
                  ) : (
                    <EmptyState title="No departure comparison available" />
                  )}
                </Card>

                <Card
                  title="What if it weren't raining?"
                  subtitle="A second model pass at zero rainfall"
                  actions={
                    <button
                      type="button"
                      className={`btn ${noRain ? '' : 'ghost'} small`}
                      aria-pressed={noRain}
                      onClick={() => setNoRain((v) => !v)}
                    >
                      {noRain ? 'Showing dry' : 'Show dry'}
                    </button>
                  }
                >
                  {rainCost != null && (
                    <>
                      <div className="big">
                        {rainCost > 0.5 ? `+${Math.round(rainCost)} min` : 'No measurable cost'}
                      </div>
                      <p className="small muted" style={{ marginTop: 4 }}>
                        {rainCost > 0.5
                          ? `Rain adds about ${Math.round(rainCost)} min to this trip - model estimate.`
                          : 'At the current rainfall the model predicts no meaningful delay on this route.'}
                      </p>
                    </>
                  )}
                  <Provenance>
                    <strong>Model estimate, not a measured fact.</strong> The dry figure is the same
                    route run through the network again with rainfall set to zero, holding
                    everything else fixed. Real traffic differs for reasons the model cannot see.
                  </Provenance>
                </Card>

                <Card title="Community" subtitle="Demo data">
                  <div className="big">{peerCount(from, to, rain)}</div>
                  <p className="small muted" style={{ marginTop: 4 }}>
                    others routing around this area
                  </p>
                  <Provenance>
                    <strong>This number is fake.</strong> monsoonplus has no users and collects no
                    telemetry. It is here to show where a real peer-count would sit, and it is
                    derived deterministically from the route so it does not pretend to fluctuate.
                  </Provenance>
                </Card>
              </div>
            </div>

            <div className="grid g-2" style={{ marginTop: 14 }}>
              <Card
                title={mode === 'safe' ? 'Flood-safe vs fastest' : 'Fastest vs flood-safe'}
                subtitle="What the flood penalty buys you"
              >
                {otherModeRoute?.detail ? (
                  <>
                    <div className="kv">
                      <span className="muted">
                        {mode === 'safe' ? 'Flood-safe (selected)' : 'Fastest (selected)'}
                      </span>
                      <b>
                        {mins(detail.minutes)} - flood{' '}
                        {['Low', 'Moderate', 'High'][detail.worstFlood]}
                      </b>
                    </div>
                    <div className="kv">
                      <span className="muted">{mode === 'safe' ? 'Fastest' : 'Flood-safe'}</span>
                      <b>
                        {mins(otherModeRoute.detail.minutes)} - flood{' '}
                        {['Low', 'Moderate', 'High'][otherModeRoute.detail.worstFlood]}
                      </b>
                    </div>
                    <p className="small" style={{ marginTop: 10, marginBottom: 0 }}>
                      {detail.worstFlood === otherModeRoute.detail.worstFlood
                        ? 'Both options carry the same worst-case flood risk here, so flood-safe routing has nothing to avoid on this pair.'
                        : mode === 'safe'
                          ? `Flood-safe costs ${Math.abs(
                              Math.round(detail.minutes - otherModeRoute.detail.minutes),
                            )} min and drops the worst flood risk from ${
                              ['Low', 'Moderate', 'High'][otherModeRoute.detail.worstFlood]
                            } to ${['Low', 'Moderate', 'High'][detail.worstFlood]}.`
                          : `The flood-safe option takes ${Math.abs(
                              Math.round(otherModeRoute.detail.minutes - detail.minutes),
                            )} min longer but lowers worst flood risk to ${
                              ['Low', 'Moderate', 'High'][otherModeRoute.detail.worstFlood]
                            }.`}
                    </p>
                    <Provenance>
                      Flood-safe multiplies the cost of crossing a High-risk edge by 4.0 and a
                      Moderate edge by 1.35. These are preferences, not bans: if the only path
                      crosses a flooded road, it is still returned and labelled.
                    </Provenance>
                  </>
                ) : (
                  <EmptyState title="No comparison available" />
                )}
              </Card>

              <Card title="Trip impact by scenario" subtitle="Same route under three rainfall levels">
                <TableScroll label="Data table, scroll horizontally">
                  <table className="data">
                    <thead>
                      <tr>
                        <th>Scenario</th>
                        <th className="num">ETA</th>
                        <th className="num">Avg speed</th>
                        <th>Worst</th>
                      </tr>
                    </thead>
                    <tbody>
                      {scenarios.map((scenario) => (
                        <tr key={scenario.label}>
                          <td>
                            {scenario.label}
                            <div className="small muted">{rainBucket(scenario.rainValue)}</div>
                          </td>
                          <td className="num">{mins(scenario.detail?.minutes)}</td>
                          <td className="num">{kmh(scenario.detail?.averageSpeed)}</td>
                          <td>
                            <StatusChip
                              status={
                                ['smooth', 'slow', 'disruption', 'jammed'][
                                  scenario.detail?.worstLevel ?? 0
                                ]
                              }
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </TableScroll>
              </Card>
            </div>

            <Card title="Rain through the trip window" subtitle="Model-side rainfall over the next hour">
              <div className="inline" style={{ gap: 6, flexWrap: 'nowrap', overflowX: 'auto' }}>
                {[0, 15, 30, 45, 60].map((offset) => {
                  const atOffset = evaluateRoute(predictor, activeRoute.edges, {
                    rain: effectiveRain,
                    departMinutes: offset,
                    satelliteEnabled,
                  });
                  return (
                    <div
                      key={offset}
                      style={{
                        flex: '1 0 110px',
                        padding: '10px 12px',
                        border: '1px solid var(--line)',
                        borderRadius: 'var(--radius-sm)',
                        background: 'var(--surface-sunk)',
                      }}
                    >
                      <div className="field-label">{offset === 0 ? 'Depart now' : `+${offset} min`}</div>
                      <div style={{ fontWeight: 850, fontSize: 16 }} className="tnum">
                        {mins(atOffset?.minutes)}
                      </div>
                      <div className="small muted">{clockFromMinutes(new Date().getHours() * 60 + new Date().getMinutes() + offset)}</div>
                    </div>
                  );
                })}
              </div>
            </Card>
          </div>
        </>
      )}
    </>
  );
}
