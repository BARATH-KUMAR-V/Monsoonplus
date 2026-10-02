/**
 * Page 5 - Event Replay.
 *
 * Two modes, and the distinction is the whole point of the page:
 *
 *  Simulated  a scripted 4-7 PM storm pushed through the trained model. 13 frames,
 *             15 minutes apart. Labelled "Simulated" on every frame.
 *
 *  Real logs  predicted-vs-actual from files the collector wrote during real Chennai
 *             weather. When no logs exist the page says so plainly and offers the
 *             command to start collecting -- it never fabricates a replay.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useStore } from '@/data/store';
import NetworkMap from '@/components/NetworkMap';
import {
  Banner,
  Card,
  Chip,
  EmptyState,
  PageHead,
  Provenance,
  Segmented,
  Stat,
  StatusChip,
} from '@/components/ui';
import { SEGMENT_BY_ID } from '@/config/network';
import { congestionLevel, floodRisk, impactScore, rainBucket } from '@/lib/model';
import { kmh, mm } from '@/lib/format';

const AXIS_STYLE = { fontSize: 11, fill: 'var(--muted)' };

/** Rebuild full segment states from a stored replay frame. */
function framesToStates(frame, satelliteEnabled) {
  if (!frame) return [];
  return frame.per_segment.map((entry) => {
    const segment = SEGMENT_BY_ID[entry.segment_id];
    const level = congestionLevel(entry.speed_kmh, segment.free_flow_kmh);
    const floodSignal = satelliteEnabled ? (entry.wetness ?? 0) * 100 : 0;
    const flood = floodRisk(floodSignal);
    const impact = impactScore({
      speed: entry.speed_kmh,
      freeFlow: segment.free_flow_kmh,
      rain: entry.rain_mm_h,
      floodSignal,
      speedIn60: entry.speed_kmh,
    });
    return {
      segmentId: entry.segment_id,
      segment,
      speed: entry.speed_kmh,
      speedNow: entry.speed_kmh,
      speedIn60: entry.speed_kmh,
      predicted: { 't+15': entry.speed_kmh, 't+30': entry.speed_kmh, 't+60': entry.speed_kmh },
      rain: entry.rain_mm_h,
      ndwi: entry.ndwi ?? 0,
      wetness: entry.wetness ?? 0,
      gate: { traffic: 0, weather: 0, satellite: 0 },
      level,
      flood,
      floodSignal,
      impact: impact.score,
      impactParts: impact.parts,
      band: [entry.speed_kmh, entry.speed_kmh],
      halfWidth: 0,
      confidence: 'n/a',
      satelliteEnabled,
    };
  });
}

/** Name the milestones in the storm from the frames themselves. */
function milestones(frames) {
  if (!frames?.length) return [];
  const events = [];
  const baselineDisrupted = frames[0].per_segment.filter((p) => {
    const segment = SEGMENT_BY_ID[p.segment_id];
    return congestionLevel(p.speed_kmh, segment.free_flow_kmh).index >= 1;
  }).length;

  const disruptedAt = (frame) =>
    frame.per_segment.filter((p) => {
      const segment = SEGMENT_BY_ID[p.segment_id];
      return congestionLevel(p.speed_kmh, segment.free_flow_kmh).index >= 1;
    }).length;
  const severeAt = (frame) =>
    frame.per_segment.filter((p) => {
      const segment = SEGMENT_BY_ID[p.segment_id];
      return congestionLevel(p.speed_kmh, segment.free_flow_kmh).index === 3;
    }).length;

  const rainStart = frames.findIndex((f) => f.city_rain_mm_h >= 3);
  if (rainStart >= 0) {
    events.push({
      frame: rainStart,
      title: 'Rain begins',
      detail: `${frames[rainStart].city_rain_mm_h} mm/h at ${frames[rainStart].clock}`,
    });
  }

  const slowing = frames.findIndex((f) => disruptedAt(f) >= baselineDisrupted + 2);
  if (slowing >= 0) {
    events.push({
      frame: slowing,
      title: 'Traffic begins slowing',
      detail: `${disruptedAt(frames[slowing])} roads at Slow or worse`,
    });
  }

  const flooding = frames.findIndex((f) => f.per_segment.some((p) => (p.wetness ?? 0) * 100 >= 18));
  if (flooding >= 0) {
    events.push({
      frame: flooding,
      title: 'Flood signal rising',
      detail: `Wetness crosses the moderate band at ${frames[flooding].clock}`,
    });
  }

  const severe = frames.findIndex((f) => severeAt(f) >= 1);
  if (severe >= 0) {
    events.push({
      frame: severe,
      title: 'Severe slowdown',
      detail: `${severeAt(frames[severe])} road(s) jammed`,
    });
  }

  let peak = 0;
  frames.forEach((frame, index) => {
    if (frame.mean_speed_kmh < frames[peak].mean_speed_kmh) peak = index;
  });
  events.push({
    frame: peak,
    title: 'Peak disruption',
    detail: `Network average ${kmh(frames[peak].mean_speed_kmh)} at ${frames[peak].clock}`,
  });

  const recovery = frames.findIndex(
    (f, index) => index > peak && f.mean_speed_kmh > frames[index - 1].mean_speed_kmh,
  );
  if (recovery >= 0) {
    events.push({
      frame: recovery,
      title: 'Recovery begins',
      detail: `Average back to ${kmh(frames[recovery].mean_speed_kmh)}`,
    });
  }

  return events.sort((a, b) => a.frame - b.frame);
}

function SimulatedReplay() {
  const { chennai, satelliteEnabled } = useStore();
  const replay = chennai?.storm_replay;
  const frames = replay?.frames || [];
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timer = useRef(null);

  useEffect(() => {
    if (!playing || frames.length === 0) {
      if (timer.current) clearInterval(timer.current);
      return undefined;
    }
    timer.current = setInterval(() => {
      setIndex((current) => {
        if (current >= frames.length - 1) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 900);
    return () => clearInterval(timer.current);
  }, [playing, frames.length]);

  const events = useMemo(() => milestones(frames), [frames]);
  const frame = frames[index];
  const states = useMemo(() => framesToStates(frame, satelliteEnabled), [frame, satelliteEnabled]);

  const chartData = useMemo(
    () =>
      frames.map((f, i) => ({
        clock: f.clock,
        rain: f.city_rain_mm_h,
        speed: f.mean_speed_kmh,
        current: i === index ? f.mean_speed_kmh : null,
      })),
    [frames, index],
  );

  if (!frames.length) {
    return (
      <EmptyState title="No storm replay in this export">
        Re-run <code>python -m ml.export.export_predictions</code> to generate it.
      </EmptyState>
    );
  }

  const activeEvent = events.find((event) => event.frame === index);

  return (
    <>
      <Banner tier="watch" tag="SIMULATED">
        Scripted storm, {replay.window}. A rainfall profile pushed through the trained model -
        not a recording of a real event.
      </Banner>

      <div className="grid g-4">
        <Stat label="Clock" value={frame.clock} note={`Frame ${index + 1} of ${frames.length}`} />
        <Stat label="City rainfall" value={mm(frame.city_rain_mm_h)} note={rainBucket(frame.city_rain_mm_h)} />
        <Stat label="Network average" value={kmh(frame.mean_speed_kmh)} />
        <Stat
          label="Roads disrupted"
          value={`${states.filter((s) => s.level.index >= 1).length} of ${states.length}`}
          tone={states.some((s) => s.level.index === 3) ? 'jammed' : 'slow'}
        />
      </div>

      <div className="grid g-wide" style={{ marginTop: 14 }}>
        <Card style={{ padding: 0 }}>
          <NetworkMap
            states={states}
            layer="traffic"
            height={400}
            showWetness={satelliteEnabled}
            topLeft={<span>{frame.clock} - {mm(frame.city_rain_mm_h)}</span>}
          />
        </Card>

        <div className="stack">
          <Card title="Playback">
            <div className="inline" style={{ marginBottom: 10 }}>
              <button type="button" className="btn small" onClick={() => setPlaying((p) => !p)}>
                {playing ? 'Pause' : 'Play'}
              </button>
              <button
                type="button"
                className="btn ghost small"
                onClick={() => {
                  setPlaying(false);
                  setIndex(0);
                }}
              >
                Restart
              </button>
            </div>
            <input
              type="range"
              min="0"
              max={frames.length - 1}
              value={index}
              onChange={(event) => {
                setPlaying(false);
                setIndex(Number(event.target.value));
              }}
              aria-label="Storm replay frame"
              aria-valuetext={`${frame.clock}, ${frame.city_rain_mm_h} millimetres per hour`}
            />
            <div className="ticks">
              <span>{frames[0].clock}</span>
              <span>{frames[Math.floor(frames.length / 2)].clock}</span>
              <span>{frames[frames.length - 1].clock}</span>
            </div>
            {activeEvent && (
              <div style={{ marginTop: 12 }}>
                <Chip tone="slow">{activeEvent.title}</Chip>
                <p className="small muted" style={{ marginTop: 6, marginBottom: 0 }}>
                  {activeEvent.detail}
                </p>
              </div>
            )}
          </Card>

          <Card title="Milestones" subtitle="Named from the frames, not hand-written">
            <div className="rows">
              {events.map((event) => (
                <button
                  key={`${event.frame}-${event.title}`}
                  type="button"
                  className={`row ${event.frame === index ? 'selected' : ''}`}
                  aria-pressed={event.frame === index}
                  onClick={() => {
                    setPlaying(false);
                    setIndex(event.frame);
                  }}
                >
                  <div className="row-top">
                    <div>
                      <div className="row-name">{event.title}</div>
                      <div className="row-meta">{event.detail}</div>
                    </div>
                    <span className="small muted">{frames[event.frame].clock}</span>
                  </div>
                </button>
              ))}
            </div>
          </Card>
        </div>
      </div>

      <Card title="Storm profile" subtitle="Rainfall and network speed across the replay" style={{ marginTop: 14 }}>
        <div style={{ height: 240 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
              <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
              <XAxis dataKey="clock" tick={AXIS_STYLE} stroke="var(--line)" />
              <YAxis yAxisId="left" tick={AXIS_STYLE} unit=" km/h" stroke="var(--line)" />
              <YAxis yAxisId="right" orientation="right" tick={AXIS_STYLE} unit=" mm" stroke="var(--line)" />
              <RTooltip contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid var(--line)' }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line
                yAxisId="left"
                dataKey="speed"
                name="network speed"
                stroke="var(--primary)"
                strokeWidth={2.5}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                yAxisId="right"
                dataKey="rain"
                name="rainfall"
                stroke="var(--rain)"
                strokeWidth={2}
                strokeDasharray="5 4"
                dot={false}
                isAnimationActive={false}
              />
              <Line
                yAxisId="left"
                dataKey="current"
                name="current frame"
                stroke="var(--jammed)"
                strokeWidth={0}
                dot={{ r: 5, fill: 'var(--jammed)' }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>
    </>
  );
}

function LogReplay() {
  const { logIndex, logData, loadLog } = useStore();

  if (logIndex.state === 'loading') {
    return <EmptyState title="Looking for collected logs" />;
  }

  if (logIndex.state === 'empty' || logIndex.dates.length === 0) {
    return (
      <>
        <Banner tier="info" tag="NO LOGS">
          No collected logs yet.
        </Banner>
        <EmptyState title="No collected logs yet - collect live data to populate this view">
          This mode replays what the model <em>predicted</em> against what actually happened, from
          files the collector wrote during real Chennai weather. Nothing is shown here until real
          data exists, because a fabricated predicted-vs-actual chart would be worthless.
          <br />
          <br />
          To start collecting, from the repository root:
          <br />
          <code>python -m collector.collect --once</code>
          <br />
          <br />
          See README section 9 for how many rainy and clear days are needed before this view and
          the pattern cards below become meaningful.
        </EmptyState>
      </>
    );
  }

  const entries = logIndex.dates;
  const payload = logData.payload;

  return (
    <>
      <Card title="Pick a collected day" subtitle={`${entries.length} day(s) available`}>
        <div className="inline">
          {entries.map((entry) => (
            <button
              key={entry.file}
              type="button"
              className={`btn ${logData.date === entry.date ? '' : 'ghost'} small`}
              aria-pressed={logData.date === entry.date}
              onClick={() => loadLog(entry)}
            >
              {entry.date} {entry.label ? `(${entry.label})` : ''}
            </button>
          ))}
        </div>
      </Card>

      {logData.state === 'loading' && <EmptyState title="Loading log" />}
      {logData.state === 'error' && (
        <EmptyState title="Could not read that log">{logData.error}</EmptyState>
      )}

      {logData.state === 'ready' && payload && (
        <>
          <Banner tier="info" tag="REAL LOG">
            {payload.date} - {payload.label || 'collected'} - {payload.samples?.length ?? 0} samples
          </Banner>
          <Card title="Predicted vs actual" subtitle="From collected log files">
            <div style={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={payload.samples || []} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
                  <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
                  <XAxis dataKey="time" tick={AXIS_STYLE} stroke="var(--line)" />
                  <YAxis tick={AXIS_STYLE} unit=" km/h" stroke="var(--line)" />
                  <RTooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line dataKey="predicted" name="predicted" stroke="var(--primary)" dot={false} />
                  <Line dataKey="actual" name="actual" stroke="var(--jammed)" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            {payload.mae != null && (
              <p style={{ marginTop: 10, fontWeight: 650 }}>
                MAE on this day: {payload.mae.toFixed(2)} km/h
                {payload.mae_rain != null &&
                  ` (rain ${payload.mae_rain.toFixed(2)}, clear ${payload.mae_clear?.toFixed(2) ?? '--'})`}
              </p>
            )}
          </Card>
        </>
      )}
    </>
  );
}

function PatternCards() {
  const { logIndex } = useStore();
  const hasLogs = logIndex.state === 'ready' && logIndex.dates.length > 0;

  return (
    <Card title="Pattern cards" subtitle="Statements extracted from collected logs">
      {!hasLogs ? (
        <EmptyState title="Not enough data yet">
          Pattern cards such as "Heavy rain: Velachery to OMR junction floods in 47 +/- 8 min"
          need several collected rainy days before they mean anything. With no logs collected,
          no pattern is claimed. This space stays empty rather than being filled with invented
          numbers.
        </EmptyState>
      ) : (
        <p className="small muted">
          {logIndex.dates.length} day(s) collected. Patterns are computed once at least three
          rainy days and one clear day exist; see README section 9.
        </p>
      )}
    </Card>
  );
}

export default function EventReplay() {
  const [tab, setTab] = useState('simulated');

  return (
    <>
      <PageHead
        title="Event Replay"
        caption="Watch a storm move through the network, or compare what the model predicted against what actually happened."
        actions={
          <Segmented
            label="Replay source"
            value={tab}
            onChange={setTab}
            options={[
              { value: 'simulated', label: 'Simulated storm' },
              { value: 'logs', label: 'Real logs' },
            ]}
          />
        }
      />

      {tab === 'simulated' ? <SimulatedReplay /> : <LogReplay />}

      <div style={{ marginTop: 14 }}>
        <PatternCards />
      </div>

      <Provenance>
        The simulated replay is the trained model's response to a scripted rainfall curve. The
        real-log replay is the only place on this site where measured Chennai traffic would
        appear, and it stays empty until you collect some.
      </Provenance>
    </>
  );
}
