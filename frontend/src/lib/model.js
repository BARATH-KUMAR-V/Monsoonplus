/**
 * Reads the trained model's exported forecast grid and answers the questions the UI
 * asks of it.
 *
 * What is and is not a model output
 * ---------------------------------
 * `forecast_grid` in predictions_synthetic_chennai.json holds, for each rain level in
 * a sweep and each segment: the speed now, the model's t+15/30/60 predictions, the
 * NDWI/wetness it saw, and the gate weights it produced. Those numbers came out of
 * MonsoonPlusNet during export.
 *
 * This module INTERPOLATES between those rows when the user drags the rain slider to a
 * value between two sampled levels, and between horizons for the +45 min step. That
 * interpolation is arithmetic on model outputs, not a second model, and anything
 * derived here is labelled "model estimate" in the UI.
 *
 * Congestion level, flood risk and Impact Score are deterministic functions of those
 * outputs using the thresholds in config/segments.json -- rules, not learned.
 */
import { SEGMENTS, SEGMENT_BY_ID, THRESHOLDS, HORIZONS, IMPACT_WEIGHTS } from '@/config/network';

const LEVELS = THRESHOLDS.congestion_levels;
const FLOOD = THRESHOLDS.flood_risk;
const BUCKETS = THRESHOLDS.rain_buckets;
const REGIMES = THRESHOLDS.rain_regimes;

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const lerp = (a, b, t) => a + (b - a) * t;

/** Dry / Light / Moderate / Heavy / Extreme. */
export function rainBucket(mmPerHour) {
  if (mmPerHour < BUCKETS.dry_max) return 'Dry';
  if (mmPerHour < BUCKETS.light_max) return 'Light';
  if (mmPerHour < BUCKETS.moderate_max) return 'Moderate';
  if (mmPerHour < BUCKETS.heavy_max) return 'Heavy';
  return 'Extreme';
}

/** The 3-bucket regime tag used on the Forecast page. */
export function rainRegime(mmPerHour) {
  if (mmPerHour < REGIMES.dry_max) return 'Dry';
  if (mmPerHour < REGIMES.steady_max) return 'Steady rain';
  return 'Cloudburst';
}

/** Ground-truth weather class, matching config.segments.rain_label in Python. */
export function rainLabel(mmPerHour) {
  const cfg = THRESHOLDS.rain_label_mm_h;
  if (mmPerHour >= cfg.heavy_rain_min) return 'heavy_rain';
  if (mmPerHour >= cfg.rain_min) return 'rain';
  return 'clear';
}

/**
 * Congestion level from the ratio of current speed to free-flow speed.
 * Returns { key, label, glyph, index } -- the key drives colour, the glyph and label
 * are what make the status readable without colour.
 */
export function congestionLevel(speed, freeFlow) {
  const ratio = freeFlow > 0 ? speed / freeFlow : 0;
  const index = LEVELS.findIndex((l) => ratio >= l.min_speed_ratio);
  const resolved = index === -1 ? LEVELS.length - 1 : index;
  const level = LEVELS[resolved];
  return {
    index: resolved,
    label: level.label,
    glyph: level.glyph,
    key: ['smooth', 'slow', 'disruption', 'jammed'][resolved],
    ratio,
  };
}

/** Flood risk band from the model's wetness/flood signal (0-100 scale). */
export function floodRisk(floodSignal) {
  if (floodSignal >= FLOOD.high_min) return { index: 2, label: 'High', key: 'jammed' };
  if (floodSignal >= FLOOD.moderate_min) return { index: 1, label: 'Moderate', key: 'slow' };
  return { index: 0, label: 'Low', key: 'smooth' };
}

/**
 * Monsoon Impact Score, 0-100.
 *   0.40 x traffic + 0.25 x rain + 0.20 x flood + 0.15 x deterioration
 * Weights come from config/segments.json, so the UI and any Python consumer agree.
 */
export function impactScore({ speed, freeFlow, rain, floodSignal, speedIn60 }) {
  const traffic = clamp((freeFlow - speed) / freeFlow / 0.8, 0, 1);
  const rainPart = clamp(rain / 100, 0, 1);
  const floodPart = clamp(floodSignal / 60, 0, 1);
  const deterioration = clamp((speed - speedIn60) / freeFlow / 0.3, 0, 1);
  const score =
    100 *
    (IMPACT_WEIGHTS.traffic * traffic +
      IMPACT_WEIGHTS.rain * rainPart +
      IMPACT_WEIGHTS.flood * floodPart +
      IMPACT_WEIGHTS.deterioration * deterioration);
  return {
    score: Math.round(clamp(score, 0, 100)),
    parts: { traffic, rain: rainPart, flood: floodPart, deterioration },
  };
}

export function impactBand(score) {
  if (score < 20) return 'smooth';
  if (score < 40) return 'slow';
  if (score < 60) return 'disruption';
  return 'jammed';
}

/* ------------------------------------------------------------------ grid -- */

/**
 * Wraps the exported forecast grid with lookup and interpolation.
 * `grid` is predictions.forecast_grid; returns null-safe helpers.
 */
export function createPredictor(grid) {
  if (!grid || !Array.isArray(grid.rows) || grid.rows.length === 0) return null;

  const rows = [...grid.rows].sort((a, b) => a.rain_mm_h - b.rain_mm_h);
  const levels = rows.map((r) => r.rain_mm_h);
  const bySegment = rows.map((row) =>
    Object.fromEntries(row.per_segment.map((p) => [p.segment_id, p])),
  );

  /** Bracket a rain value between two sampled levels, with the blend factor. */
  function bracket(rain) {
    const value = clamp(rain, levels[0], levels[levels.length - 1]);
    let upper = levels.findIndex((l) => l >= value);
    if (upper === -1) upper = levels.length - 1;
    const lower = Math.max(0, upper - 1);
    const span = levels[upper] - levels[lower];
    const t = span === 0 ? 0 : (value - levels[lower]) / span;
    return { lower, upper, t };
  }

  /** The model's record for one segment at a rain level, blended. */
  function at(segmentId, rain) {
    const { lower, upper, t } = bracket(rain);
    const a = bySegment[lower][segmentId];
    const b = bySegment[upper][segmentId];
    if (!a || !b) return null;

    const predicted = {};
    HORIZONS.forEach((h) => {
      const key = `t+${h}`;
      predicted[key] = lerp(a.predicted_kmh[key] ?? 0, b.predicted_kmh[key] ?? 0, t);
    });

    return {
      segmentId,
      speedNow: lerp(a.speed_now_kmh, b.speed_now_kmh, t),
      predicted,
      rain: lerp(a.rain_mm_h, b.rain_mm_h, t),
      ndwi: lerp(a.ndwi, b.ndwi, t),
      wetness: lerp(a.wetness, b.wetness, t),
      gate: {
        traffic: lerp(a.gate.traffic, b.gate.traffic, t),
        weather: lerp(a.gate.weather, b.gate.weather, t),
        satellite: lerp(a.gate.satellite, b.gate.satellite, t),
      },
    };
  }

  /**
   * Speed at an arbitrary minute offset (0..60).
   * 0 is the model's "now"; 15/30/60 are direct model outputs; anything between is
   * linearly interpolated between the two nearest horizons.
   */
  function speedAt(segmentId, rain, minutes) {
    const record = at(segmentId, rain);
    if (!record) return null;
    const offset = clamp(minutes, 0, Math.max(...HORIZONS));
    if (offset <= 0) return record.speedNow;

    const points = [{ m: 0, v: record.speedNow }].concat(
      HORIZONS.map((h) => ({ m: h, v: record.predicted[`t+${h}`] })),
    );
    for (let i = 1; i < points.length; i += 1) {
      if (offset <= points[i].m) {
        const prev = points[i - 1];
        const next = points[i];
        const span = next.m - prev.m;
        return span === 0 ? next.v : lerp(prev.v, next.v, (offset - prev.m) / span);
      }
    }
    return points[points.length - 1].v;
  }

  /** Full derived state for one segment: speed, status, flood, impact, confidence. */
  function stateFor(segmentId, rain, minutes = 0, options = {}) {
    const segment = SEGMENT_BY_ID[segmentId];
    const record = at(segmentId, rain);
    if (!record || !segment) return null;

    const satelliteOn = options.satelliteEnabled !== false;
    const speed = speedAt(segmentId, rain, minutes);
    const speedIn60 = speedAt(segmentId, rain, 60);
    const level = congestionLevel(speed, segment.free_flow_kmh);

    // Flood signal: the model's wetness scaled to the 0-100 band the thresholds use.
    // With the satellite layer switched off the UI must not invent one, so it reports
    // zero and the Data & System page flips the model status to "Degraded".
    const floodSignal = satelliteOn ? record.wetness * 100 : 0;
    const flood = floodRisk(floodSignal);

    const impact = impactScore({
      speed,
      freeFlow: segment.free_flow_kmh,
      rain: record.rain,
      floodSignal,
      speedIn60,
    });

    // Confidence band widens with horizon and with rain, using the measured per-regime
    // error from the export when available.
    const band = options.confidenceBand ?? null;
    const label = rainLabel(record.rain);
    const halfWidth = band && band[label] != null
      ? band[label] * (0.6 + (minutes / 60) * 0.7)
      : 1.3 * (0.6 + minutes * 0.05 + (record.rain / 100) * 1.2);

    return {
      segment,
      segmentId,
      speed,
      speedNow: record.speedNow,
      speedIn60,
      predicted: record.predicted,
      rain: record.rain,
      ndwi: record.ndwi,
      wetness: record.wetness,
      gate: record.gate,
      level,
      flood,
      floodSignal,
      impact: impact.score,
      impactParts: impact.parts,
      band: [Math.max(1, speed - halfWidth), speed + halfWidth],
      halfWidth,
      // Confidence LEVEL is reported per weather regime, not from the band width.
      // The measured heavy-rain band happens to be narrow on this synthetic data, so
      // a width-derived label would read "high confidence" in a cloudburst -- which
      // would be technically derived but misleading, and would contradict the badge
      // Smart Trip shows for the same conditions. Both now read the same source.
      confidence: (options.confidenceLabel && options.confidenceLabel[label]) ||
        (label === 'clear' ? 'High' : label === 'rain' ? 'Medium' : 'Low'),
      satelliteEnabled: satelliteOn,
    };
  }

  /** Every segment at once. */
  function allStates(rain, minutes = 0, options = {}) {
    return SEGMENTS.map((s) => stateFor(s.id, rain, minutes, options)).filter(Boolean);
  }

  return { at, speedAt, stateFor, allStates, levels, rows };
}

/**
 * Gate weights averaged over segments at a given rain level -- the Model Lab's
 * modality breakdown. These are the trained network's own numbers.
 */
export function averageGate(states) {
  if (!states.length) return null;
  const total = states.reduce(
    (acc, s) => ({
      traffic: acc.traffic + s.gate.traffic,
      weather: acc.weather + s.gate.weather,
      satellite: acc.satellite + s.gate.satellite,
    }),
    { traffic: 0, weather: 0, satellite: 0 },
  );
  const n = states.length;
  return {
    traffic: total.traffic / n,
    weather: total.weather / n,
    satellite: total.satellite / n,
  };
}
