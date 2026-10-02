/**
 * Flood-risk ESTIMATE.
 *
 * This is deliberately a transparent rule, not a model. Every input is something the
 * app actually has, every contribution is reported separately, and the result carries
 * the ESTIMATE label wherever it is shown.
 *
 *   rainfall intensity   how hard it is raining / forecast to rain on this stretch
 *   recent accumulation  several wet hours saturate the ground and matter more than one
 *   terrain              a dip in the road relative to its surroundings, plus absolute
 *                        height above sea level
 *   history              neighbourhoods repeatedly reported as waterlogging-prone
 *   community reports    someone standing there says it is flooded right now
 *
 * The one place a LEARNED prediction is used instead is inside the pilot road network,
 * where MonsoonPlusNet was actually trained. `source` on the result says which applies,
 * and the two are never silently mixed.
 */
import { SOURCE } from '@/services/provenance';
import { WATERLOGGING_PRIORS } from '@/config/waterlogging';
import { lowLyingPoints, lownessScore } from '@/services/elevation';

const clamp01 = (v) => Math.min(1, Math.max(0, v));

/** Great-circle km. */
function distanceKm(aLat, aLon, bLat, bLon) {
  const r = Math.PI / 180;
  const dLat = (bLat - aLat) * r;
  const dLon = (bLon - aLon) * r;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(aLat * r) * Math.cos(bLat * r) * Math.sin(dLon / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(h));
}

/** 0-1 historical prior for a point, plus which area drove it. */
export function historyAt(lat, lon) {
  let best = { weight: 0, name: null };
  WATERLOGGING_PRIORS.forEach((area) => {
    const d = distanceKm(lat, lon, area.lat, area.lon);
    if (d > area.km) return;
    // Taper to zero at the edge rather than drawing a hard circle on the city.
    const falloff = 1 - (d / area.km) * 0.55;
    const weight = area.weight * falloff;
    if (weight > best.weight) best = { weight, name: area.name };
  });
  return best;
}

/** Rain intensity -> 0-1. 25 mm/h is the "very heavy" threshold used across the app. */
const intensityScore = (mmPerHour) => clamp01((mmPerHour ?? 0) / 25);

/** Several wet hours matter more than one heavy burst. 60 mm over the window saturates. */
const accumulationScore = (mmTotal) => clamp01((mmTotal ?? 0) / 60);

export const FLOOD_BANDS = [
  { max: 0.18, key: 'low', label: 'Low', tone: 'smooth', advice: 'No pooling expected.' },
  { max: 0.4, key: 'watch', label: 'Watch', tone: 'slow', advice: 'Shallow water possible at the usual spots.' },
  { max: 0.65, key: 'moderate', label: 'Moderate', tone: 'disruption', advice: 'Water likely to collect. Slow down and avoid the kerb lane.' },
  { max: 1.01, key: 'high', label: 'High', tone: 'jammed', advice: 'Waterlogging likely. Avoid this stretch if you can.' },
];

export const floodBand = (score) => FLOOD_BANDS.find((b) => score < b.max) || FLOOD_BANDS[FLOOD_BANDS.length - 1];

/**
 * Estimate flood risk for ONE point.
 *
 * @param point        { lat, lon }
 * @param rain         { nowMm, next3hMm, recent6hMm }
 * @param terrain      { drop, elevation, loweness } | null
 * @param reports      active community reports near this point
 * @returns { score, band, source, parts, reasons }
 */
export function estimatePointFlood(point, rain, terrain, reports = []) {
  const parts = {
    intensity: intensityScore(Math.max(rain?.nowMm ?? 0, rain?.next3hMm ?? 0)),
    accumulation: accumulationScore(rain?.recent6hMm ?? 0),
    terrain: terrain ? clamp01((terrain.drop ?? 0) / 6) * 0.6 + (terrain.lowness ?? 0) * 0.4 : 0,
    history: historyAt(point.lat, point.lon).weight,
    reported: 0,
  };

  const nearby = reports.filter(
    (r) => ['flood', 'waterlogging', 'blocked'].includes(r.type) && distanceKm(point.lat, point.lon, r.lat, r.lon) < 0.7,
  );
  if (nearby.length) parts.reported = nearby.some((r) => r.severity === 'severe') ? 1 : 0.65;

  // Rain is the trigger; terrain and history decide WHERE it collects. So terrain and
  // history are gated by rain rather than added to it -- a dip in dry weather is just
  // a dip, and saying "high flood risk" there in January would destroy trust.
  const wetness = clamp01(parts.intensity * 0.65 + parts.accumulation * 0.55);
  const susceptibility = clamp01(parts.terrain * 0.55 + parts.history * 0.6);

  let score = clamp01(wetness * (0.45 + 0.75 * susceptibility));

  // A person standing in the water outranks any amount of inference.
  if (parts.reported) score = Math.max(score, 0.55 + 0.35 * parts.reported);

  const reasons = [];
  if (parts.reported) reasons.push(nearby.length > 1 ? `${nearby.length} people reported water here` : 'Someone reported water here');
  if (parts.intensity > 0.3) reasons.push('heavy rain on this stretch');
  else if (parts.intensity > 0.1) reasons.push('rain on this stretch');
  if (parts.accumulation > 0.35) reasons.push('the ground is already saturated');
  if (parts.terrain > 0.35) reasons.push('the road dips here');
  if (parts.history > 0.5) {
    const name = historyAt(point.lat, point.lon).name;
    reasons.push(name ? `${name} floods often` : 'this area floods often');
  }

  return {
    score,
    band: floodBand(score),
    source: parts.reported ? SOURCE.COMMUNITY : SOURCE.ESTIMATE,
    parts,
    reasons,
  };
}

/**
 * Estimate flood risk for a whole route from its sampled points.
 * @returns { score, band, source, worst, segments, reasons }
 */
export function estimateRouteFlood(samples, { elevations = null, reports = [] } = {}) {
  if (!samples?.length) return null;

  const dips = lowLyingPoints(elevations);
  const dipByIndex = Object.fromEntries(dips.map((d) => [d.index, d]));
  const lowness = lownessScore(elevations);

  const points = samples.map((sample, index) => {
    const terrain = elevations
      ? { drop: dipByIndex[index]?.drop ?? 0, elevation: elevations[index], lowness: lowness ?? 0 }
      : null;
    return { ...sample, flood: estimatePointFlood(sample, sample.rain, terrain, reports), terrain };
  });

  // A route is as floodable as its worst stretch: one impassable 200 m decides the trip.
  // But a single marginal point should not condemn an otherwise fine route, so the score
  // blends the worst point with the average.
  const scores = points.map((p) => p.flood.score);
  const worstScore = Math.max(...scores);
  const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  const score = worstScore * 0.7 + meanScore * 0.3;

  const worst = points[scores.indexOf(worstScore)];
  const risky = points.filter((p) => p.flood.score >= 0.4);
  const reported = points.some((p) => p.flood.source === SOURCE.COMMUNITY);

  const reasons = [...new Set(points.flatMap((p) => (p.flood.score >= 0.4 ? p.flood.reasons : [])))];

  return {
    score,
    band: floodBand(score),
    source: reported ? SOURCE.COMMUNITY : SOURCE.ESTIMATE,
    basis: elevations ? 'rainfall, terrain height, past flooding reports' : 'rainfall and past flooding reports (terrain data unavailable)',
    worst,
    riskyCount: risky.length,
    points,
    reasons,
    hasTerrain: Boolean(elevations),
  };
}
