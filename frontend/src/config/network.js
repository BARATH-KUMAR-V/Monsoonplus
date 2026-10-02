/**
 * The road network, imported directly from the repo's single source of truth
 * (`config/segments.json` via the `@config` alias in vite.config.js).
 *
 * There is no second copy of the segment list in the frontend. Adding a road means
 * editing that one JSON file; the model, collector, backend and this UI all pick it up.
 */
import segments from '@config/segments.json';

export const NETWORK = segments;
export const CITY = segments.city;
export const MODEL_CFG = segments.model;
export const THRESHOLDS = segments.thresholds;
export const ATTRIBUTIONS = segments.attributions;
export const LANDMARKS = segments.landmarks;
export const MONSOON_CALENDAR = segments.monsoon_calendar;

export const JUNCTIONS = segments.junctions;
export const SEGMENTS = segments.segments;

export const JUNCTION_BY_ID = Object.fromEntries(JUNCTIONS.map((j) => [j.id, j]));
export const SEGMENT_BY_ID = Object.fromEntries(SEGMENTS.map((s) => [s.id, s]));

export const HORIZONS = MODEL_CFG.horizons_minutes;

/** Great-circle distance in km between two [lat, lon] pairs. */
export function haversineKm(a, b) {
  const r = Math.PI / 180;
  const dLat = (b[0] - a[0]) * r;
  const dLon = (b[1] - a[1]) * r;
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(a[0] * r) * Math.cos(b[0] * r) * Math.sin(dLon / 2) ** 2;
  return 2 * 6371 * Math.asin(Math.sqrt(h));
}

export const latlon = (junctionId) => {
  const j = JUNCTION_BY_ID[junctionId];
  return [j.lat, j.lon];
};

/** Driven length per segment -- same formula and constant as config/segments.py. */
export const SEGMENT_LENGTH_KM = Object.fromEntries(
  SEGMENTS.map((s) => [
    s.id,
    haversineKm(latlon(s.a), latlon(s.b)) * MODEL_CFG.road_length_detour_factor,
  ]),
);

export const SEGMENT_MIDPOINT = Object.fromEntries(
  SEGMENTS.map((s) => {
    const a = JUNCTION_BY_ID[s.a];
    const b = JUNCTION_BY_ID[s.b];
    return [s.id, [(a.lat + b.lat) / 2, (a.lon + b.lon) / 2]];
  }),
);

/** junction id -> [{ to, segmentId }] for the routing graph. */
export const ADJACENCY = (() => {
  const adjacency = Object.fromEntries(JUNCTIONS.map((j) => [j.id, []]));
  SEGMENTS.forEach((s) => {
    adjacency[s.a].push({ to: s.b, segmentId: s.id });
    adjacency[s.b].push({ to: s.a, segmentId: s.id });
  });
  return adjacency;
})();

export const segmentEndpoints = (segment) =>
  `${JUNCTION_BY_ID[segment.a].name} - ${JUNCTION_BY_ID[segment.b].name}`;

/** Which Chennai monsoon regime a month falls in (1-12). */
export function monsoonRegime(month) {
  const found = MONSOON_CALENDAR.regimes.find((r) => r.months.includes(month));
  return found || { label: 'Unknown', note: '' };
}

export const RAIN_WEIGHTS = THRESHOLDS.rain_weights;
export const IMPACT_WEIGHTS = THRESHOLDS.impact_score_weights;
export const ROUTING_PENALTIES = THRESHOLDS.routing_penalties;
