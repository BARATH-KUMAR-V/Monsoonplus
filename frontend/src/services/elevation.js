/**
 * Ground elevation, from Open-Meteo's free elevation endpoint (Copernicus DEM, 90 m).
 *
 * This is the input that lets MonsoonPlus say something useful about flooding OUTSIDE
 * its trained pilot area. Water collects in dips, so a stretch of road noticeably lower
 * than the ground around it is where it will pool. That is an ESTIMATE from terrain and
 * rainfall, never a learned prediction, and the UI labels it that way.
 */
import { getJson, qs } from './http';

const ELEVATION = 'https://api.open-meteo.com/v1/elevation';
const MAX_POINTS = 100;

/** @returns number[] metres above sea level, aligned with `points`, or null on failure */
export async function fetchElevations(points, { signal } = {}) {
  if (!points?.length) return null;
  const slice = points.slice(0, MAX_POINTS);
  try {
    const url = `${ELEVATION}?${qs({
      latitude: slice.map((p) => Number(p.lat).toFixed(4)),
      longitude: slice.map((p) => Number(p.lon).toFixed(4)),
    })}`;
    const data = await getJson(url, 'Elevation', { cacheMs: 86400000, signal, timeout: 12000 });
    return Array.isArray(data.elevation) ? data.elevation : null;
  } catch {
    // Terrain is one input among several. Losing it degrades the flood estimate; the
    // engine handles a null by leaning on rainfall and reports instead.
    return null;
  }
}

/**
 * Which sampled points sit in a dip relative to the route as a whole.
 * @returns [{ index, elevation, drop }] where `drop` is metres below the route median
 */
export function lowLyingPoints(elevations, { minDrop = 2.5 } = {}) {
  if (!elevations?.length || elevations.length < 3) return [];
  const sorted = [...elevations].filter((e) => Number.isFinite(e)).sort((a, b) => a - b);
  if (!sorted.length) return [];
  const median = sorted[Math.floor(sorted.length / 2)];
  return elevations
    .map((elevation, index) => ({ index, elevation, drop: median - elevation }))
    .filter((p) => Number.isFinite(p.elevation) && p.drop >= minDrop);
}

/** A single absolute-lowness figure for a route, 0-1. Coastal Chennai is near sea level. */
export function lownessScore(elevations) {
  const valid = (elevations || []).filter((e) => Number.isFinite(e));
  if (!valid.length) return null;
  const min = Math.min(...valid);
  // Below 3 m is genuinely flood-prone ground; above 15 m is not a pooling risk.
  if (min <= 3) return 1;
  if (min >= 15) return 0;
  return (15 - min) / 12;
}
