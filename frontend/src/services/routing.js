/**
 * Route finding over the real OpenStreetMap road network, via the public OSRM service.
 *
 * This replaces nothing: the repo's own time-dependent Dijkstra over the 10-road pilot
 * network still exists and is still what the trained model drives. OSRM is what makes
 * "anywhere" possible -- it knows every road in the world, but nothing about rain. The
 * decision engine joins the two.
 *
 * Note on travel modes: the public OSRM demo server only hosts the car profile. A bike
 * or walking trip therefore follows driving roads, and the UI says so rather than
 * pretending it computed a footpath route.
 */
import { getJson, qs, ServiceError } from './http';

const OSRM = import.meta.env.VITE_OSRM_URL || 'https://router.project-osrm.org';

const pt = (p) => `${Number(p.lon).toFixed(6)},${Number(p.lat).toFixed(6)}`;

/** Collapse OSRM steps into the handful of road names a person would actually say. */
function majorRoads(route) {
  const names = [];
  (route.legs || []).forEach((leg) => {
    (leg.steps || []).forEach((step) => {
      const name = (step.name || '').trim();
      if (!name) return;
      const previous = names[names.length - 1];
      if (previous && previous.name === name) {
        previous.distance += step.distance || 0;
        return;
      }
      names.push({ name, distance: step.distance || 0 });
    });
  });
  return names
    .sort((a, b) => b.distance - a.distance)
    .slice(0, 6)
    .filter((r) => r.distance > 150);
}

/**
 * @param from  { lat, lon }
 * @param to    { lat, lon }
 * @returns [{ id, distanceKm, baseMinutes, coordinates:[[lat,lon]], roads, legsSteps }]
 *          `baseMinutes` is OSRM's free-flow estimate. It has no live traffic in it --
 *          the decision engine adds the traffic and weather view on top.
 */
export async function findRoutes(from, to, { alternatives = 3, signal } = {}) {
  if (!from || !to) return [];
  const url =
    `${OSRM}/route/v1/driving/${pt(from)};${pt(to)}?` +
    qs({ alternatives, overview: 'full', geometries: 'geojson', steps: true, continue_straight: false });

  const data = await getJson(url, 'Routing', { cacheMs: 120000, signal, timeout: 15000 });

  if (data.code === 'NoRoute') {
    throw new ServiceError('Routing', 'No road route exists between those two points.');
  }
  if (data.code !== 'Ok' || !Array.isArray(data.routes) || !data.routes.length) {
    throw new ServiceError('Routing', data.message || 'The routing service could not plan this trip.');
  }

  return data.routes.slice(0, 3).map((route, index) => ({
    id: `r${index}`,
    letter: String.fromCharCode(65 + index),
    distanceKm: route.distance / 1000,
    baseMinutes: route.duration / 60,
    // OSRM gives [lon, lat]; Leaflet wants [lat, lon].
    coordinates: (route.geometry?.coordinates || []).map(([lon, lat]) => [lat, lon]),
    roads: majorRoads(route),
  }));
}

/** Evenly spaced points along a route, for sampling weather and elevation. */
export function samplePoints(coordinates, count = 8) {
  if (!coordinates?.length) return [];
  if (coordinates.length <= count) return coordinates.map(([lat, lon], i) => ({ lat, lon, fraction: i / Math.max(1, coordinates.length - 1) }));
  const step = (coordinates.length - 1) / (count - 1);
  return Array.from({ length: count }, (_, i) => {
    const index = Math.round(i * step);
    const [lat, lon] = coordinates[index];
    return { lat, lon, fraction: index / (coordinates.length - 1) };
  });
}

/** Rough bounding box of a set of [lat, lon] pairs, for fitting the map. */
export function boundsOf(coordinates) {
  if (!coordinates?.length) return null;
  let [minLat, minLon, maxLat, maxLon] = [90, 180, -90, -180];
  coordinates.forEach(([lat, lon]) => {
    minLat = Math.min(minLat, lat);
    maxLat = Math.max(maxLat, lat);
    minLon = Math.min(minLon, lon);
    maxLon = Math.max(maxLon, lon);
  });
  return [[minLat, minLon], [maxLat, maxLon]];
}
