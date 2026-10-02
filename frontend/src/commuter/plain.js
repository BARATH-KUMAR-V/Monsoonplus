/**
 * Plain-language helpers for the commuter view.
 *
 * Everything here turns the model's per-road state into the three things a person on
 * the way home actually needs: is it clear, is it risky, and how many extra minutes.
 * No new model, no new thresholds -- it reads the same states the developer view uses.
 */
import { JUNCTIONS, SEGMENT_LENGTH_KM, haversineKm } from '@/config/network';
import { evaluateRoute, findRoute } from '@/lib/routing';

/** 0 = clear, 1 = caution, 2 = severe. */
export function hazardOf(state) {
  if (state.flood.index === 2 || state.level.index >= 3) return 2;
  if (state.flood.index === 1 || state.level.index >= 1) return 1;
  return 0;
}

export const HAZARD_WORD = ['Clear', 'Caution', 'Avoid'];
export const HAZARD_TONE = ['smooth', 'slow', 'jammed'];

/** Extra minutes versus an empty road. */
export function delayMinutes(state) {
  const km = SEGMENT_LENGTH_KM[state.segmentId];
  const now = (km / Math.max(state.speed, 1)) * 60;
  const free = (km / state.segment.free_flow_kmh) * 60;
  return Math.max(0, now - free);
}

export function roundDelay(minutes) {
  if (minutes < 2) return 'under 2 minutes';
  return `about ${Math.round(minutes / 5) * 5 || Math.round(minutes)} minutes`;
}

/** One short reason a road is flagged, in everyday words. */
export function whyFlagged(state) {
  if (state.flood.index === 2) return 'Waterlogging is likely';
  if (state.flood.index === 1) return 'Water may collect here';
  if (state.level.index >= 3) return 'Traffic is nearly at a standstill';
  if (state.level.index === 2) return 'Heavy traffic';
  if (state.level.index === 1) return 'Slower than usual';
  return 'Moving normally';
}

const severity = (s) => hazardOf(s) * 1000 + s.impact;

export function worstRoads(states, limit = 3) {
  return [...states]
    .filter((s) => hazardOf(s) > 0)
    .sort((a, b) => severity(b) - severity(a))
    .slice(0, limit);
}

/**
 * The bold line at the top of Home and Trip. "Likely" and "risk" are deliberate: the
 * app forecasts conditions, it does not receive eyewitness reports.
 */
export function headline(states) {
  const bad = worstRoads(states, 50);
  if (!bad.length) {
    return { tier: 'ok', title: 'Roads are mostly clear right now.', detail: 'No flooding or heavy delays expected on the monitored roads.' };
  }
  const top = bad[0];
  const severe = hazardOf(top) === 2;
  const what = top.flood.index === 2 ? 'Severe waterlogging risk' : severe ? 'Severe jam' : top.flood.index === 1 ? 'Flooding possible' : 'Slow traffic';
  const rest = bad.length - 1;
  return {
    tier: severe ? 'critical' : 'warning',
    title: `${what} on ${top.segment.name}.`,
    detail: `Expect ${roundDelay(delayMinutes(top))} extra${rest > 0 ? `, and ${rest} more road${rest > 1 ? 's' : ''} affected` : ''}.`,
  };
}

/** Status of a whole trip: worst thing on it plus the extra minutes it costs. */
export function tripStatus(predictor, edges, { rain, satelliteEnabled }) {
  const ev = evaluateRoute(predictor, edges, { rain, satelliteEnabled });
  if (!ev) return null;
  let freeMin = 0;
  let worst = null;
  ev.legs.forEach((leg) => {
    freeMin += (leg.km / leg.state.segment.free_flow_kmh) * 60;
    if (!worst || severity(leg.state) > severity(worst)) worst = leg.state;
  });
  const hazard = Math.max(0, ...ev.legs.map((l) => hazardOf(l.state)));
  return { ev, hazard, delay: Math.max(0, ev.minutes - freeMin), worst };
}

export function savedRouteStatus(predictor, route, opts) {
  const edges = findRoute(predictor, { from: route.from, to: route.to, rain: opts.rain, mode: 'safe' });
  if (!edges || !edges.length) return null;
  return { edges, ...tripStatus(predictor, edges, opts) };
}

export function savedRouteSentence(route, status) {
  const name = `${route.fromName} to ${route.toName}`;
  if (!status) return `We could not find a route for ${name}.`;
  if (status.hazard === 0) return `Your route ${name} is currently clear.`;
  const road = status.worst?.segment.name;
  if (status.hazard === 2) return `Avoid ${name} for now: ${whyFlagged(status.worst).toLowerCase()} on ${road}. Expect ${roundDelay(status.delay)} extra.`;
  return `${name}: ${whyFlagged(status.worst).toLowerCase()} on ${road}. Expect ${roundDelay(status.delay)} extra.`;
}

export function nearestJunction(lat, lon) {
  let best = null;
  JUNCTIONS.forEach((j) => {
    const d = haversineKm([lat, lon], [j.lat, j.lon]);
    if (!best || d < best.km) best = { id: j.id, name: j.name, km: d };
  });
  return best;
}
