/**
 * Time-dependent Dijkstra over the junction graph.
 *
 * "Time-dependent" means the cost of an edge depends on when you reach it: a route
 * that takes 40 minutes crosses its last segment 40 minutes from now, by which point
 * the forecast says conditions have changed. Each relaxation therefore looks up the
 * segment speed at `departure + elapsed`, not at departure.
 *
 * Flood-safe mode multiplies the traversal cost of risky edges, with the factors held
 * in config/segments.json:
 *   high flood risk     x4.0   (take a long way round rather than cross it)
 *   moderate flood risk x1.35  (prefer to avoid, but not at any price)
 * Those are preferences expressed as cost, not hard bans -- if the only path to a
 * destination crosses a flooded road, the router still returns it and the UI says so.
 */
import {
  ADJACENCY,
  JUNCTION_BY_ID,
  SEGMENT_BY_ID,
  SEGMENT_LENGTH_KM,
  ROUTING_PENALTIES,
} from '@/config/network';

const MAX_HORIZON = 60;

/**
 * @param predictor  from createPredictor()
 * @param options    { from, to, rain, mode: 'fast'|'safe', departMinutes, penalise:Set }
 * @returns [{ segmentId, from, to }] or null when unreachable
 */
export function findRoute(predictor, { from, to, rain, mode = 'fast', departMinutes = 0, penalise }) {
  if (!predictor || !JUNCTION_BY_ID[from] || !JUNCTION_BY_ID[to]) return null;
  if (from === to) return [];

  const cost = {};
  const elapsed = {};
  const previous = {};
  const settled = {};
  Object.keys(JUNCTION_BY_ID).forEach((id) => {
    cost[id] = Infinity;
    elapsed[id] = 0;
  });
  cost[from] = 0;

  for (;;) {
    let current = null;
    Object.keys(cost).forEach((id) => {
      if (!settled[id] && cost[id] < Infinity && (current === null || cost[id] < cost[current])) {
        current = id;
      }
    });
    if (current === null) break;
    settled[current] = true;
    if (current === to) break;

    ADJACENCY[current].forEach(({ to: neighbour, segmentId }) => {
      if (settled[neighbour]) return;

      const arrival = Math.min(MAX_HORIZON, departMinutes + elapsed[current]);
      const state = predictor.stateFor(segmentId, rain, arrival);
      if (!state) return;

      const minutes = (SEGMENT_LENGTH_KM[segmentId] / Math.max(state.speed, 1)) * 60;
      let edgeCost = minutes;
      if (mode === 'safe') {
        if (state.flood.index === 2) edgeCost *= ROUTING_PENALTIES.flood_high;
        else if (state.flood.index === 1) edgeCost *= ROUTING_PENALTIES.flood_moderate;
      }
      if (penalise && penalise.has(segmentId)) edgeCost *= ROUTING_PENALTIES.alt_route_reuse;

      if (cost[current] + edgeCost < cost[neighbour]) {
        cost[neighbour] = cost[current] + edgeCost;
        elapsed[neighbour] = elapsed[current] + minutes;
        previous[neighbour] = { from: current, segmentId };
      }
    });
  }

  if (cost[to] === Infinity) return null;

  const edges = [];
  let node = to;
  let guard = 0;
  while (node !== from && guard < 64) {
    const step = previous[node];
    if (!step) return null;
    edges.unshift({ segmentId: step.segmentId, from: step.from, to: node });
    node = step.from;
    guard += 1;
  }
  return edges;
}

/**
 * Up to `limit` distinct routes, found by progressively penalising the segments
 * already used. A plain k-shortest-paths would be more rigorous; this is the standard
 * cheap approximation and it reliably produces visibly different alternatives on a
 * graph this small.
 */
export function alternativeRoutes(predictor, options, limit = 3) {
  const found = [];
  const penalise = new Set();
  for (let attempt = 0; attempt < 6 && found.length < limit; attempt += 1) {
    const edges = findRoute(predictor, { ...options, penalise });
    if (!edges || edges.length === 0) break;
    const key = edges.map((e) => e.segmentId).join('>');
    if (!found.some((r) => r.key === key)) found.push({ key, edges });
    edges.forEach((e) => penalise.add(e.segmentId));
  }
  return found;
}

/** Walk a route and total up time, distance and worst conditions encountered. */
export function evaluateRoute(predictor, edges, { rain, departMinutes = 0, satelliteEnabled = true }) {
  if (!predictor || !edges) return null;

  let minutes = 0;
  let distance = 0;
  let worstLevel = 0;
  let worstFlood = 0;
  const legs = [];

  edges.forEach((edge) => {
    const arrival = Math.min(MAX_HORIZON, departMinutes + minutes);
    const state = predictor.stateFor(edge.segmentId, rain, arrival, { satelliteEnabled });
    if (!state) return;

    const km = SEGMENT_LENGTH_KM[edge.segmentId];
    const legMinutes = (km / Math.max(state.speed, 1)) * 60;
    minutes += legMinutes;
    distance += km;
    worstLevel = Math.max(worstLevel, state.level.index);
    worstFlood = Math.max(worstFlood, state.flood.index);

    legs.push({
      segmentId: edge.segmentId,
      name: SEGMENT_BY_ID[edge.segmentId].name,
      from: JUNCTION_BY_ID[edge.from].name,
      to: JUNCTION_BY_ID[edge.to].name,
      minutes: legMinutes,
      km,
      state,
    });
  });

  return {
    minutes,
    minutesRounded: Math.round(minutes),
    distance,
    averageSpeed: minutes > 0 ? distance / (minutes / 60) : 0,
    worstLevel,
    worstFlood,
    legs,
    crossesHighFlood: worstFlood === 2,
  };
}

/** "via Guindy - Taramani", or "direct" for a single-segment route. */
export function routeVia(edges) {
  if (!edges || edges.length <= 1) return 'direct';
  return `via ${edges
    .slice(0, -1)
    .map((e) => JUNCTION_BY_ID[e.to].name)
    .join(' - ')}`;
}

/**
 * "Leave now or wait?" -- evaluate the same route at several departure offsets and
 * pick the quickest. Returns every option so the UI can show the trade-off rather
 * than just the winner.
 */
export function departureAdvice(predictor, edges, { rain, offsets = [0, 15, 30], satelliteEnabled = true }) {
  const options = offsets
    .map((offset) => {
      const result = evaluateRoute(predictor, edges, {
        rain,
        departMinutes: offset,
        satelliteEnabled,
      });
      return result ? { offset, minutes: result.minutesRounded, detail: result } : null;
    })
    .filter(Boolean);

  if (!options.length) return null;
  const best = options.reduce((a, b) => (b.minutes < a.minutes ? b : a));
  return { options, best };
}
