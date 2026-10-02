/**
 * Multi-objective route scoring -- the decision engine.
 *
 * Five normalised 0-1 risk components, weighted by the user's profile:
 *
 *   Score = w_time x time + w_traffic x traffic + w_flood x flood
 *         + w_weather x weather + w_hazard x hazard
 *
 * Lower is better. The objective is the safest PRACTICAL route, so travel time is a
 * real term rather than a tiebreak -- a route that is twice as long to avoid a puddle
 * is not a good answer either.
 *
 * Every component is kept on the result, which is what makes "Why this route?" possible
 * without inventing a story after the fact.
 */
import { getProfile } from './profiles';
import { estimateRouteFlood } from './flood';
import { estimateTravelTime, trafficRisk } from './traffic';
import { SOURCE } from '@/services/provenance';
import { tripRainAt } from './exposure';

const clamp01 = (v) => Math.min(1, Math.max(0, v));

/**
 * Weather exposure for the trip: how wet, windy and murky it will be while under way.
 * Walking and riding feel this directly; a car mostly does not.
 */
function weatherRisk(samples, exposure) {
  if (!samples?.length) return { risk: 0, peakRainMm: 0, peakGust: 0, worstVisibility: null };
  const peakRainMm = exposure?.peak ?? 0;
  const peakGust = Math.max(...samples.map((s) => s.wind?.gust ?? 0));
  const visibilities = samples.map((s) => s.visibility).filter((v) => Number.isFinite(v));
  const worstVisibility = visibilities.length ? Math.min(...visibilities) : null;

  const wet = clamp01(peakRainMm / 20);
  const windy = clamp01((peakGust - 25) / 55);
  const murky = worstVisibility != null ? clamp01((4000 - worstVisibility) / 3500) : 0;

  return { risk: clamp01(wet * 0.6 + windy * 0.25 + murky * 0.15), peakRainMm, peakGust, worstVisibility };
}

/** Community reports touching this route. */
function hazardRisk(samples, reports) {
  if (!reports?.length || !samples?.length) return { risk: 0, hits: [] };
  const hits = [];
  reports.forEach((report) => {
    const near = samples.some((s) => {
      const dLat = (s.lat - report.lat) * 111;
      const dLon = (s.lon - report.lon) * 108;
      return Math.hypot(dLat, dLon) < 0.8;
    });
    if (near) hits.push(report);
  });
  if (!hits.length) return { risk: 0, hits: [] };
  const worst = hits.some((h) => h.severity === 'severe') ? 1 : 0.55;
  return { risk: clamp01(worst * (0.75 + 0.25 * Math.min(1, hits.length / 3))), hits };
}

/**
 * Score every candidate route together, because the time component is relative: a route
 * is "slow" only compared with the alternatives on offer.
 *
 * @param candidates [{ route, samples, elevations }]
 * @param options    { profileId, departAt, reports, modelFlood }
 * @returns scored routes, best first
 */
export function scoreRoutes(candidates, { profileId, departAt = new Date(), reports = [], modelFlood = null } = {}) {
  if (!candidates?.length) return [];
  const profile = getProfile(profileId);
  const w = profile.weights;

  const enriched = candidates.map(({ route, samples, elevations }) => {
    const exposure = tripRainAt(samples, departAt);
    const tripRain = exposure.peak;

    const time = estimateTravelTime(route.baseMinutes, departAt, tripRain);
    const flood = estimateRouteFlood(samples, { elevations, reports });
    const weather = weatherRisk(samples, exposure);
    const hazard = hazardRisk(samples, reports);

    // Inside the pilot network the trained model replaces the rule-based estimate.
    const floodResult =
      modelFlood && modelFlood[route.id]
        ? { ...modelFlood[route.id], source: SOURCE.MODEL }
        : flood;

    return { route, samples, elevations, time, flood: floodResult, weather, hazard, tripRain, exposure };
  });

  const times = enriched.map((e) => e.time.minutes);
  const fastest = Math.min(...times);
  const slowest = Math.max(...times);
  const spread = Math.max(1, slowest - fastest);

  const scored = enriched.map((entry) => {
    // Time risk is "how much worse than the fastest option", capped so that a route
    // 20+ minutes slower is simply "much slower" rather than scaling without limit.
    const extraMinutes = entry.time.minutes - fastest;
    const timeRisk = clamp01(Math.max(extraMinutes / 20, (extraMinutes / spread) * 0.6));

    const components = {
      time: timeRisk,
      traffic: trafficRisk(entry.time.factor),
      flood: entry.flood?.score ?? 0,
      weather: entry.weather.risk,
      hazard: entry.hazard.risk,
    };

    const score =
      w.time * components.time +
      w.traffic * components.traffic +
      w.flood * components.flood +
      w.weather * components.weather +
      w.hazard * components.hazard;

    // What actually pushed this route up or down, biggest first. This is the evidence
    // the explanation is built from, so it is computed here rather than re-derived.
    const contributions = Object.entries(components)
      .map(([key, value]) => ({ key, value, weighted: w[key] * value }))
      .sort((a, b) => b.weighted - a.weighted);

    return { ...entry, components, contributions, score, extraMinutes, profile };
  });

  const ranked = [...scored].sort((a, b) => a.score - b.score);
  const best = ranked[0];

  return ranked.map((entry, index) => ({
    ...entry,
    rank: index,
    recommended: entry === best,
    // How much worse than the recommendation, for the comparison table.
    scoreGap: entry.score - best.score,
  }));
}

/** Short word for an overall route verdict. */
export function routeVerdict(scored) {
  const flood = scored.flood?.score ?? 0;
  const hazard = scored.components.hazard;
  if (flood >= 0.65 || hazard >= 0.9) return { key: 'avoid', label: 'Avoid', tone: 'jammed' };
  if (flood >= 0.4 || hazard > 0 || scored.components.weather >= 0.55) return { key: 'caution', label: 'Caution', tone: 'slow' };
  return { key: 'clear', label: 'Clear', tone: 'smooth' };
}
