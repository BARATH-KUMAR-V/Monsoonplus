/**
 * "Leave now, wait, or leave by ___?"
 *
 * Both questions are the same search: evaluate the route at a series of departure times
 * and pick the best one. Because the traffic estimate is time-of-day based and the rain
 * forecast is hourly, a later departure genuinely scores differently.
 */
import { estimateTravelTime } from './traffic';
import { estimateRouteFlood } from './flood';
import { tripRainAt } from './exposure';

/**
 * Evaluate departures at 15-minute steps.
 * @returns [{ offset, departAt, minutes, arriveAt, rainPeak, floodScore, penalty }]
 */
export function departureOptions(scored, { from = new Date(), horizonMinutes = 180, step = 15, reports = [] } = {}) {
  const options = [];
  for (let offset = 0; offset <= horizonMinutes; offset += step) {
    const departAt = new Date(from.getTime() + offset * 60000);
    const rain = tripRainAt(scored.samples, departAt);

    const samplesAtTime = (scored.samples || []).map((s, i) => ({
      ...s,
      rain: { ...s.rain, nowMm: rain.perPoint[i] ?? rain.peak, next3hMm: rain.perPoint[i] ?? rain.peak },
    }));

    const time = estimateTravelTime(scored.route.baseMinutes, departAt, rain.peak);
    const flood = estimateRouteFlood(samplesAtTime, { elevations: scored.elevations, reports });

    options.push({
      offset,
      departAt,
      minutes: time.minutes,
      arriveAt: new Date(departAt.getTime() + time.minutes * 60000),
      rainPeak: rain.peak,
      floodScore: flood?.score ?? 0,
      // Waiting costs real time, so a later slot must be meaningfully better to win.
      // 0.35 min of penalty per minute waited means a 30-minute wait must save ~10 min
      // of driving or remove a serious flood risk to be worth recommending.
      penalty: time.minutes + offset * 0.35 + (flood?.score ?? 0) * 45,
    });
  }
  return options;
}

/** The recommendation itself, in words. */
export function departureAdvice(scored, { from = new Date(), reports = [] } = {}) {
  const options = departureOptions(scored, { from, reports });
  if (!options.length) return null;

  const now = options[0];
  const best = options.reduce((a, b) => (b.penalty < a.penalty ? b : a));

  if (best.offset === 0) {
    const worsening = options.slice(1, 5).some((o) => o.minutes > now.minutes + 5 || o.floodScore > now.floodScore + 0.15);
    return {
      verdict: 'leave-now',
      options,
      best,
      headline: 'Leave now.',
      detail: worsening
        ? 'Conditions on this route are forecast to get worse over the next hour, so waiting would cost you time.'
        : 'Waiting is not forecast to make this trip any better.',
    };
  }

  const savedMinutes = Math.round(now.minutes - best.minutes);
  const floodDrop = now.floodScore - best.floodScore;
  const reasons = [];
  if (savedMinutes >= 3) reasons.push(`about ${savedMinutes} minutes less driving`);
  if (floodDrop > 0.15) reasons.push('less water on the road by then');
  if (now.rainPeak - best.rainPeak > 2) reasons.push('the heaviest rain should have passed');

  return {
    verdict: 'wait',
    options,
    best,
    headline: `Wait about ${best.offset} minutes.`,
    detail: reasons.length
      ? `Leaving at ${clock(best.departAt)} means ${reasons.join(', and ')}.`
      : `Leaving at ${clock(best.departAt)} scores better on your settings.`,
  };
}

/**
 * Arrive-by planning: the latest departure that still gets you there in time, preferring
 * a safer slot when several work.
 */
export function arriveByAdvice(scored, arriveBy, { from = new Date(), reports = [] } = {}) {
  const horizon = Math.max(30, Math.round((arriveBy - from) / 60000));
  const options = departureOptions(scored, { from, horizonMinutes: horizon, step: 15, reports });
  const feasible = options.filter((o) => o.arriveAt <= arriveBy);

  if (!feasible.length) {
    const quickest = options.reduce((a, b) => (a.minutes < b.minutes ? a : b));
    const shortBy = Math.round((quickest.arriveAt - arriveBy) / 60000);
    return {
      verdict: 'too-late',
      options,
      headline: `You are unlikely to make ${clock(arriveBy)}.`,
      detail: `Even leaving straight away, this trip is forecast to take ${Math.round(quickest.minutes)} minutes, arriving about ${shortBy} minutes late.`,
    };
  }

  // Among departures that arrive in time, prefer the one with least risk, and break ties
  // towards leaving later so the user is not made to rush for no reason.
  const best = feasible.reduce((a, b) => {
    const costA = a.floodScore * 60 + a.minutes * 0.5 - a.offset * 0.08;
    const costB = b.floodScore * 60 + b.minutes * 0.5 - b.offset * 0.08;
    return costB < costA ? b : a;
  });
  const latest = feasible[feasible.length - 1];

  return {
    verdict: 'plan',
    options,
    best,
    latest,
    headline: `Leave at ${clock(best.departAt)}.`,
    detail:
      best.offset === 0
        ? `That gets you there around ${clock(best.arriveAt)}. The latest you could leave is ${clock(latest.departAt)}, but conditions are forecast to be worse by then.`
        : `That arrives around ${clock(best.arriveAt)}, with ${Math.round((arriveBy - best.arriveAt) / 60000)} minutes to spare.`,
  };
}

export function clock(date) {
  return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
}
