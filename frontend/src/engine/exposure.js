/**
 * How much rain a trip is exposed to, computed in ONE place.
 *
 * This existed in two slightly different forms -- once in the route scoring and once in
 * the departure planner -- and they disagreed, so the "leave now" box and the route card
 * showed different travel times for the same departure. Both now call this.
 */

/** Rain at one sampled point, for the moment the traveller actually reaches it. */
export function rainAtPoint(sample, departAt) {
  const rows = sample.hourly || [];
  if (!rows.length) {
    // No series for this point: fall back to whatever the sample already carries.
    return Math.max(sample.rain?.nowMm ?? 0, sample.rain?.next3hMm ?? 0);
  }
  const arrival = departAt.getTime() + (sample.fraction || 0) * 40 * 60000;
  const nearest = rows.reduce((best, row) => (Math.abs(row.date - arrival) < Math.abs(best.date - arrival) ? row : best), rows[0]);
  const index = rows.indexOf(nearest);
  // The hour you arrive, and the two after it: a trip is not instantaneous.
  return Math.max(...rows.slice(index, index + 3).map((r) => r.precipitation ?? 0), nearest.precipitation ?? 0);
}

/** The worst and the average rain across a whole route, for one departure time. */
export function tripRainAt(samples, departAt) {
  if (!samples?.length) return { peak: 0, mean: 0, perPoint: [] };
  const perPoint = samples.map((sample) => rainAtPoint(sample, departAt));
  return {
    peak: Math.max(...perPoint),
    mean: perPoint.reduce((a, b) => a + b, 0) / perPoint.length,
    perPoint,
  };
}
