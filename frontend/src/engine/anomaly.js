/**
 * Anomaly detection over real observations.
 *
 * The baseline is the last few weeks of measured hourly data for the same location from
 * Open-Meteo, so "unusual" means unusual for this place recently -- not unusual against
 * a number someone typed into the code.
 *
 * Two detectors, both declared:
 *   z-score   for roughly symmetric quantities (temperature, wind)
 *   tail rank for rainfall, which is heavily skewed -- most hours are zero, so a mean
 *             and standard deviation are meaningless and a percentile is not
 *
 * The output deliberately states WHAT is unusual and never WHY. A sudden traffic
 * slowdown might be flooding, or an accident, or a procession. Claiming the cause from
 * one number would be exactly the kind of overreach this project should avoid.
 */

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

function stdev(xs, mu) {
  if (xs.length < 2) return 0;
  return Math.sqrt(xs.reduce((sum, x) => sum + (x - mu) ** 2, 0) / (xs.length - 1));
}

/** Fraction of baseline values at or below `value`. */
function percentileOf(value, baseline) {
  if (!baseline.length) return null;
  const below = baseline.filter((b) => b <= value).length;
  return below / baseline.length;
}

export function zScore(value, baseline) {
  const clean = baseline.filter(Number.isFinite);
  if (clean.length < 12 || !Number.isFinite(value)) return null;
  const mu = mean(clean);
  const sd = stdev(clean, mu);
  if (sd < 1e-6) return null;
  return { z: (value - mu) / sd, mean: mu, sd, n: clean.length };
}

/**
 * @param current  { temperature, rain, wind }
 * @param history  rows from fetchRecentHistory
 * @returns [{ kind, severity, title, detail, evidence }]
 */
export function detectWeatherAnomalies(current, history) {
  if (!history?.length || history.length < 48) return [];
  const found = [];

  // --- rainfall: percentile over wet hours only -----------------------------
  const rainSeries = history.map((h) => h.precipitation).filter(Number.isFinite);
  const wetHours = rainSeries.filter((r) => r > 0.1);
  const nowRain = current?.precipitation ?? 0;
  if (nowRain > 0.5 && wetHours.length >= 10) {
    const pct = percentileOf(nowRain, wetHours);
    const maxSeen = Math.max(...rainSeries);
    if (pct != null && pct >= 0.97) {
      found.push({
        kind: 'rain',
        severity: nowRain > maxSeen ? 'severe' : 'warning',
        title: 'Unusually heavy rain for this area',
        detail:
          nowRain > maxSeen
            ? `${nowRain.toFixed(1)} mm in the last hour is more than any hour in the past ${Math.round(history.length / 24)} days here.`
            : `${nowRain.toFixed(1)} mm in the last hour is heavier than ${Math.round(pct * 100)}% of rainy hours here recently.`,
        evidence: { value: nowRain, percentile: pct, baselineHours: rainSeries.length },
      });
    }
  }

  // --- temperature ----------------------------------------------------------
  const temp = zScore(current?.temperature_2m, history.map((h) => h.temperature_2m));
  if (temp && Math.abs(temp.z) >= 2.5) {
    found.push({
      kind: 'temperature',
      severity: Math.abs(temp.z) >= 3.2 ? 'warning' : 'info',
      title: temp.z > 0 ? 'Unusually hot for this area' : 'Unusually cool for this area',
      detail: `${current.temperature_2m.toFixed(1)}°C against a recent average of ${temp.mean.toFixed(1)}°C.`,
      evidence: { z: temp.z, mean: temp.mean, sd: temp.sd, n: temp.n },
    });
  }

  // --- wind -----------------------------------------------------------------
  const wind = zScore(current?.wind_speed_10m, history.map((h) => h.wind_speed_10m));
  if (wind && wind.z >= 2.5) {
    found.push({
      kind: 'wind',
      severity: wind.z >= 3.2 ? 'warning' : 'info',
      title: 'Unusually windy for this area',
      detail: `${Math.round(current.wind_speed_10m)} km/h against a recent average of ${Math.round(wind.mean)} km/h.`,
      evidence: { z: wind.z, mean: wind.mean, sd: wind.sd, n: wind.n },
    });
  }

  return found;
}

/**
 * Traffic anomaly: observed speed against this road's own normal.
 * Used in the pilot network where the app has per-segment history.
 */
export function detectTrafficAnomaly(currentSpeed, baselineSpeeds, { roadName } = {}) {
  const result = zScore(currentSpeed, baselineSpeeds);
  if (!result || result.z > -2.2) return null;
  return {
    kind: 'traffic',
    severity: result.z <= -3 ? 'warning' : 'info',
    title: `Unusual slowdown${roadName ? ` on ${roadName}` : ''}`,
    detail: `${Math.round(currentSpeed)} km/h against a usual ${Math.round(result.mean)} km/h. The cause is not known from this data alone.`,
    evidence: { z: result.z, mean: result.mean, sd: result.sd, n: result.n },
  };
}
