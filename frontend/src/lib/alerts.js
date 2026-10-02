/**
 * Weather and traffic alerts.
 *
 * These are PLAIN THRESHOLD RULES over the forecast, not a second machine-learning
 * model. The rules are listed below in full and the UI prints the same wording, so
 * nobody reading the Forecast page can mistake "Heavy rain starts in 15 min" for a
 * learned prediction. The only learned component anywhere in the alert text is the
 * traffic speed, which comes from MonsoonPlusNet.
 *
 * Rule set:
 *   R1  rain crosses 15 mm/h within 60 min      -> warning  "Heavy rain starts in N min"
 *   R2  rain crosses 55 mm/h within 120 min     -> critical "Cloudburst likely in N min"
 *   R3  wetness crosses the high-flood band     -> warning  "Flood risk rises in N h"
 *   R4  any segment drops below 40% free-flow   -> warning  "<road> expected to jam"
 *   R5  rain falling and currently above 15     -> info     "Rain easing"
 */
import { THRESHOLDS } from '@/config/network';
import { rainBucket } from '@/lib/model';

const HEAVY = THRESHOLDS.rain_label_mm_h.heavy_rain_min;
const CLOUDBURST = THRESHOLDS.rain_regimes.steady_max;

export const ALERT_RULES = [
  { id: 'R1', text: `Rain forecast to cross ${HEAVY} mm/h within the next hour`, tier: 'warning' },
  { id: 'R2', text: `Rain forecast to cross ${CLOUDBURST} mm/h within two hours`, tier: 'critical' },
  { id: 'R3', text: 'Accumulated wetness forecast to reach the high flood-risk band', tier: 'warning' },
  { id: 'R4', text: 'A road is forecast to fall below 40% of its free-flow speed', tier: 'warning' },
  { id: 'R5', text: 'Rain currently heavy but easing', tier: 'info' },
];

const TIER_ORDER = { critical: 0, warning: 1, watch: 2, info: 3 };

/**
 * @param hourly   [{ time, precipitation }] from Open-Meteo or the fallback
 * @param states   current per-segment model states
 * @param rainNow  current rain mm/h
 */
export function buildAlerts({ hourly = [], states = [], rainNow = 0, trend = 'steady' }) {
  const alerts = [];

  const crossing = (threshold, withinHours) => {
    for (let i = 0; i < Math.min(hourly.length, withinHours); i += 1) {
      if ((hourly[i].precipitation ?? 0) >= threshold) return i;
    }
    return -1;
  };

  const heavyIn = crossing(HEAVY, 2);
  if (heavyIn >= 0 && rainNow < HEAVY) {
    const value = hourly[heavyIn].precipitation;
    alerts.push({
      rule: 'R1',
      tier: 'warning',
      title:
        heavyIn === 0
          ? `Heavy rain starting now (${value.toFixed(0)} mm/h forecast)`
          : `Heavy rain starts in about ${heavyIn} h (${value.toFixed(0)} mm/h forecast)`,
      detail: 'Threshold rule over the Open-Meteo forecast, not a model prediction.',
    });
  }

  const burstIn = crossing(CLOUDBURST, 3);
  if (burstIn >= 0) {
    alerts.push({
      rule: 'R2',
      tier: 'critical',
      title:
        burstIn === 0
          ? 'Cloudburst conditions now'
          : `Cloudburst conditions possible in about ${burstIn} h`,
      detail: `Forecast rainfall crosses ${CLOUDBURST} mm/h. Threshold rule.`,
    });
  }

  const highFlood = states.filter((s) => s.flood.index === 2);
  if (highFlood.length) {
    alerts.push({
      rule: 'R3',
      tier: 'warning',
      title: `Flood risk high on ${highFlood.length} road${highFlood.length > 1 ? 's' : ''}`,
      detail: highFlood.map((s) => s.segment.name).join(', '),
    });
  }

  const jamming = states.filter((s) => s.level.index === 3);
  if (jamming.length) {
    alerts.push({
      rule: 'R4',
      tier: 'warning',
      title: `${jamming.length} road${jamming.length > 1 ? 's' : ''} at or below 40% of free-flow speed`,
      detail: jamming.map((s) => s.segment.name).join(', '),
    });
  }

  if (rainNow >= HEAVY && trend === 'easing') {
    alerts.push({
      rule: 'R5',
      tier: 'info',
      title: `Rain easing (${rainBucket(rainNow)} now)`,
      detail: 'Conditions should improve over the next hour.',
    });
  }

  if (!alerts.length) {
    alerts.push({
      rule: null,
      tier: 'info',
      title: 'No active alerts',
      detail: 'No threshold rule is currently triggered.',
    });
  }

  return alerts.sort((a, b) => TIER_ORDER[a.tier] - TIER_ORDER[b.tier]);
}

/**
 * "On days like today, traffic typically jams on X within N min" -- derived from the
 * exported historical baseline, or an explicit "not enough data" when it is absent.
 */
export function historicalLikeToday({ historical, states, rainNow }) {
  if (!historical || !historical.per_segment) {
    return { available: false, text: 'No historical baseline in this dataset yet.' };
  }
  const worst = [...states].sort((a, b) => b.impact - a.impact)[0];
  if (!worst) return { available: false, text: 'No road states available.' };

  const baseline = historical.per_segment[worst.segmentId];
  if (!baseline) {
    return { available: false, text: `No baseline recorded for ${worst.segment.name}.` };
  }
  const typical = baseline.reduce((a, b) => a + b, 0) / baseline.length;
  const drop = typical - worst.speed;
  const minutesToJam = drop > 0 ? Math.max(5, Math.round((drop / Math.max(typical, 1)) * 90)) : null;

  return {
    available: true,
    segment: worst.segment.name,
    typical,
    current: worst.speed,
    text:
      minutesToJam == null
        ? `${worst.segment.name} is running at or above its usual speed for this time of day.`
        : `At ${rainNow.toFixed(0)} mm/h, ${worst.segment.name} runs ${drop.toFixed(1)} km/h below its usual ${typical.toFixed(1)} km/h, and typically tightens further within about ${minutesToJam} min.`,
    source: historical.label,
  };
}
