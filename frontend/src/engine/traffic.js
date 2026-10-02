/**
 * Traffic ESTIMATE for roads outside the trained pilot network.
 *
 * MonsoonPlus has no live traffic feed on the free tier: TomTom's free quota cannot
 * cover a whole city, and OSRM's travel time is free-flow -- it is the time the trip
 * would take on an empty road. So for an arbitrary route this module applies a
 * transparent, declared pattern: Chennai's weekday peaks, plus the well-established
 * effect of rain on road speed.
 *
 * It is labelled ESTIMATE everywhere. Inside the pilot network the trained model's
 * prediction is used instead and labelled AI PREDICTION. The two are never blended.
 */
import { SOURCE } from '@/services/provenance';

/**
 * Multiplier on free-flow travel time by hour of day, Monday to Friday.
 * Chennai's evening peak is longer and heavier than its morning one.
 */
const WEEKDAY_CURVE = [
  1.00, 1.00, 1.00, 1.00, 1.00, 1.05, // 00-05
  1.15, 1.30, 1.50, 1.55, 1.40, 1.30, // 06-11
  1.25, 1.25, 1.22, 1.28, 1.40, 1.58, // 12-17
  1.70, 1.65, 1.45, 1.28, 1.15, 1.05, // 18-23
];

/** Weekends are busy later and never as sharply. */
const WEEKEND_CURVE = [
  1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
  1.02, 1.06, 1.12, 1.20, 1.28, 1.32,
  1.30, 1.26, 1.24, 1.26, 1.32, 1.40,
  1.45, 1.42, 1.32, 1.20, 1.10, 1.02,
];

/**
 * How much rain slows traffic. Published road-capacity studies put light rain at a few
 * per cent and heavy rain well past 15%; Chennai's drainage makes the heavy end worse,
 * and these values sit at the conservative edge of that range.
 */
function rainFactor(mmPerHour) {
  const mm = mmPerHour ?? 0;
  if (mm < 0.2) return 1.0;
  if (mm < 2.5) return 1.07;
  if (mm < 7.6) return 1.18;
  if (mm < 25) return 1.38;
  return 1.6;
}

export const TRAFFIC_BANDS = [
  { max: 1.15, key: 'free', label: 'Light', tone: 'smooth' },
  { max: 1.35, key: 'moderate', label: 'Moderate', tone: 'slow' },
  { max: 1.6, key: 'heavy', label: 'Heavy', tone: 'disruption' },
  { max: 99, key: 'jam', label: 'Very heavy', tone: 'jammed' },
];

export const trafficBand = (factor) => TRAFFIC_BANDS.find((b) => factor < b.max) || TRAFFIC_BANDS[3];

/**
 * @param baseMinutes  OSRM free-flow duration
 * @param departAt     Date the trip starts
 * @param rainMmPerHour  expected rain during the trip
 * @returns { minutes, factor, band, source, basis, delayFromTraffic, delayFromRain }
 */
export function estimateTravelTime(baseMinutes, departAt, rainMmPerHour) {
  const when = departAt instanceof Date ? departAt : new Date(departAt || Date.now());
  const day = when.getDay();
  const curve = day === 0 || day === 6 ? WEEKEND_CURVE : WEEKDAY_CURVE;

  // Interpolate across the hour so a 7:55 departure is not treated as 7:00.
  const hour = when.getHours();
  const blend = when.getMinutes() / 60;
  const congestion = curve[hour] * (1 - blend) + curve[(hour + 1) % 24] * blend;

  const rain = rainFactor(rainMmPerHour);
  const factor = congestion * rain;

  return {
    minutes: baseMinutes * factor,
    factor,
    congestion,
    rain,
    band: trafficBand(congestion),
    source: SOURCE.ESTIMATE,
    basis: 'typical Chennai traffic for this time of day, adjusted for rain',
    delayFromTraffic: baseMinutes * (congestion - 1),
    delayFromRain: baseMinutes * congestion * (rain - 1),
  };
}

/** 0-1 risk component for the route score. */
export const trafficRisk = (factor) => Math.min(1, Math.max(0, (factor - 1) / 0.9));
