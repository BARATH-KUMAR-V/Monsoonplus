/**
 * Weather, from Open-Meteo. Free, keyless, no registration, generous fair-use limits.
 *
 * Everything this module returns is an OFFICIAL forecast: it is Open-Meteo's own model
 * output, not MonsoonPlus's. The app's own LSTM predictions live in the model layer and
 * are labelled separately, so a reviewer can always tell which is which.
 */
import { getJson, qs } from './http';

const FORECAST = 'https://api.open-meteo.com/v1/forecast';
const AIR = 'https://air-quality-api.open-meteo.com/v1/air-quality';

const CURRENT_FIELDS = [
  'temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 'is_day',
  'precipitation', 'rain', 'weather_code', 'cloud_cover', 'pressure_msl',
  'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
];
const HOURLY_FIELDS = [
  'temperature_2m', 'relative_humidity_2m', 'apparent_temperature', 'precipitation_probability',
  'precipitation', 'rain', 'weather_code', 'cloud_cover', 'visibility',
  'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m', 'uv_index',
];
const DAILY_FIELDS = [
  'weather_code', 'temperature_2m_max', 'temperature_2m_min', 'apparent_temperature_max',
  'sunrise', 'sunset', 'uv_index_max', 'precipitation_sum', 'precipitation_hours',
  'precipitation_probability_max', 'wind_speed_10m_max', 'wind_gusts_10m_max',
];

/** WMO weather codes -> words a person uses. */
export const WMO = {
  0: { text: 'Clear sky', icon: 'sun' },
  1: { text: 'Mainly clear', icon: 'sun' },
  2: { text: 'Partly cloudy', icon: 'cloud-sun' },
  3: { text: 'Cloudy', icon: 'cloud' },
  45: { text: 'Fog', icon: 'fog' },
  48: { text: 'Freezing fog', icon: 'fog' },
  51: { text: 'Light drizzle', icon: 'drizzle' },
  53: { text: 'Drizzle', icon: 'drizzle' },
  55: { text: 'Heavy drizzle', icon: 'drizzle' },
  56: { text: 'Freezing drizzle', icon: 'drizzle' },
  57: { text: 'Freezing drizzle', icon: 'drizzle' },
  61: { text: 'Light rain', icon: 'rain' },
  63: { text: 'Rain', icon: 'rain' },
  65: { text: 'Heavy rain', icon: 'rain-heavy' },
  66: { text: 'Freezing rain', icon: 'rain' },
  67: { text: 'Freezing rain', icon: 'rain-heavy' },
  71: { text: 'Light snow', icon: 'snow' },
  73: { text: 'Snow', icon: 'snow' },
  75: { text: 'Heavy snow', icon: 'snow' },
  77: { text: 'Snow grains', icon: 'snow' },
  80: { text: 'Light showers', icon: 'rain' },
  81: { text: 'Showers', icon: 'rain' },
  82: { text: 'Violent showers', icon: 'rain-heavy' },
  85: { text: 'Snow showers', icon: 'snow' },
  86: { text: 'Snow showers', icon: 'snow' },
  95: { text: 'Thunderstorm', icon: 'storm' },
  96: { text: 'Thunderstorm with hail', icon: 'storm' },
  99: { text: 'Thunderstorm with hail', icon: 'storm' },
};

export const describeCode = (code) => WMO[code] || { text: 'Unknown', icon: 'cloud' };

/** Compass point from a wind bearing, so the UI never says "wind 247 degrees". */
export function compass(degrees) {
  if (degrees == null) return null;
  const points = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  return points[Math.round(((degrees % 360) / 45)) % 8];
}

/** Zip Open-Meteo's parallel arrays into a list of row objects. */
function rows(block, fields) {
  if (!block?.time) return [];
  return block.time.map((time, i) => {
    const row = { time, date: new Date(time) };
    fields.forEach((f) => {
      if (block[f]) row[f] = block[f][i];
    });
    return row;
  });
}

/**
 * The full weather picture for one place.
 * @returns { current, hourly, daily, units, fetchedAt, lat, lon, timezone }
 */
export async function fetchWeather(lat, lon, { days = 14, signal } = {}) {
  const url = `${FORECAST}?${qs({
    latitude: Number(lat).toFixed(4),
    longitude: Number(lon).toFixed(4),
    current: CURRENT_FIELDS,
    hourly: HOURLY_FIELDS,
    daily: DAILY_FIELDS,
    timezone: 'auto',
    forecast_days: days,
    past_hours: 6,
  })}`;

  const data = await getJson(url, 'Weather', { cacheMs: 600000, signal, timeout: 14000 });

  return {
    lat: data.latitude,
    lon: data.longitude,
    timezone: data.timezone,
    units: { ...data.current_units, ...data.hourly_units, ...data.daily_units },
    current: data.current ? { ...data.current, date: new Date(data.current.time) } : null,
    hourly: rows(data.hourly, HOURLY_FIELDS),
    daily: rows(data.daily, DAILY_FIELDS),
    fetchedAt: new Date().toISOString(),
  };
}

/** Air quality is a separate Open-Meteo endpoint, and is allowed to fail on its own. */
export async function fetchAirQuality(lat, lon, { signal } = {}) {
  const url = `${AIR}?${qs({
    latitude: Number(lat).toFixed(3),
    longitude: Number(lon).toFixed(3),
    current: ['pm10', 'pm2_5', 'us_aqi', 'carbon_monoxide', 'nitrogen_dioxide', 'ozone'],
    timezone: 'auto',
  })}`;
  const data = await getJson(url, 'Air quality', { cacheMs: 900000, signal, timeout: 10000 });
  return { ...data.current, units: data.current_units, fetchedAt: new Date().toISOString() };
}

/** US AQI bands, with the advice that actually matters to a person. */
export function aqiBand(aqi) {
  if (aqi == null) return null;
  if (aqi <= 50) return { label: 'Good', tone: 'smooth', advice: 'Air quality is fine for everyone.' };
  if (aqi <= 100) return { label: 'Moderate', tone: 'slow', advice: 'Fine for most people. Unusually sensitive people may want to limit long outdoor effort.' };
  if (aqi <= 150) return { label: 'Unhealthy for sensitive groups', tone: 'disruption', advice: 'Children, older adults and people with asthma should take it easy outdoors.' };
  if (aqi <= 200) return { label: 'Unhealthy', tone: 'jammed', advice: 'Everyone should reduce long or heavy outdoor activity.' };
  return { label: 'Very unhealthy', tone: 'jammed', advice: 'Avoid outdoor activity where you can. Consider a mask outdoors.' };
}

/** UV bands. */
export function uvBand(uv) {
  if (uv == null) return null;
  if (uv < 3) return { label: 'Low', tone: 'smooth', advice: 'No protection needed.' };
  if (uv < 6) return { label: 'Moderate', tone: 'slow', advice: 'Cover up around midday.' };
  if (uv < 8) return { label: 'High', tone: 'disruption', advice: 'Use sunscreen and shade from 11am to 3pm.' };
  if (uv < 11) return { label: 'Very high', tone: 'jammed', advice: 'Avoid the midday sun.' };
  return { label: 'Extreme', tone: 'jammed', advice: 'Stay out of the sun in the middle of the day.' };
}

/**
 * "Will it rain in the next three hours?"
 * Straight from the hourly forecast -- a reading of Open-Meteo, not a second model.
 */
export function nextRain(hourly, { hours = 3, now = new Date() } = {}) {
  if (!hourly?.length) return null;
  const upcoming = hourly.filter((h) => h.date >= new Date(now.getTime() - 30 * 60000)).slice(0, hours + 1);
  if (!upcoming.length) return null;

  const peak = upcoming.reduce((a, b) => ((b.precipitation ?? 0) > (a.precipitation ?? 0) ? b : a));
  const chance = Math.max(...upcoming.map((h) => h.precipitation_probability ?? 0));
  const firstWet = upcoming.find((h) => (h.precipitation ?? 0) >= 0.2 || (h.precipitation_probability ?? 0) >= 60);
  const minutesAway = firstWet ? Math.max(0, Math.round((firstWet.date - now) / 60000)) : null;

  return {
    chance,
    peakMm: peak.precipitation ?? 0,
    startsInMinutes: minutesAway,
    willRain: Boolean(firstWet),
    total: upcoming.reduce((sum, h) => sum + (h.precipitation ?? 0), 0),
  };
}

/** mm/h -> the words this app uses everywhere. */
export function rainWord(mmPerHour) {
  if (mmPerHour == null) return 'unknown';
  if (mmPerHour < 0.2) return 'none';
  if (mmPerHour < 2.5) return 'light';
  if (mmPerHour < 7.6) return 'moderate';
  if (mmPerHour < 25) return 'heavy';
  return 'very heavy';
}

/**
 * Severe-weather indication. These are transparent thresholds over the official
 * forecast -- the app's own indication, deliberately NOT presented as a government
 * warning. IMD issues those, and MonsoonPlus does not have a feed for them.
 */
export function severeConditions(weather, { hours = 12 } = {}) {
  if (!weather?.hourly?.length) return [];
  const now = Date.now();
  const window = weather.hourly.filter((h) => h.date >= now - 3600000 && h.date <= now + hours * 3600000);
  if (!window.length) return [];

  const found = [];
  const when = (row) => {
    const mins = Math.round((row.date - now) / 60000);
    if (mins <= 15) return 'now';
    if (mins < 60) return `in about ${mins} minutes`;
    return `in about ${Math.round(mins / 60)} hours`;
  };

  const storm = window.find((h) => [95, 96, 99].includes(h.weather_code));
  if (storm) found.push({ kind: 'storm', severity: 'severe', title: `Thunderstorm expected ${when(storm)}`, detail: 'Lightning is possible. Avoid open ground and shelter indoors.' });

  const extreme = window.find((h) => (h.precipitation ?? 0) >= 25);
  const heavy = window.find((h) => (h.precipitation ?? 0) >= 7.6);
  if (extreme) found.push({ kind: 'rain', severity: 'severe', title: `Very heavy rain expected ${when(extreme)}`, detail: `Up to ${extreme.precipitation.toFixed(0)} mm in an hour. Roads may flood and travel will be slow.` });
  else if (heavy) found.push({ kind: 'rain', severity: 'warning', title: `Heavy rain expected ${when(heavy)}`, detail: `Around ${heavy.precipitation.toFixed(1)} mm in an hour. Expect spray, poor visibility and slower traffic.` });

  const gust = window.find((h) => (h.wind_gusts_10m ?? 0) >= 60);
  if (gust) found.push({ kind: 'wind', severity: gust.wind_gusts_10m >= 85 ? 'severe' : 'warning', title: `Strong winds expected ${when(gust)}`, detail: `Gusts near ${Math.round(gust.wind_gusts_10m)} km/h. Difficult for two-wheelers and high-sided vehicles.` });

  const hot = window.find((h) => (h.apparent_temperature ?? 0) >= 40);
  if (hot) found.push({ kind: 'heat', severity: hot.apparent_temperature >= 45 ? 'severe' : 'warning', title: `Feels like ${Math.round(hot.apparent_temperature)}°C ${when(hot)}`, detail: 'Carry water and avoid long exposure in the middle of the day.' });

  const fog = window.find((h) => (h.visibility ?? 99999) < 1000);
  if (fog) found.push({ kind: 'fog', severity: 'warning', title: `Poor visibility expected ${when(fog)}`, detail: 'Visibility under 1 km. Use low beams and keep extra distance.' });

  const order = { severe: 0, warning: 1 };
  return found.sort((a, b) => order[a.severity] - order[b.severity]);
}

/**
 * Weather for several points at once (the points along a route).
 * Open-Meteo accepts comma-separated coordinates and answers with an array.
 */
export async function fetchWeatherAtPoints(points, { signal } = {}) {
  if (!points?.length) return [];
  const url = `${FORECAST}?${qs({
    latitude: points.map((p) => Number(p.lat).toFixed(3)),
    longitude: points.map((p) => Number(p.lon).toFixed(3)),
    hourly: ['precipitation', 'precipitation_probability', 'weather_code', 'wind_speed_10m', 'wind_gusts_10m', 'wind_direction_10m', 'visibility', 'temperature_2m'],
    timezone: 'auto',
    forecast_days: 2,
  })}`;

  const data = await getJson(url, 'Route weather', { cacheMs: 600000, signal, timeout: 15000 });
  const list = Array.isArray(data) ? data : [data];
  return list.map((entry, i) => ({
    ...points[i],
    hourly: rows(entry.hourly, ['precipitation', 'precipitation_probability', 'weather_code', 'wind_speed_10m', 'wind_gusts_10m', 'wind_direction_10m', 'visibility', 'temperature_2m']),
  }));
}

/**
 * Recent observed history for one place, used as the baseline for anomaly detection.
 * Open-Meteo serves up to 92 past days from the same endpoint, so this is real measured
 * and reanalysed data, not something the app invented.
 */
export async function fetchRecentHistory(lat, lon, { days = 28, signal } = {}) {
  const url = `${FORECAST}?${qs({
    latitude: Number(lat).toFixed(3),
    longitude: Number(lon).toFixed(3),
    hourly: ['temperature_2m', 'precipitation', 'wind_speed_10m'],
    past_days: days,
    forecast_days: 1,
    timezone: 'auto',
  })}`;
  const data = await getJson(url, 'Weather history', { cacheMs: 3600000, signal, timeout: 15000 });
  return rows(data.hourly, ['temperature_2m', 'precipitation', 'wind_speed_10m']).filter((r) => r.date <= new Date());
}
