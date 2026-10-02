/**
 * Place search and reverse lookup, over OpenStreetMap data.
 *
 * Photon (Komoot) is the primary: it is CORS-friendly, needs no key, and is built for
 * type-ahead. Nominatim is the fallback and handles some full-address queries better.
 * Both are free community services, so results are cached and search is debounced by
 * the caller.
 */
import { getJson, qs } from './http';

const PHOTON = 'https://photon.komoot.io/api';
const NOMINATIM = 'https://nominatim.openstreetmap.org';

/** Bias results towards the user's map view, otherwise Chennai. */
export const DEFAULT_BIAS = { lat: 13.0027, lon: 80.2206 };

function fromPhoton(feature) {
  const p = feature.properties || {};
  const [lon, lat] = feature.geometry?.coordinates || [];
  const detail = [p.street, p.district, p.city, p.county, p.state].filter(Boolean);
  // Drop a detail line that merely repeats the name.
  const context = detail.filter((d) => d !== p.name).slice(0, 3).join(', ');
  return {
    id: `photon:${p.osm_type || 'x'}${p.osm_id || Math.random()}`,
    name: p.name || p.street || p.city || 'Unnamed place',
    context: context || p.country || '',
    lat,
    lon,
    kind: p.osm_value || p.type || 'place',
  };
}

function fromNominatim(item) {
  const parts = String(item.display_name || '').split(',').map((s) => s.trim());
  return {
    id: `nominatim:${item.place_id}`,
    name: item.name || parts[0] || 'Unnamed place',
    context: parts.slice(1, 4).join(', '),
    lat: Number(item.lat),
    lon: Number(item.lon),
    kind: item.type || 'place',
  };
}

/**
 * Search for a place by text.
 * @returns [{ id, name, context, lat, lon, kind }]
 */
export async function searchPlaces(query, { bias = DEFAULT_BIAS, limit = 7, signal } = {}) {
  const text = String(query || '').trim();
  if (text.length < 3) return [];

  try {
    const url = `${PHOTON}?${qs({ q: text, limit, lat: bias?.lat, lon: bias?.lon, lang: 'en' })}`;
    const data = await getJson(url, 'Place search', { cacheMs: 300000, signal, timeout: 9000 });
    const results = (data.features || []).map(fromPhoton).filter((r) => Number.isFinite(r.lat));
    if (results.length) return results;
  } catch {
    /* fall through to Nominatim */
  }

  const url = `${NOMINATIM}/search?${qs({ format: 'jsonv2', q: text, limit, addressdetails: 0 })}`;
  const data = await getJson(url, 'Place search', { cacheMs: 300000, signal, timeout: 9000 });
  return (Array.isArray(data) ? data : []).map(fromNominatim).filter((r) => Number.isFinite(r.lat));
}

/** Turn a map tap or a GPS fix into a readable place name. */
export async function reverseGeocode(lat, lon, { signal } = {}) {
  const fallback = { id: `pt:${lat.toFixed(5)},${lon.toFixed(5)}`, name: 'Selected point', context: `${lat.toFixed(4)}, ${lon.toFixed(4)}`, lat, lon, kind: 'point' };
  try {
    const url = `${PHOTON}/reverse?${qs({ lat, lon, limit: 1, lang: 'en' })}`;
    const data = await getJson(url, 'Place lookup', { cacheMs: 300000, signal, timeout: 8000 });
    const feature = (data.features || [])[0];
    if (feature) return { ...fromPhoton(feature), lat, lon };
  } catch {
    /* a missing name is cosmetic -- never block the trip on it */
  }
  try {
    const url = `${NOMINATIM}/reverse?${qs({ format: 'jsonv2', lat, lon, zoom: 17 })}`;
    const data = await getJson(url, 'Place lookup', { cacheMs: 300000, signal, timeout: 8000 });
    if (data && data.display_name) return { ...fromNominatim(data), lat, lon };
  } catch {
    /* ignore */
  }
  return fallback;
}

/** Browser geolocation as a promise. The position never leaves the device. */
export function currentPosition({ timeout = 10000 } = {}) {
  return new Promise((resolve, reject) => {
    if (!('geolocation' in navigator)) {
      reject(new Error('This browser cannot share your location.'));
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude, accuracy: pos.coords.accuracy }),
      (err) => reject(new Error(err.code === 1 ? 'Location permission was denied.' : 'Could not get your location.')),
      { enableHighAccuracy: false, timeout, maximumAge: 120000 },
    );
  });
}
