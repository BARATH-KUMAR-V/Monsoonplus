/**
 * State for the consumer side of MonsoonPlus: who the user is, where they go, what
 * they have reported, and the weather for wherever they are looking.
 *
 * Deliberately a separate provider from the pilot-model store in store.jsx. That store
 * owns the trained 10-road network and its three data modes; this one owns the
 * "anywhere" product built on live public services. Keeping them apart means the
 * existing Model Lab, Replay and Data & System pages keep working untouched.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { DEFAULT_PROFILE, getProfile } from '@/engine/profiles';
import { detectWeatherAnomalies } from '@/engine/anomaly';
import { fetchAirQuality, fetchRecentHistory, fetchWeather, severeConditions } from '@/services/weather';
import { currentPosition, reverseGeocode } from '@/services/geocode';

const Ctx = createContext(null);
const KEY = 'monsoonplus.app.v1';
const CHENNAI = { name: 'Chennai', context: 'Tamil Nadu, India', lat: 13.0827, lon: 80.2707, kind: 'city' };

/** Community reports expire. An eight-hour-old puddle is not news. */
export const REPORT_TTL_HOURS = 8;

export const REPORT_TYPES = [
  { id: 'flood', label: 'Flooded road', icon: 'flood', severity: 'severe', blurb: 'Deep water, hard to pass' },
  { id: 'waterlogging', label: 'Waterlogging', icon: 'water', severity: 'moderate', blurb: 'Shallow water collecting' },
  { id: 'blocked', label: 'Road blocked', icon: 'blocked', severity: 'severe', blurb: 'Cannot get through at all' },
  { id: 'rain', label: 'Very heavy rain', icon: 'rain', severity: 'moderate', blurb: 'Hard to see, hard to ride' },
  { id: 'accident', label: 'Accident', icon: 'accident', severity: 'severe', blurb: 'Crash or breakdown' },
  { id: 'tree', label: 'Fallen tree or branch', icon: 'tree', severity: 'moderate', blurb: 'Blocking part of the road' },
  { id: 'other', label: 'Something else', icon: 'other', severity: 'moderate', blurb: 'Any other hazard' },
];
export const REPORT_BY_ID = Object.fromEntries(REPORT_TYPES.map((r) => [r.id, r]));

function load() {
  try {
    return JSON.parse(localStorage.getItem(KEY)) || {};
  } catch {
    return {};
  }
}

function save(state) {
  try {
    localStorage.setItem(KEY, JSON.stringify(state));
  } catch {
    /* private mode or quota: preferences are a convenience, never load-bearing */
  }
}

const fresh = (reports) =>
  (reports || []).filter((r) => Date.now() - new Date(r.at).getTime() < REPORT_TTL_HOURS * 3600000);

export function ConsumerProvider({ children }) {
  const saved = load();

  const [profileId, setProfileId] = useState(saved.profileId || DEFAULT_PROFILE);
  const [onboarded, setOnboarded] = useState(saved.onboarded === true);
  const [places, setPlaces] = useState(saved.places || []);
  const [reports, setReports] = useState(fresh(saved.reports));
  const [recentPlaces, setRecentPlaces] = useState(saved.recentPlaces || []);
  const [alertsEnabled, setAlertsEnabled] = useState(saved.alertsEnabled === true);
  const [voiceEnabled, setVoiceEnabled] = useState(saved.voiceEnabled === true);
  const [bigText, setBigText] = useState(saved.bigText === true);

  // Where the weather pages are pointed. Starts at the last place used, else Chennai.
  const [focus, setFocus] = useState(saved.focus || CHENNAI);
  const [located, setLocated] = useState({ state: 'idle', place: null, error: null });

  const [weather, setWeather] = useState({ state: 'idle', data: null, error: null });
  const [air, setAir] = useState({ state: 'idle', data: null, error: null });
  const [history, setHistory] = useState({ state: 'idle', data: null });

  const requestId = useRef(0);

  useEffect(() => {
    save({ profileId, onboarded, places, reports, recentPlaces, alertsEnabled, voiceEnabled, bigText, focus });
  }, [profileId, onboarded, places, reports, recentPlaces, alertsEnabled, voiceEnabled, bigText, focus]);

  useEffect(() => {
    document.documentElement.dataset.textSize = bigText ? 'large' : 'normal';
  }, [bigText]);

  // Drop expired reports on a timer as well as on load, so a long-open tab stays honest.
  useEffect(() => {
    const timer = setInterval(() => setReports((prev) => fresh(prev)), 300000);
    return () => clearInterval(timer);
  }, []);

  // ---- weather for the focused place --------------------------------------
  const loadWeather = useCallback(async (place) => {
    if (!place) return;
    const id = ++requestId.current;
    setWeather({ state: 'loading', data: null, error: null });
    setAir({ state: 'loading', data: null, error: null });

    try {
      const data = await fetchWeather(place.lat, place.lon);
      if (id !== requestId.current) return;
      setWeather({ state: 'ready', data, error: null });
    } catch (error) {
      if (id !== requestId.current) return;
      setWeather({ state: 'error', data: null, error: error.message });
    }

    // Air quality and history are extras: each fails on its own without taking the
    // page with it.
    fetchAirQuality(place.lat, place.lon)
      .then((data) => id === requestId.current && setAir({ state: 'ready', data, error: null }))
      .catch((error) => id === requestId.current && setAir({ state: 'error', data: null, error: error.message }));

    fetchRecentHistory(place.lat, place.lon)
      .then((data) => id === requestId.current && setHistory({ state: 'ready', data }))
      .catch(() => id === requestId.current && setHistory({ state: 'error', data: null }));
  }, []);

  useEffect(() => {
    loadWeather(focus);
  }, [focus, loadWeather]);

  // ---- geolocation ---------------------------------------------------------
  const locate = useCallback(async ({ moveFocus = true } = {}) => {
    setLocated({ state: 'locating', place: null, error: null });
    try {
      const pos = await currentPosition();
      const place = await reverseGeocode(pos.lat, pos.lon);
      setLocated({ state: 'ready', place, error: null });
      if (moveFocus) setFocus(place);
      return place;
    } catch (error) {
      setLocated({ state: 'error', place: null, error: error.message });
      return null;
    }
  }, []);

  // ---- saved places --------------------------------------------------------
  const savePlace = useCallback((place, role = 'custom') => {
    setPlaces((prev) => {
      const without = prev.filter((p) => !(p.role === role && role !== 'custom') && p.id !== place.id);
      return [...without, { ...place, role, savedAt: new Date().toISOString() }].slice(-12);
    });
  }, []);

  const removePlace = useCallback((id) => setPlaces((prev) => prev.filter((p) => p.id !== id)), []);

  const noteRecent = useCallback((place) => {
    if (!place) return;
    setRecentPlaces((prev) => [place, ...prev.filter((p) => p.id !== place.id)].slice(0, 6));
  }, []);

  // ---- community reports ---------------------------------------------------
  const addReport = useCallback((report) => {
    const type = REPORT_BY_ID[report.type] || REPORT_BY_ID.other;
    const entry = {
      id: `rep_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      at: new Date().toISOString(),
      severity: type.severity,
      label: type.label,
      ...report,
    };
    setReports((prev) => [entry, ...prev].slice(0, 100));
    return entry;
  }, []);

  const removeReport = useCallback((id) => setReports((prev) => prev.filter((r) => r.id !== id)), []);

  // ---- derived -------------------------------------------------------------
  const profile = useMemo(() => getProfile(profileId), [profileId]);

  const severe = useMemo(
    () => (weather.data ? severeConditions(weather.data) : []),
    [weather.data],
  );

  const anomalies = useMemo(
    () => (weather.data?.current && history.data ? detectWeatherAnomalies(weather.data.current, history.data) : []),
    [weather.data, history.data],
  );

  const home = places.find((p) => p.role === 'home') || null;
  const work = places.find((p) => p.role === 'work') || null;

  const value = {
    // identity
    profileId, setProfileId, profile, onboarded, setOnboarded,
    // places
    places, home, work, savePlace, removePlace,
    recentPlaces, noteRecent,
    focus, setFocus, located, locate,
    // weather
    weather, air, history, reloadWeather: () => loadWeather(focus),
    severe, anomalies,
    // reports
    reports, addReport, removeReport,
    // preferences
    alertsEnabled, setAlertsEnabled, voiceEnabled, setVoiceEnabled, bigText, setBigText,
  };

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp() {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error('useApp must be used inside <ConsumerProvider>');
  return ctx;
}

export { CHENNAI };
