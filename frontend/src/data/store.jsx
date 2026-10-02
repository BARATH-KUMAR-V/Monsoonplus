/**
 * The app's single data context: which of the three modes is active, what data that
 * mode loaded, and the global controls (rain level, satellite toggle, contrast).
 *
 * The three modes
 * ---------------
 * dataset  reads the committed predictions_*.json. Deterministic, zero network calls,
 *          always works. This is the default and the safe demo path.
 * live     asks the optional local collector/backend for fresh TomTom + Open-Meteo
 *          data. Falls back to the last known payload, then to dataset mode, and says
 *          which of those it is showing. Never a blank screen.
 * log      reads collected log files for a chosen date. Handles "no logs yet"
 *          explicitly rather than inventing anything.
 */
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { createPredictor } from '@/lib/model';

const StoreContext = createContext(null);

export const MODES = [
  {
    id: 'dataset',
    label: 'Dataset',
    badge: 'Benchmark data - deterministic',
    blurb: 'Precomputed model outputs. No network, identical on every load.',
  },
  {
    id: 'live',
    label: 'Live',
    badge: 'Live',
    blurb: 'Real TomTom + Open-Meteo via the local collector. Optional.',
  },
  {
    id: 'log',
    label: 'Log',
    badge: 'Replay',
    blurb: 'Collected predicted-vs-actual logs, by date.',
  },
];

const DATA_BASE = `${import.meta.env.BASE_URL || '/'}data`.replace(/\/{2,}/g, '/');
const LIVE_API = import.meta.env.VITE_MONSOONPLUS_API || 'http://127.0.0.1:8000';
const STORAGE_KEY = 'monsoonplus.prefs.v1';

function loadPrefs() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
  } catch {
    return {};
  }
}

function savePrefs(prefs) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
  } catch {
    /* private browsing, quota, or storage disabled -- preferences are a nicety */
  }
}

async function fetchJson(url, { timeout = 8000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

export function StoreProvider({ children }) {
  const prefs = loadPrefs();

  const [mode, setMode] = useState(prefs.mode || 'dataset');
  const [rain, setRain] = useState(prefs.rain ?? 38);
  const [trend, setTrend] = useState(prefs.trend || 'rising');
  const [satelliteEnabled, setSatelliteEnabled] = useState(prefs.satelliteEnabled !== false);
  const [highContrast, setHighContrast] = useState(prefs.highContrast === true);
  const [savedRoutes, setSavedRoutes] = useState(prefs.savedRoutes || []);
  // 'commuter' is the default: plain-language screens. 'developer' shows the model internals.
  const [viewMode, setViewMode] = useState(prefs.viewMode === 'developer' ? 'developer' : 'commuter');
  const [alertsEnabled, setAlertsEnabled] = useState(prefs.alertsEnabled === true);

  const [chennai, setChennai] = useState(null);
  const [metrLa, setMetrLa] = useState(null);
  const [modelCard, setModelCard] = useState(null);
  const [status, setStatus] = useState('loading'); // loading | ready | error
  const [error, setError] = useState(null);

  const [live, setLive] = useState({ state: 'idle', payload: null, error: null, fetchedAt: null });
  const [logIndex, setLogIndex] = useState({ state: 'idle', dates: [], error: null });
  const [logData, setLogData] = useState({ state: 'idle', payload: null, date: null, error: null });

  // ---- persist preferences ------------------------------------------------
  useEffect(() => {
    savePrefs({ mode, rain, trend, satelliteEnabled, highContrast, savedRoutes, viewMode, alertsEnabled });
  }, [mode, rain, trend, satelliteEnabled, highContrast, savedRoutes, viewMode, alertsEnabled]);

  useEffect(() => {
    document.documentElement.dataset.contrast = highContrast ? 'high' : 'normal';
  }, [highContrast]);

  // ---- dataset mode: always loaded, it is the fallback for everything else --
  useEffect(() => {
    let cancelled = false;
    setStatus('loading');

    Promise.allSettled([
      fetchJson(`${DATA_BASE}/predictions_synthetic_chennai.json`),
      fetchJson(`${DATA_BASE}/predictions_metr_la.json`),
      fetchJson(`${DATA_BASE}/model_card.json`),
    ]).then(([chennaiResult, metrResult, cardResult]) => {
      if (cancelled) return;
      if (chennaiResult.status === 'fulfilled') {
        setChennai(chennaiResult.value);
        setStatus('ready');
      } else {
        setError(chennaiResult.reason?.message || 'could not load the Chennai dataset');
        setStatus('error');
      }
      if (metrResult.status === 'fulfilled') setMetrLa(metrResult.value);
      if (cardResult.status === 'fulfilled') setModelCard(cardResult.value);
    });

    return () => {
      cancelled = true;
    };
  }, []);

  // ---- live mode ----------------------------------------------------------
  const refreshLive = useCallback(async () => {
    setLive((prev) => ({ ...prev, state: 'loading', error: null }));
    try {
      const payload = await fetchJson(`${LIVE_API}/api/live`, { timeout: 6000 });
      setLive({
        state: 'ready',
        payload,
        error: null,
        fetchedAt: new Date().toISOString(),
      });
    } catch (err) {
      // Keep whatever we had: "last known data" beats a blank screen.
      setLive((prev) => ({
        state: prev.payload ? 'stale' : 'unavailable',
        payload: prev.payload,
        error: err.message,
        fetchedAt: prev.fetchedAt,
      }));
    }
  }, []);

  useEffect(() => {
    if (mode !== 'live') return undefined;
    refreshLive();
    const timer = setInterval(refreshLive, 120000);
    return () => clearInterval(timer);
  }, [mode, refreshLive]);

  // ---- log mode -----------------------------------------------------------
  useEffect(() => {
    if (mode !== 'log' || logIndex.state !== 'idle') return;
    setLogIndex({ state: 'loading', dates: [], error: null });
    fetchJson(`${DATA_BASE}/logs_index.json`, { timeout: 5000 })
      .then((payload) => {
        setLogIndex({ state: 'ready', dates: payload.entries || [], error: null });
      })
      .catch(() =>
        // No index file is the normal state before any collection has run.
        setLogIndex({ state: 'empty', dates: [], error: null }),
      );
  }, [mode, logIndex.state]);

  const loadLog = useCallback(async (entry) => {
    setLogData({ state: 'loading', payload: null, date: entry?.date ?? null, error: null });
    try {
      const payload = await fetchJson(`${DATA_BASE}/logs/${entry.file}`);
      setLogData({ state: 'ready', payload, date: entry.date, error: null });
    } catch (err) {
      setLogData({ state: 'error', payload: null, date: entry?.date ?? null, error: err.message });
    }
  }, []);

  // ---- derived ------------------------------------------------------------
  const predictor = useMemo(
    () => (chennai ? createPredictor(chennai.forecast_grid) : null),
    [chennai],
  );

  const confidenceBand = chennai?.confidence?.speed_band_kmh ?? null;
  const confidenceLabel = chennai?.confidence?.confidence_label_by_regime ?? null;

  const states = useMemo(() => {
    if (!predictor) return [];
    return predictor.allStates(rain, 0, { satelliteEnabled, confidenceBand, confidenceLabel });
  }, [predictor, rain, satelliteEnabled, confidenceBand, confidenceLabel]);

  const statesAt = useCallback(
    (minutes, rainOverride) =>
      predictor
        ? predictor.allStates(rainOverride ?? rain, minutes, {
            satelliteEnabled,
            confidenceBand,
            confidenceLabel,
          })
        : [],
    [predictor, rain, satelliteEnabled, confidenceBand, confidenceLabel],
  );

  /**
   * What the badge in the top bar says. Live mode that failed falls back to dataset
   * numbers, and this is where that is made explicit rather than hidden.
   */
  const effectiveMode = useMemo(() => {
    if (mode === 'live') {
      if (live.state === 'ready') return { id: 'live', label: 'LIVE', tone: 'ok' };
      if (live.state === 'stale') return { id: 'live', label: 'LIVE (last known)', tone: 'warn' };
      return { id: 'dataset', label: 'DATASET (live unavailable)', tone: 'warn' };
    }
    if (mode === 'log') {
      return logData.state === 'ready'
        ? { id: 'log', label: 'REPLAY', tone: 'ok' }
        : { id: 'log', label: 'REPLAY (no log loaded)', tone: 'warn' };
    }
    return { id: 'dataset', label: 'DATASET', tone: 'ok' };
  }, [mode, live.state, logData.state]);

  const saveRoute = useCallback((route) => {
    setSavedRoutes((prev) => {
      if (prev.some((r) => r.from === route.from && r.to === route.to)) return prev;
      return [...prev, route].slice(-8);
    });
  }, []);

  const removeRoute = useCallback((index) => {
    setSavedRoutes((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const value = {
    // mode
    mode,
    setMode,
    effectiveMode,
    // controls
    rain,
    setRain,
    trend,
    setTrend,
    satelliteEnabled,
    setSatelliteEnabled,
    highContrast,
    setHighContrast,
    // data
    status,
    error,
    chennai,
    metrLa,
    modelCard,
    predictor,
    states,
    statesAt,
    confidenceBand,
    confidenceLabel,
    // live
    live,
    refreshLive,
    liveApiUrl: LIVE_API,
    // logs
    logIndex,
    logData,
    loadLog,
    // view
    viewMode,
    setViewMode,
    alertsEnabled,
    setAlertsEnabled,
    // routes
    savedRoutes,
    saveRoute,
    removeRoute,
  };

  return <StoreContext.Provider value={value}>{children}</StoreContext.Provider>;
}

export function useStore() {
  const context = useContext(StoreContext);
  if (!context) throw new Error('useStore must be used inside <StoreProvider>');
  return context;
}
