import { Suspense, lazy } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import Shell from '@/components/Shell';
import { ErrorState, Loading } from '@/components/ui';
import { useStore } from '@/data/store';
import { useApp } from '@/data/consumer';
import AlertWatcher from '@/commuter/AlertWatcher';
import Onboarding from '@/commuter/Onboarding';

// The pilot-model pages, for reviewers. Unchanged from the research build.
const Overview = lazy(() => import('@/pages/Overview'));
const SmartTrip = lazy(() => import('@/pages/SmartTrip'));
const LiveMap = lazy(() => import('@/pages/LiveMap'));
const Forecast = lazy(() => import('@/pages/Forecast'));
const EventReplay = lazy(() => import('@/pages/EventReplay'));
const ModelLab = lazy(() => import('@/pages/ModelLab'));
const DataSystem = lazy(() => import('@/pages/DataSystem'));

// The consumer app.
const Home = lazy(() => import('@/commuter/Home'));
const Trip = lazy(() => import('@/commuter/Trip'));
const Weather = lazy(() => import('@/commuter/Weather'));
const WeatherMap = lazy(() => import('@/commuter/WeatherMap'));
const Report = lazy(() => import('@/commuter/Report'));
const Places = lazy(() => import('@/commuter/Places'));
const Alerts = lazy(() => import('@/commuter/Alerts'));
const Ask = lazy(() => import('@/commuter/Ask'));
const Settings = lazy(() => import('@/commuter/Settings'));
const About = lazy(() => import('@/commuter/About'));

function NotFound() {
  return (
    <ErrorState title="Page not found">
      That page does not exist. Use the menu to get back.
    </ErrorState>
  );
}

export default function App() {
  const { status, error, viewMode } = useStore();
  const { onboarded } = useApp();
  const consumer = viewMode === 'commuter';

  // First launch: one question, before anything else is shown.
  if (consumer && !onboarded) return <Onboarding />;

  return (
    <Shell>
      <AlertWatcher />

      {/*
        The pilot store only gates the DEVELOPER pages: those read the committed model
        exports. The consumer app depends on live public services instead, so it must
        not be blocked by a missing predictions file.
      */}
      {!consumer && status === 'loading' && <Loading label="Loading model data" rows={4} />}

      {!consumer && status === 'error' && (
        <ErrorState title="Could not load the model data" onRetry={() => window.location.reload()}>
          {error}. The developer view expects <code>public/data/predictions_synthetic_chennai.json</code> to exist. Run{' '}
          <code>python -m ml.export.export_predictions</code> from the repository root to regenerate it.
        </ErrorState>
      )}

      {(consumer || status === 'ready') && (
        <Suspense fallback={<Loading label="Loading" />}>
          <Routes>
            {consumer ? (
              <>
                <Route path="/" element={<Home />} />
                <Route path="/trip" element={<Trip />} />
                <Route path="/weather" element={<Weather />} />
                <Route path="/map" element={<WeatherMap />} />
                <Route path="/report" element={<Report />} />
                <Route path="/places" element={<Places />} />
                <Route path="/alerts" element={<Alerts />} />
                <Route path="/ask" element={<Ask />} />
                <Route path="/settings" element={<Settings />} />
                <Route path="/about" element={<About />} />
                {/* Research pages are developer-only. */}
                {['/forecast', '/replay', '/model', '/system'].map((path) => (
                  <Route key={path} path={path} element={<Navigate to="/" replace />} />
                ))}
              </>
            ) : (
              <>
                <Route path="/" element={<Overview />} />
                <Route path="/trip" element={<SmartTrip />} />
                <Route path="/map" element={<LiveMap />} />
                <Route path="/forecast" element={<Forecast />} />
                <Route path="/replay" element={<EventReplay />} />
                <Route path="/model" element={<ModelLab />} />
                <Route path="/system" element={<DataSystem />} />
                <Route path="/about" element={<About />} />
                <Route path="/settings" element={<Settings />} />
              </>
            )}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Suspense>
      )}
    </Shell>
  );
}
