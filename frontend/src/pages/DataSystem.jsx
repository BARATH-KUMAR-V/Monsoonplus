/**
 * Page 7 - Data & System.
 *
 * The "what am I actually looking at" page: which mode is active and why, how fresh
 * each input is, the satellite kill switch, and the attributions that must appear on
 * the live site.
 */
import { useMemo } from 'react';
import { useStore } from '@/data/store';
import {
  Banner,
  Card,
  Chip,
  EmptyState,
  FreshnessDot,
  PageHead,
  Provenance,
  Segmented,
  Stat,
} from '@/components/ui';
import { MODES } from '@/data/store';
import { ATTRIBUTIONS, SEGMENTS, JUNCTIONS, HORIZONS } from '@/config/network';
import { relativeTime } from '@/lib/format';

function ModeExplainer() {
  const { mode, setMode, effectiveMode, live, logIndex, refreshLive, liveApiUrl } = useStore();
  const active = MODES.find((m) => m.id === mode);

  const reason = useMemo(() => {
    if (mode === 'dataset') {
      return 'Reading the committed prediction files. No network calls, identical on every load.';
    }
    if (mode === 'live') {
      switch (live.state) {
        case 'ready':
          return `Connected to the collector at ${liveApiUrl}. Showing live TomTom and Open-Meteo data.`;
        case 'stale':
          return `The collector stopped responding (${live.error}). Showing the last payload received, which is why the badge says "last known".`;
        case 'loading':
          return 'Contacting the local collector.';
        default:
          return `No collector reachable at ${liveApiUrl}, so the app has fallen back to Dataset mode. This is the expected state unless you started the backend yourself - see README section 6.`;
      }
    }
    return logIndex.dates.length
      ? `${logIndex.dates.length} collected log file(s) available.`
      : 'No collected logs exist yet, so Log mode has nothing to replay.';
  }, [mode, live, logIndex, liveApiUrl]);

  return (
    <Card
      title="Active mode"
      subtitle="Which of the three data paths is feeding the screens"
      actions={
        <Segmented
          label="Data mode"
          value={mode}
          onChange={setMode}
          options={MODES.map((m) => ({ value: m.id, label: m.label }))}
        />
      }
    >
      <div className="inline" style={{ marginBottom: 10 }}>
        <Chip tone={effectiveMode.tone === 'ok' ? 'smooth' : 'slow'}>{effectiveMode.label}</Chip>
        <span className="small muted">{active?.badge}</span>
      </div>
      <p style={{ margin: 0, fontSize: 12.5 }}>{reason}</p>

      {mode === 'live' && (
        <button type="button" className="btn small" style={{ marginTop: 12 }} onClick={refreshLive}>
          Retry now
        </button>
      )}

      <Provenance>
        The badge in the top bar shows this at all times. When Live mode falls back it says so
        rather than quietly serving precomputed numbers as live ones.
      </Provenance>
    </Card>
  );
}

function SourceStatus() {
  const { mode, live, satelliteEnabled, setSatelliteEnabled, chennai } = useStore();

  const liveAge = live.fetchedAt ? (Date.now() - new Date(live.fetchedAt).getTime()) / 60000 : null;
  const isLive = mode === 'live' && live.state === 'ready';

  const sources = [
    {
      key: 'traffic',
      name: 'Traffic',
      provider: 'TomTom Flow Segment Data',
      detail: isLive ? 'live via the local collector' : 'precomputed from the synthetic window',
      ok: true,
      age: isLive ? liveAge : 0,
      optional: 'Needs a free TomTom key. Optional - the app works fully without it.',
    },
    {
      key: 'weather',
      name: 'Weather',
      provider: 'Open-Meteo',
      detail: isLive ? 'live, no API key required' : 'precomputed rainfall sweep',
      ok: true,
      age: isLive ? liveAge : 0,
      optional: 'No key, no account, no card. Open-Meteo is keyless for non-commercial use.',
    },
    {
      key: 'satellite',
      name: 'Satellite',
      provider: 'Sentinel-2 NDWI',
      detail: satelliteEnabled
        ? 'synthetic NDWI derived from rainfall accumulation'
        : 'switched off',
      ok: satelliteEnabled,
      age: satelliteEnabled ? 0 : null,
      optional:
        'A real Earth Engine export replaces the synthetic signal. Free account, approval can take a day or two.',
    },
  ];

  return (
    <Card
      title="Input sources"
      subtitle={chennai?.generated_at ? `Dataset exported ${relativeTime(chennai.generated_at)}` : ''}
    >
      <div className="stack">
        {sources.map((source) => (
          <div
            key={source.key}
            style={{
              padding: 11,
              border: '1px solid var(--line)',
              borderRadius: 'var(--radius-sm)',
              background: 'var(--surface-sunk)',
            }}
          >
            <div className="spread">
              <div>
                <div style={{ fontWeight: 800, fontSize: 12.5 }}>{source.name}</div>
                <div className="small muted">{source.provider}</div>
              </div>
              <FreshnessDot ageMinutes={source.age} ok={source.ok} />
            </div>
            <p className="small" style={{ margin: '8px 0 0' }}>
              {source.detail}
            </p>
            <p className="small muted" style={{ margin: '4px 0 0' }}>
              {source.optional}
            </p>
            {source.key === 'satellite' && (
              <button
                type="button"
                className={`btn ${satelliteEnabled ? 'ghost' : ''} small`}
                style={{ marginTop: 8 }}
                aria-pressed={satelliteEnabled}
                onClick={() => setSatelliteEnabled(!satelliteEnabled)}
              >
                {satelliteEnabled ? 'Switch satellite off' : 'Switch satellite on'}
              </button>
            )}
          </div>
        ))}
      </div>
    </Card>
  );
}

function ModelStatus() {
  const { satelliteEnabled, chennai, modelCard } = useStore();
  const degraded = !satelliteEnabled;

  return (
    <Card title="Model status" subtitle={chennai?.checkpoint ?? 'checkpoint'}>
      <div className="inline" style={{ marginBottom: 10 }}>
        <Chip tone={degraded ? 'slow' : 'smooth'}>{degraded ? 'Degraded' : 'Nominal'}</Chip>
      </div>
      {degraded && (
        <Banner tier="watch" tag="DEGRADED">
          The satellite input is switched off, so flood risk is reported as Low everywhere and
          the Impact Score loses its flood term. The model is running on two modalities.
        </Banner>
      )}
      <div className="kv">
        <span className="muted">Architecture</span>
        <b>{modelCard?.config?.graph_layer?.toUpperCase() ?? 'GCN'} + gated fusion</b>
      </div>
      <div className="kv">
        <span className="muted">Parameters</span>
        <b>{modelCard?.parameters?.total?.toLocaleString() ?? '--'}</b>
      </div>
      <div className="kv">
        <span className="muted">Horizons</span>
        <b>{HORIZONS.join(' / ')} min</b>
      </div>
      <div className="kv">
        <span className="muted">Trained on</span>
        <b>{chennai?.checkpoint_meta?.dataset ?? 'synthetic Chennai window'}</b>
      </div>
      <div className="kv">
        <span className="muted">Network</span>
        <b>
          {SEGMENTS.length} segments, {JUNCTIONS.length} junctions
        </b>
      </div>
    </Card>
  );
}

export default function DataSystem() {
  const { chennai, metrLa, modelCard } = useStore();

  return (
    <>
      <PageHead
        title="Data & System"
        caption="Where every number on this site comes from, and what is switched on right now."
      />

      <div className="grid g-3">
        <ModeExplainer />
        <SourceStatus />
        <ModelStatus />
      </div>

      <h2 className="section-title">Datasets in this build</h2>
      <div className="grid g-2">
        <Card title="Chennai" subtitle={chennai?.label}>
          <div className="kv">
            <span className="muted">Real measurements?</span>
            <b>{chennai?.is_real ? 'Yes' : 'No - synthetic'}</b>
          </div>
          <div className="kv">
            <span className="muted">Source</span>
            <b className="small">{chennai?.evaluation?.dataset_source}</b>
          </div>
          <div className="kv">
            <span className="muted">Test windows</span>
            <b>{chennai?.evaluation?.test_windows}</b>
          </div>
          <div className="kv">
            <span className="muted">Window length</span>
            <b>{chennai?.window_summary?.days} days</b>
          </div>
          <div className="kv">
            <span className="muted">Weather mix</span>
            <b className="small">
              {chennai?.window_summary?.label_counts
                ? Object.entries(chennai.window_summary.label_counts)
                    .map(([k, v]) => `${k.replace('_', ' ')} ${v}`)
                    .join(', ')
                : '--'}
            </b>
          </div>
          <Provenance>
            Generated by <code>ml/data/chennai_synthetic.py</code> from an explicit physical
            process: storm cells, an accumulation-driven flood term, and per-road rain
            sensitivity. Labelled synthetic everywhere it appears.
          </Provenance>
        </Card>

        <Card title="METR-LA" subtitle={metrLa?.label ?? 'not exported'}>
          {metrLa ? (
            <>
              <div className="kv">
                <span className="muted">Real measurements?</span>
                <b>{metrLa.is_real ? 'Yes - real METR-LA' : 'No - stand-in'}</b>
              </div>
              <div className="kv">
                <span className="muted">Source file</span>
                <b className="small">{metrLa.source}</b>
              </div>
              <div className="kv">
                <span className="muted">Sensors used</span>
                <b>{metrLa.sensors_used}</b>
              </div>
              <div className="kv">
                <span className="muted">Test windows</span>
                <b>{metrLa.evaluation?.test_windows}</b>
              </div>
              <Provenance>{metrLa.note}</Provenance>
            </>
          ) : (
            <EmptyState title="METR-LA not exported">
              Run <code>python -m ml.export.export_predictions</code> without{' '}
              <code>--skip-metr-la</code>.
            </EmptyState>
          )}
        </Card>
      </div>

      <h2 className="section-title">Powered by</h2>
      <Card subtitle="These attributions are required by the data providers' terms">
        <ul style={{ paddingLeft: 18, margin: 0, fontSize: 12.5, lineHeight: 1.9 }}>
          {ATTRIBUTIONS.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
        <Provenance>
          Listed in <code>config/segments.json</code> so the site, the exports and the README
          cannot drift apart.
        </Provenance>
      </Card>

      <h2 className="section-title">Honest limits</h2>
      <Card>
        <ul style={{ paddingLeft: 18, margin: 0, fontSize: 12.5, lineHeight: 1.8 }}>
          {(modelCard?.known_limitations || []).map((limitation, index) => (
            <li key={index}>{limitation}</li>
          ))}
        </ul>
        <Banner tier="watch" tag="NOT A SAFETY SYSTEM" >
          monsoonplus is a student project and a planning aid. Never use it to decide whether a
          flooded road is safe to cross.
        </Banner>
      </Card>
    </>
  );
}
