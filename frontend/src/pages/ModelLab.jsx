/**
 * Page 6 - Model Lab.
 *
 * Every number here is read from the exported evaluation, which was produced by
 * actually running ml/training/evaluate.py against the test split. Nothing is typed in
 * by hand. The "honesty notes" block is generated from the numbers themselves, so an
 * unflattering result shows up here automatically rather than needing someone to
 * remember to mention it.
 */
import { useMemo, useState } from 'react';
import {
  Bar as RBar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip as RTooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useStore } from '@/data/store';
import {
  Banner,
  Card,
  Chip,
  EmptyState,
  PageHead,
  Provenance,
  Segmented,
  Stat,
  StatusChip,
  TableScroll,
} from '@/components/ui';
import { SEGMENT_BY_ID } from '@/config/network';
import { averageGate } from '@/lib/model';
import { kmh, mm, pct } from '@/lib/format';

const AXIS_STYLE = { fontSize: 11, fill: 'var(--muted)' };
const LABELS = [
  { key: 'clear', label: 'Clear' },
  { key: 'rain', label: 'Rain' },
  { key: 'heavy_rain', label: 'Heavy rain' },
];

function Architecture() {
  return (
    <Card title="Architecture" subtitle="Trimodal gated fusion over the road graph">
      <pre
        className="mono"
        style={{
          margin: 0,
          overflowX: 'auto',
          background: 'var(--surface-sunk)',
          border: '1px solid var(--line)',
          borderRadius: 'var(--radius-sm)',
          padding: 12,
          lineHeight: 1.6,
        }}
      >
{`traffic   (N, 24, 3)  --GRU-->  h_t  \\
weather   (N, 24, 3)  --GRU-->  h_w   >--  gated fusion  -->  GCN x2  -->  head
satellite (N, 2)      --MLP-->  h_s  /     (per-node, per-sample            |
                                            weights over 3 modalities)      v
                                                          speed at t+15 / t+30 / t+60`}
      </pre>
      <ul className="small" style={{ paddingLeft: 18, marginTop: 12, marginBottom: 0 }}>
        <li>
          The gate reads all three embeddings plus the current rainfall, so the mix can differ
          per road and per situation within one forward pass.
        </li>
        <li>
          Node count never appears in a weight shape, which is what lets a checkpoint pretrained
          on METR-LA's 207 sensors load onto Chennai's 10 segments.
        </li>
        <li>
          The head predicts a <strong>change</strong> from the last observed speed, so the model
          starts at roughly persistence and must earn any improvement.
        </li>
        <li>
          GCN is the default. GAT is a one-line swap:{' '}
          <code>python -m ml.training.train --graph-layer gat</code>.
        </li>
      </ul>
    </Card>
  );
}

function ErrorTable({ evaluation, title, subtitle }) {
  if (!evaluation) return null;
  const model = evaluation.model;
  const rows = [
    { name: 'monsoonplus', data: model, isModel: true },
    ...(evaluation.baselines || []).map((b) => ({ name: b.name, data: b, isModel: false })),
  ];

  return (
    <Card title={title} subtitle={subtitle}>
      <TableScroll label="Data table, scroll horizontally">
        <table className="data">
          <thead>
            <tr>
              <th>Model</th>
              <th className="num">MAE overall</th>
              <th className="num">Clear</th>
              <th className="num">Rain</th>
              <th className="num">Heavy rain</th>
              <th className="num">Rain-weighted</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.name}
                style={row.isModel ? { background: 'var(--primary-tint)', fontWeight: 700 } : undefined}
              >
                <td>{row.name.replace(/_/g, ' ')}</td>
                <td className="num">{row.data.mae_overall?.toFixed(2) ?? '--'}</td>
                <td className="num">{row.data.mae_by_label?.clear?.toFixed(2) ?? '--'}</td>
                <td className="num">{row.data.mae_by_label?.rain?.toFixed(2) ?? '--'}</td>
                <td className="num">{row.data.mae_by_label?.heavy_rain?.toFixed(2) ?? '--'}</td>
                <td className="num">{row.data.rain_weighted_mae?.toFixed(2) ?? '--'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </TableScroll>
      <p className="small muted" style={{ marginTop: 8 }}>
        All figures are mean absolute error in km/h on the held-out test split, averaged over
        t+15, t+30 and t+60. Lower is better. {evaluation.test_windows} test windows.
      </p>
    </Card>
  );
}

function ImprovementTable({ evaluation }) {
  if (!evaluation?.baselines?.length) return null;
  return (
    <Card
      title="Improvement over each baseline"
      subtitle="Split by weather, because an aggregate alone can hide the opposite story"
    >
      <TableScroll label="Data table, scroll horizontally">
        <table className="data">
          <thead>
            <tr>
              <th>Baseline</th>
              <th className="num">Overall</th>
              <th className="num">Clear</th>
              <th className="num">Rain</th>
              <th className="num">Heavy rain</th>
            </tr>
          </thead>
          <tbody>
            {evaluation.baselines.map((baseline) => {
              const byLabel = baseline.model_improvement_percent_by_label || {};
              const cell = (value) =>
                value == null ? (
                  '--'
                ) : (
                  <span style={{ color: value > 0 ? 'var(--smooth-text)' : 'var(--jammed-text)' }}>
                    {value > 0 ? '+' : ''}
                    {value.toFixed(1)}%
                  </span>
                );
              return (
                <tr key={baseline.name}>
                  <td>{baseline.name.replace(/_/g, ' ')}</td>
                  <td className="num">{cell(baseline.model_improvement_percent_overall)}</td>
                  <td className="num">{cell(byLabel.clear)}</td>
                  <td className="num">{cell(byLabel.rain)}</td>
                  <td className="num">{cell(byLabel.heavy_rain)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </TableScroll>
      <p className="small muted" style={{ marginTop: 8 }}>
        Positive means monsoonplus has the lower error.
      </p>
    </Card>
  );
}

function GateBreakdown({ evaluation }) {
  const gate = evaluation?.gate_by_regime;
  const claim = evaluation?.gate_shift_claim;
  if (!gate) return null;

  const data = LABELS.map((entry) => ({
    regime: entry.label,
    Traffic: gate[entry.key]?.traffic ?? 0,
    Weather: gate[entry.key]?.weather ?? 0,
    Satellite: gate[entry.key]?.satellite ?? 0,
  }));

  return (
    <Card
      title="What the model leans on"
      subtitle="Gate weights read straight out of the trained network"
      actions={
        claim && (
          <Chip tone={claim.supported ? 'smooth' : 'jammed'}>
            {claim.supported ? 'Claim supported' : 'Claim NOT supported'}
          </Chip>
        )
      }
    >
      <div style={{ height: 240 }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
            <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
            <XAxis dataKey="regime" tick={AXIS_STYLE} stroke="var(--line)" />
            <YAxis tick={AXIS_STYLE} unit="%" stroke="var(--line)" />
            <RTooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <RBar dataKey="Traffic" stackId="g" fill="var(--primary)" isAnimationActive={false} />
            <RBar dataKey="Weather" stackId="g" fill="var(--rain)" isAnimationActive={false} />
            <RBar dataKey="Satellite" stackId="g" fill="var(--flood)" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {claim && (
        <div
          className="banner"
          style={{
            marginTop: 12,
            marginBottom: 0,
            background: claim.supported ? 'var(--smooth-tint)' : 'var(--jammed-tint)',
            borderColor: claim.supported ? 'var(--smooth)' : 'var(--jammed)',
            color: claim.supported ? 'var(--smooth-text)' : 'var(--jammed-text)',
          }}
        >
          <span>{claim.reason}</span>
        </div>
      )}

      <Provenance>
        <strong>These are the model's own gate outputs</strong>, averaged per weather regime over
        the test split - not an attribution method and not an estimate. If the gate had not moved
        toward weather and satellite in heavy rain, this card would say so.
      </Provenance>
    </Card>
  );
}

function Playground() {
  const { predictor } = useStore();
  const [rain, setRain] = useState(60);
  const [segmentId, setSegmentId] = useState('r2');

  const state = predictor?.stateFor(segmentId, rain, 30);
  const gate = state?.gate;

  return (
    <Card
      title="Model playground"
      subtitle="Move the inputs, watch the gate and the prediction respond"
      actions={<Chip tone="neutral">Illustrative</Chip>}
    >
      <div className="grid g-2" style={{ gap: 12 }}>
        <div className="field">
          <label className="field-label" htmlFor="pg-segment">
            Road
          </label>
          <select
            id="pg-segment"
            className="inp"
            value={segmentId}
            onChange={(event) => setSegmentId(event.target.value)}
          >
            {Object.values(SEGMENT_BY_ID).map((segment) => (
              <option key={segment.id} value={segment.id}>
                {segment.name}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label className="field-label" htmlFor="pg-rain">
            Rainfall {mm(rain)}
          </label>
          <input
            id="pg-rain"
            type="range"
            min="0"
            max="100"
            value={rain}
            onChange={(event) => setRain(Number(event.target.value))}
          />
        </div>
      </div>

      {state && (
        <div style={{ marginTop: 12 }}>
          <div className="grid g-3" style={{ gap: 10 }}>
            <Stat label="Speed at t+30" value={kmh(state.predicted['t+30'])} />
            <Stat label="Status" value={state.level.label} tone={state.level.key} />
            <Stat label="Impact score" value={state.impact} />
          </div>
          {gate && (
            <div style={{ marginTop: 12 }}>
              <div className="field-label">Gate weights for this road at this rainfall</div>
              <div className="kv">
                <span className="muted">Traffic</span>
                <b>{pct(gate.traffic, 1)}</b>
              </div>
              <div className="kv">
                <span className="muted">Weather</span>
                <b>{pct(gate.weather, 1)}</b>
              </div>
              <div className="kv">
                <span className="muted">Satellite</span>
                <b>{pct(gate.satellite, 1)}</b>
              </div>
            </div>
          )}
        </div>
      )}

      <Provenance>
        <strong>Illustrative.</strong> This reads the exported forecast grid and interpolates
        between the rainfall levels the model was actually run at. It is the trained network's
        behaviour, sampled - not a live forward pass in your browser.
      </Provenance>
    </Card>
  );
}

function PerSegment({ evaluation }) {
  if (!evaluation?.per_segment?.length) return null;
  return (
    <Card title="Error per road" subtitle="Where the model is weakest">
      <TableScroll label="Data table, scroll horizontally">
        <table className="data">
          <thead>
            <tr>
              <th>Road</th>
              <th className="num">MAE</th>
              <th className="num">Clear</th>
              <th className="num">Rain</th>
              <th className="num">Heavy rain</th>
            </tr>
          </thead>
          <tbody>
            {[...evaluation.per_segment]
              .sort((a, b) => b.mae_overall - a.mae_overall)
              .map((row) => (
                <tr key={row.segment_id}>
                  <td>{SEGMENT_BY_ID[row.segment_id]?.name ?? row.segment_id}</td>
                  <td className="num">{row.mae_overall?.toFixed(2)}</td>
                  <td className="num">{row.mae_clear?.toFixed(2) ?? '--'}</td>
                  <td className="num">{row.mae_rain?.toFixed(2) ?? '--'}</td>
                  <td className="num">{row.mae_heavy_rain?.toFixed(2) ?? '--'}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </TableScroll>
    </Card>
  );
}

export default function ModelLab() {
  const { chennai, metrLa, modelCard } = useStore();
  const [dataset, setDataset] = useState('chennai');

  const evaluation =
    dataset === 'chennai' ? chennai?.evaluation : metrLa?.evaluation;
  const meta = dataset === 'chennai' ? chennai : metrLa;

  const rainVsClear = useMemo(() => {
    if (!chennai?.evaluation) return [];
    const model = chennai.evaluation.model;
    const persistence = chennai.evaluation.baselines?.find((b) => b.name === 'persistence');
    return LABELS.map((entry) => ({
      regime: entry.label,
      monsoonplus: model.mae_by_label?.[entry.key] ?? null,
      persistence: persistence?.mae_by_label?.[entry.key] ?? null,
    }));
  }, [chennai]);

  if (!chennai) {
    return <EmptyState title="No evaluation data" />;
  }

  return (
    <>
      <PageHead
        title="Model Lab"
        caption="The numbers behind the forecasts, including the ones that do not flatter the model."
        actions={
          <Segmented
            label="Dataset"
            value={dataset}
            onChange={setDataset}
            options={[
              { value: 'chennai', label: 'Chennai' },
              { value: 'metrla', label: 'METR-LA' },
            ]}
          />
        }
      />

      {dataset === 'chennai' ? (
        <Banner tier="watch" tag="SYNTHETIC">
          Synthetic Chennai window. Generated from an explicit physical process, not measured
          Chennai traffic.
        </Banner>
      ) : (
        <Banner tier={metrLa?.is_real ? 'info' : 'watch'} tag={metrLa?.is_real ? 'REAL DATA' : 'STAND-IN'}>
          {metrLa?.is_real
            ? `Real METR-LA: 207 Los Angeles freeway detectors, 5-minute speeds. Evaluated on the first ${metrLa?.sensors_used} sensors so it runs on a free CPU tier.`
            : 'METR-LA stand-in - the real dataset was not downloaded.'}
        </Banner>
      )}

      <Architecture />

      {!evaluation ? (
        <EmptyState title="No evaluation for this dataset">
          Run <code>python -m ml.export.export_predictions</code> to generate it.
        </EmptyState>
      ) : (
        <>
          <div className="grid g-4" style={{ marginTop: 14 }}>
            <Stat
              label="MAE overall"
              value={`${evaluation.model.mae_overall?.toFixed(2)} km/h`}
              note={`${evaluation.test_windows} test windows`}
            />
            <Stat
              label="Clear weather"
              value={`${evaluation.model.mae_by_label?.clear?.toFixed(2) ?? '--'} km/h`}
              tone="smooth"
            />
            <Stat
              label="Heavy rain"
              value={`${evaluation.model.mae_by_label?.heavy_rain?.toFixed(2) ?? '--'} km/h`}
              tone="jammed"
            />
            <Stat
              label="Rain-weighted MAE"
              value={`${evaluation.model.rain_weighted_mae?.toFixed(2)} km/h`}
              note="The metric early stopping used"
            />
          </div>

          <div style={{ marginTop: 14 }}>
            <ErrorTable
              evaluation={evaluation}
              title={`${meta?.label ?? dataset} - model versus baselines`}
              subtitle="Same test split, same evaluation code for every row"
            />
          </div>

          <div style={{ marginTop: 14 }}>
            <ImprovementTable evaluation={evaluation} />
          </div>

          {evaluation.honesty_notes?.length > 0 && (
            <Card
              title="Honest reading of these numbers"
              subtitle="Generated from the results, not written by hand"
              style={{ marginTop: 14 }}
            >
              <ul style={{ paddingLeft: 18, margin: 0 }}>
                {evaluation.honesty_notes.map((note, index) => (
                  <li key={index} style={{ marginBottom: 8, fontSize: 12.5 }}>
                    {note}
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </>
      )}

      {dataset === 'chennai' && (
        <>
          <h2 className="section-title">Rain versus clear</h2>
          <div className="grid g-2">
            <Card title="Error by weather regime" subtitle="monsoonplus against persistence">
              <div style={{ height: 250 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={rainVsClear} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
                    <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
                    <XAxis dataKey="regime" tick={AXIS_STYLE} stroke="var(--line)" />
                    <YAxis tick={AXIS_STYLE} unit=" km/h" stroke="var(--line)" />
                    <RTooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <RBar dataKey="monsoonplus" fill="var(--primary)" isAnimationActive={false} />
                    <RBar dataKey="persistence" fill="var(--slow)" isAnimationActive={false} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <Provenance>
                This split is the project's central result. A single aggregate MAE is never
                reported without it.
              </Provenance>
            </Card>

            <GateBreakdown evaluation={chennai.evaluation} />
          </div>

          <div className="grid g-2" style={{ marginTop: 14 }}>
            <Playground />
            <PerSegment evaluation={chennai.evaluation} />
          </div>
        </>
      )}

      <h2 className="section-title">Where it fails</h2>
      <Card>
        <ul style={{ paddingLeft: 18, margin: 0, fontSize: 12.5, lineHeight: 1.7 }}>
          {(modelCard?.known_limitations || []).map((limitation, index) => (
            <li key={index}>{limitation}</li>
          ))}
          {!modelCard && (
            <li>
              model_card.json was not found. Run{' '}
              <code>python -m ml.export.export_predictions</code>.
            </li>
          )}
        </ul>
        <Provenance>
          Read from <code>model_card.json</code>, written by the export script. If you change what
          the model trains on, this list changes with it.
        </Provenance>
      </Card>

      {modelCard && (
        <Card title="Training recipe" subtitle="How the shipped checkpoint was produced" style={{ marginTop: 14 }}>
          <ol style={{ paddingLeft: 18, margin: 0, fontSize: 12.5, lineHeight: 1.7 }}>
            {modelCard.training_phases.map((phase, index) => (
              <li key={index}>{phase}</li>
            ))}
          </ol>
          <div style={{ marginTop: 12 }}>
            <div className="kv">
              <span className="muted">Model selection</span>
              <b>{modelCard.model_selection}</b>
            </div>
            <div className="kv">
              <span className="muted">Parameters</span>
              <b>{modelCard.parameters?.total?.toLocaleString()}</b>
            </div>
            <div className="kv">
              <span className="muted">Graph layer</span>
              <b>{modelCard.config?.graph_layer?.toUpperCase()}</b>
            </div>
            <div className="kv">
              <span className="muted">Horizons</span>
              <b>{modelCard.config?.horizons?.join(' / ')} min</b>
            </div>
          </div>
        </Card>
      )}
    </>
  );
}
