/**
 * Frontend smoke test. Run with `npm run smoke` from frontend/.
 *
 * Checks the things that silently break a static demo:
 *
 *  1. Every data file the app fetches exists and is valid JSON.
 *  2. No NaN or Infinity made it into any of them (invalid JSON; JSON.parse throws).
 *  3. The exported data agrees with config/segments.json about the road network.
 *  4. Every segment appears in the forecast grid, at every sampled rainfall level.
 *  5. Gate weights are a real distribution (sum to ~100%).
 *  6. Speeds are physically plausible.
 *  7. No page hardcodes a segment list of its own.
 *  8. The built site loads nothing from a CDN.
 *
 * Deliberately has no dependencies and starts no browser: it must run anywhere,
 * including in CI on a machine with no display.
 */

import { readFileSync, existsSync, readdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const frontend = join(here, '..');
const repo = join(frontend, '..');
const dataDir = join(frontend, 'public', 'data');

let failures = 0;
let checks = 0;

function check(label, condition, detail = '') {
  checks += 1;
  if (condition) {
    console.log(`  ok    ${label}`);
  } else {
    failures += 1;
    console.log(`  FAIL  ${label}${detail ? ` -- ${detail}` : ''}`);
  }
}

function section(title) {
  console.log(`\n${title}`);
}

// ---------------------------------------------------------------- 1 & 2 --
section('Data files');

const required = [
  'predictions_synthetic_chennai.json',
  'predictions_metr_la.json',
  'predictions_live_template.json',
  'model_card.json',
];

const loaded = {};
for (const name of required) {
  const path = join(dataDir, name);
  if (!existsSync(path)) {
    check(`${name} exists`, false, 'run: python -m ml.export.export_predictions');
    continue;
  }
  const raw = readFileSync(path, 'utf8');
  check(`${name} exists`, true);
  check(
    `${name} has no NaN/Infinity`,
    !/\bNaN\b/.test(raw) && !/\bInfinity\b/.test(raw),
    'NaN is not valid JSON and JSON.parse will throw in the browser',
  );
  try {
    loaded[name] = JSON.parse(raw);
    check(`${name} parses`, true);
  } catch (error) {
    check(`${name} parses`, false, error.message);
  }
}

const chennai = loaded['predictions_synthetic_chennai.json'];

// -------------------------------------------------------------------- 3 --
section('Agreement with config/segments.json');

const config = JSON.parse(readFileSync(join(repo, 'config', 'segments.json'), 'utf8'));
const configIds = config.segments.map((s) => s.id);
const configJunctions = config.junctions.map((j) => j.id);

check('config has 10 segments', configIds.length === 10, `found ${configIds.length}`);
check('config has 8 junctions', configJunctions.length === 8, `found ${configJunctions.length}`);

if (chennai) {
  const exportedIds = (chennai.network?.segments || []).map((s) => s.id);
  check(
    'export segment ids match config exactly',
    JSON.stringify(exportedIds) === JSON.stringify(configIds),
    `export=${exportedIds.join(',')}`,
  );

  // Attributions are a licence obligation. If the exported copy drifts from the
  // config, the deployed site credits the wrong providers.
  check(
    'exported attributions match config exactly',
    JSON.stringify(chennai.attributions) === JSON.stringify(config.attributions),
    'run: python -m ml.export.export_predictions',
  );

  // Every segment endpoint must be a real junction, or the map draws nothing.
  const dangling = (chennai.network?.segments || []).filter(
    (s) => !configJunctions.includes(s.a) || !configJunctions.includes(s.b),
  );
  check('no segment references a missing junction', dangling.length === 0);
}

// -------------------------------------------------------------- 4, 5, 6 --
section('Forecast grid');

if (chennai?.forecast_grid?.rows?.length) {
  const rows = chennai.forecast_grid.rows;
  check('grid has rainfall levels', rows.length >= 5, `${rows.length} levels`);

  const everyRowComplete = rows.every((row) => row.per_segment.length === configIds.length);
  check('every rainfall level covers every segment', everyRowComplete);

  let gateOk = true;
  let speedOk = true;
  let horizonOk = true;
  const horizons = config.model.horizons_minutes;

  for (const row of rows) {
    for (const entry of row.per_segment) {
      const gate = entry.gate || {};
      const total = (gate.traffic ?? 0) + (gate.weather ?? 0) + (gate.satellite ?? 0);
      if (Math.abs(total - 100) > 1.5) gateOk = false;

      if (!(entry.speed_now_kmh > 0 && entry.speed_now_kmh < 120)) speedOk = false;
      for (const h of horizons) {
        const value = entry.predicted_kmh?.[`t+${h}`];
        if (typeof value !== 'number' || value <= 0 || value > 120) horizonOk = false;
      }
    }
  }

  check('gate weights sum to 100% everywhere', gateOk);
  check('current speeds are physically plausible', speedOk);
  check(`predictions exist for every horizon (${horizons.join('/')} min)`, horizonOk);

  // Rain must actually slow things down, or the whole premise is broken.
  const dry = rows[0];
  const wet = rows[rows.length - 1];
  const mean = (row) =>
    row.per_segment.reduce((a, p) => a + p.speed_now_kmh, 0) / row.per_segment.length;
  check(
    'heavier rain yields lower speeds',
    mean(wet) < mean(dry),
    `dry ${mean(dry).toFixed(1)} vs wet ${mean(wet).toFixed(1)} km/h`,
  );
} else {
  check('forecast grid present', false, 'run the exporter');
}

// -------------------------------------------------------------- evaluation --
section('Evaluation payload');

if (chennai?.evaluation) {
  const evaluation = chennai.evaluation;
  check('model MAE present', typeof evaluation.model?.mae_overall === 'number');
  check('baselines present', (evaluation.baselines || []).length >= 2);
  check(
    'rain-vs-clear split present',
    evaluation.model?.mae_by_label &&
      'clear' in evaluation.model.mae_by_label &&
      'heavy_rain' in evaluation.model.mae_by_label,
    'the project never reports an aggregate without this split',
  );
  check('gate-by-regime present', Boolean(evaluation.gate_by_regime));
  check('honesty notes array present', Array.isArray(evaluation.honesty_notes));

  // Modality ablation (Model Lab "Does each modality earn its place?"). Optional as a
  // whole, but if any part is present it must be complete and self-consistent.
  const byName = Object.fromEntries((evaluation.baselines || []).map((b) => [b.name, b]));
  const ablationNames = [
    'traffic_only_masked',
    'traffic_weather_masked',
    'traffic_only_linear',
    'traffic_weather_linear',
    'traffic_weather_satellite_linear',
  ];
  const present = ablationNames.filter((name) => name in byName);
  check(
    'modality-ablation baselines are all present or all absent',
    present.length === 0 || present.length === ablationNames.length,
    `found ${present.join(', ') || 'none'}`,
  );
  if (present.length) {
    check(
      'ablation rows have a finite heavy-rain MAE',
      ablationNames.every((name) => Number.isFinite(byName[name]?.mae_by_label?.heavy_rain)),
    );
    check(
      'linear-probe rows were scored on the same windows as the model',
      ['traffic_only_linear', 'traffic_weather_linear', 'traffic_weather_satellite_linear'].every(
        (name) =>
          Object.values(byName[name].count_by_label).reduce((a, b) => a + b, 0) === evaluation.test_windows,
      ),
      `model test_windows = ${evaluation.test_windows}`,
    );
    check(
      'ablation methodology is stated alongside the numbers',
      typeof evaluation.ablation_methodology === 'string' && evaluation.ablation_methodology.length > 40,
    );
  }
  if (evaluation.network_ablation) {
    const names = (evaluation.network_ablation.rows || []).map((row) => row.name);
    check(
      'retrained-network ablation has traffic_only / traffic_weather / all_three',
      ['traffic_only', 'traffic_weather', 'all_three'].every((name) => names.includes(name)),
    );
  }
} else {
  check('evaluation present', false);
}

// -------------------------------------------------------------------- 7 --
section('Single source of truth');

const networkJs = readFileSync(join(frontend, 'src', 'config', 'network.js'), 'utf8');
check(
  'frontend imports config/segments.json',
  networkJs.includes('@config/segments.json'),
  'the UI must not keep its own copy of the road list',
);

const pagesDir = join(frontend, 'src', 'pages');
let hardcoded = [];
for (const file of readdirSync(pagesDir)) {
  const text = readFileSync(join(pagesDir, file), 'utf8');
  if (/['"]r1['"]\s*,\s*['"]r2['"]/.test(text)) hardcoded.push(file);
}
check('no page hardcodes a segment list', hardcoded.length === 0, hardcoded.join(', '));

// -------------------------------------------------------------------- 8 --
section('No external resources');

const indexHtml = readFileSync(join(frontend, 'index.html'), 'utf8');
const cdnPattern = /(src|href)=["']https?:\/\//gi;
const externals = indexHtml.match(cdnPattern) || [];
check(
  'index.html loads nothing from a CDN',
  externals.length === 0,
  externals.join(', '),
);

const distDir = join(frontend, 'dist');
if (existsSync(distDir)) {
  const builtHtml = readFileSync(join(distDir, 'index.html'), 'utf8');
  const builtExternals = builtHtml.match(cdnPattern) || [];
  check('built index.html loads nothing from a CDN', builtExternals.length === 0);
  // axe-core is imported inside an `import.meta.env.DEV` branch, which Vite replaces
  // with `false` when building. If it ever leaks into the bundle, that is ~600 KB of
  // development tooling shipped to every visitor.
  const bundleFiles = readdirSync(join(distDir, 'assets'));
  let axeLeaked = false;
  for (const file of bundleFiles) {
    if (!file.endsWith('.js')) continue;
    const text = readFileSync(join(distDir, 'assets', file), 'utf8');
    if (text.includes('axe.run') || /axe-core/.test(text)) axeLeaked = true;
  }
  check('axe-core is NOT in the production bundle', !axeLeaked);

  // The font must be served from our own origin, never fetched from Google.
  const hasWoff2 = bundleFiles.some((f) => f.endsWith('.woff2'));
  check('Public Sans ships as a self-hosted woff2', hasWoff2);

  let googleFonts = false;
  for (const file of bundleFiles) {
    if (!file.endsWith('.css')) continue;
    const text = readFileSync(join(distDir, 'assets', file), 'utf8');
    if (/fonts\.(googleapis|gstatic)\.com/.test(text)) googleFonts = true;
  }
  check('no Google Fonts request in the built CSS', !googleFonts);
} else {
  console.log('  skip  built output not present (run `npm run build` to include it)');
}

// -------------------------------------------------------------------------
console.log(`\n${checks - failures}/${checks} checks passed`);
if (failures > 0) {
  console.error(`${failures} check(s) FAILED`);
  process.exit(1);
}
console.log('frontend smoke test passed');
