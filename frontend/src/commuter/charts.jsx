/**
 * The two charts the weather page needs, drawn as inline SVG.
 *
 * Deliberately NOT a dual-axis chart. Rain in millimetres and temperature in degrees
 * are different scales, and overlaying them on two y-axes is the classic way to imply
 * a relationship that the data does not contain. They are two stacked charts sharing
 * one x-axis instead, which is readable and honest.
 *
 * Colours come from the project's existing validated token set in tokens.css -- the
 * same palette the Python palette checker measures for contrast and colour-vision
 * separation. Each chart carries a single series, so identity never rests on colour.
 */
import { useId, useState } from 'react';

const PAD = { left: 40, right: 10, top: 10, bottom: 18 };

function useHover() {
  const [hover, setHover] = useState(null);
  return [hover, setHover];
}

/** Precipitation, as bars. Magnitude from a zero baseline, so bars are correct. */
export function RainChart({ rows, height = 92, label = 'Rain expected' }) {
  const id = useId();
  const [hover, setHover] = useHover();
  if (!rows?.length) return null;

  const width = 720;
  const innerW = width - PAD.left - PAD.right;
  const innerH = height - PAD.top - PAD.bottom;
  const max = Math.max(1, ...rows.map((r) => r.value));
  const step = innerW / rows.length;
  const barW = Math.max(2, step - 2); // 2px surface gap between adjacent bars

  // Whole numbers unless the whole range is sub-millimetre, so "0.0" never collides
  // with the axis edge.
  const decimals = max < 3 ? 1 : 0;
  const ticks = [0, max / 2, max];

  return (
    <figure className="chart">
      <figcaption>{label} <span className="chart-unit">mm per hour</span></figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${id}-t`} className="chart-svg" preserveAspectRatio="none">
        <title id={`${id}-t`}>{label} over the next {rows.length} hours, peaking at {max.toFixed(1)} millimetres</title>
        {ticks.map((t) => {
          const y = PAD.top + innerH - (t / max) * innerH;
          return (
            <g key={t}>
              <line x1={PAD.left} x2={width - PAD.right} y1={y} y2={y} className="chart-grid" />
              <text x={PAD.left - 6} y={y + 3} className="chart-axis" textAnchor="end">{t.toFixed(decimals)}</text>
            </g>
          );
        })}
        {rows.map((row, i) => {
          const h = (row.value / max) * innerH;
          const x = PAD.left + i * step + (step - barW) / 2;
          const y = PAD.top + innerH - h;
          return (
            <rect
              key={i}
              x={x}
              y={row.value > 0 ? y : PAD.top + innerH - 1}
              width={barW}
              height={row.value > 0 ? Math.max(1.5, h) : 1}
              rx={row.value > 0 ? 2 : 0}
              className="chart-bar"
              onMouseEnter={() => setHover({ i, x: x + barW / 2, y, row })}
              onMouseLeave={() => setHover(null)}
            />
          );
        })}
        {hover && <circle cx={hover.x} cy={hover.y} r="4" className="chart-dot" />}
      </svg>
      <div className="chart-xaxis">
        {rows.filter((_, i) => i % Math.ceil(rows.length / 6) === 0).map((r) => (
          <span key={r.label}>{r.label}</span>
        ))}
      </div>
      {hover && (
        <p className="chart-tip" role="status">
          <b>{hover.row.label}</b> {hover.row.value.toFixed(1)} mm
          {hover.row.extra ? ` · ${hover.row.extra}` : ''}
        </p>
      )}
    </figure>
  );
}

/** Temperature, as a line. Change over time, so a line is the right mark. */
export function TempChart({ rows, height = 92, label = 'Temperature' }) {
  const id = useId();
  const [hover, setHover] = useHover();
  if (!rows?.length) return null;

  const width = 720;
  const innerW = width - PAD.left - PAD.right;
  const innerH = height - PAD.top - PAD.bottom;
  const values = rows.map((r) => r.value).filter(Number.isFinite);
  const min = Math.floor(Math.min(...values) - 1);
  const max = Math.ceil(Math.max(...values) + 1);
  const span = Math.max(1, max - min);
  const step = innerW / Math.max(1, rows.length - 1);

  const x = (i) => PAD.left + i * step;
  const y = (v) => PAD.top + innerH - ((v - min) / span) * innerH;
  const path = rows.map((r, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)} ${y(r.value).toFixed(1)}`).join(' ');

  // Label only the extremes, never every point.
  const hottest = rows.reduce((a, b) => (b.value > a.value ? b : a));
  const coolest = rows.reduce((a, b) => (b.value < a.value ? b : a));

  return (
    <figure className="chart">
      <figcaption>{label} <span className="chart-unit">°C</span></figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby={`${id}-t`} className="chart-svg" preserveAspectRatio="none">
        <title id={`${id}-t`}>{label} ranging from {coolest.value.toFixed(0)} to {hottest.value.toFixed(0)} degrees</title>
        {[min, (min + max) / 2, max].map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={width - PAD.right} y1={y(t)} y2={y(t)} className="chart-grid" />
            <text x={PAD.left - 6} y={y(t) + 3} className="chart-axis" textAnchor="end">{Math.round(t)}</text>
          </g>
        ))}
        <path d={path} className="chart-line" fill="none" />
        {rows.map((row, i) => (
          <circle
            key={i}
            cx={x(i)}
            cy={y(row.value)}
            r="9"
            fill="transparent"
            onMouseEnter={() => setHover({ i, row })}
            onMouseLeave={() => setHover(null)}
          />
        ))}
        {hover && <circle cx={x(hover.i)} cy={y(hover.row.value)} r="4.5" className="chart-dot-line" />}
      </svg>
      {hover && (
        <p className="chart-tip" role="status">
          <b>{hover.row.label}</b> {hover.row.value.toFixed(1)}°C
        </p>
      )}
    </figure>
  );
}

/** Daily min-max as a range bar. One axis, one measure. */
export function DayRange({ day, min, max, lo, hi }) {
  const span = Math.max(1, hi - lo);
  const left = ((min - lo) / span) * 100;
  const width = ((max - min) / span) * 100;
  return (
    <div className="dayrange" title={`${day}: ${Math.round(min)} to ${Math.round(max)} degrees`}>
      <span className="dayrange-track" aria-hidden="true">
        <i style={{ left: `${left}%`, width: `${Math.max(6, width)}%` }} />
      </span>
    </div>
  );
}
