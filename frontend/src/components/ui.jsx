/**
 * Shared presentational pieces.
 *
 * The important one is <StatusChip>: status is never communicated by colour alone, so
 * every chip renders a glyph, a text label and a colour together. The glyph shapes are
 * deliberately distinct in silhouette (tick, triangle, diamond, square) so they are
 * still separable in greyscale or at small sizes.
 */
import { clamp } from '@/lib/model';

export const STATUS_META = {
  smooth: { label: 'Smooth', glyph: 'check', dash: null },
  slow: { label: 'Slow', glyph: 'triangle', dash: '10 6' },
  disruption: { label: 'Disruption likely', glyph: 'diamond', dash: '2 7' },
  jammed: { label: 'Jammed', glyph: 'square', dash: '14 5 3 5' },
};

export const STATUS_COLOR = {
  smooth: 'var(--smooth)',
  slow: 'var(--slow)',
  disruption: 'var(--disruption)',
  jammed: 'var(--jammed)',
};

export function Glyph({ name, size = 10 }) {
  const common = { width: size, height: size, viewBox: '0 0 10 10', 'aria-hidden': true };
  switch (name) {
    case 'check':
      return (
        <svg {...common}>
          <path d="M1 5.4 3.7 8 9 2" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      );
    case 'triangle':
      return (
        <svg {...common}>
          <path d="M5 0.6 9.6 9H0.4z" fill="currentColor" />
        </svg>
      );
    case 'diamond':
      return (
        <svg {...common}>
          <path d="M5 0.4 9.6 5 5 9.6 0.4 5z" fill="currentColor" />
        </svg>
      );
    case 'square':
      return (
        <svg {...common}>
          <rect x="1" y="1" width="8" height="8" rx="1.2" fill="currentColor" />
        </svg>
      );
    default:
      return null;
  }
}

/** Colour + glyph + text, always all three. */
export function StatusChip({ status, children, title }) {
  const meta = STATUS_META[status] || { label: status, glyph: null };
  return (
    <span className={`chip ${status}`} title={title}>
      <span className="chip-glyph">
        <Glyph name={meta.glyph} />
      </span>
      {children || meta.label}
    </span>
  );
}

export function Chip({ tone = 'neutral', children, title }) {
  return (
    <span className={`chip ${tone}`} title={title}>
      {children}
    </span>
  );
}

export function Card({ title, subtitle, actions, children, className = '', ...rest }) {
  return (
    <section className={`card pad ${className}`} {...rest}>
      {(title || actions) && (
        <header className="card-head">
          <div>
            {title && <h2 className="card-title">{title}</h2>}
            {subtitle && <p className="card-sub">{subtitle}</p>}
          </div>
          {actions && <div className="inline">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, note, tone }) {
  return (
    <div className="card stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={tone ? { color: STATUS_COLOR[tone] } : undefined}>
        {value}
      </div>
      {note && <div className="stat-note">{note}</div>}
    </div>
  );
}

export function Segmented({ options, value, onChange, label }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
          title={option.title}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function Bar({ value, max = 100, tone = 'smooth' }) {
  const width = `${clamp((value / max) * 100, 0, 100)}%`;
  return (
    <div className="bar">
      <span style={{ width, background: STATUS_COLOR[tone] || 'var(--primary)' }} />
    </div>
  );
}

export function KeyValue({ label, children }) {
  return (
    <div className="kv">
      <span className="muted">{label}</span>
      <b>{children}</b>
    </div>
  );
}

/** A labelled explanation of where a number came from. Used liberally on purpose. */
export function Provenance({ children }) {
  return <p className="note">{children}</p>;
}

export function Loading({ label = 'Loading', rows = 3 }) {
  return (
    <div aria-busy="true" aria-live="polite">
      <span className="visually-hidden">{label}</span>
      <div className="stack">
        {Array.from({ length: rows }).map((_, index) => (
          <div key={index} className="skeleton" style={{ height: index === 0 ? 48 : 34 }} />
        ))}
      </div>
    </div>
  );
}

export function EmptyState({ title, children, action }) {
  return (
    <div className="state">
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {action}
    </div>
  );
}

export function ErrorState({ title = 'Something went wrong', children, onRetry }) {
  return (
    <div className="state" role="alert">
      <h3>{title}</h3>
      {children && <p>{children}</p>}
      {onRetry && (
        <button type="button" className="btn" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

/**
 * A horizontally scrollable wrapper for wide tables.
 *
 * A region that scrolls but contains nothing focusable is unreachable by keyboard --
 * axe flags this as `scrollable-region-focusable`, and it is a real barrier: a
 * keyboard-only user simply cannot see the columns past the fold. Giving the
 * container tabIndex=0 and a labelled region role makes it a proper scroll stop.
 */
export function TableScroll({ label, children }) {
  return (
    <div className="table-scroll" tabIndex={0} role="region" aria-label={label}>
      {children}
    </div>
  );
}

export function PageHead({ title, caption, actions }) {
  return (
    <div className="page-head">
      <div>
        <h1>{title}</h1>
        {caption && <p className="cap">{caption}</p>}
      </div>
      {actions && <div className="inline">{actions}</div>}
    </div>
  );
}

export function Banner({ tier = 'info', tag, children }) {
  return (
    <div className={`banner ${tier}`} role={tier === 'critical' ? 'alert' : 'status'}>
      {tag && <span className="banner-tag">{tag}</span>}
      <span>{children}</span>
    </div>
  );
}

/** Freshness dot: green under 15 min, amber under an hour, red beyond. */
export function FreshnessDot({ ageMinutes, ok = true }) {
  let tone = 'bad';
  if (ok && ageMinutes != null) {
    if (ageMinutes < 15) tone = 'ok';
    else if (ageMinutes < 60) tone = 'warn';
  }
  const text = !ok ? 'unavailable' : ageMinutes == null ? 'unknown' : `${Math.round(ageMinutes)} min old`;
  return (
    <span className="inline" style={{ gap: 6 }}>
      <span className={`dot ${tone}`} aria-hidden="true" />
      <span className="small muted">{text}</span>
    </span>
  );
}
