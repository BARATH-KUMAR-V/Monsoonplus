import { SOURCE_META } from '@/services/provenance';
import { relativeTime } from '@/lib/format';

/**
 * The little label that says where a number came from. It appears next to every claim
 * the app makes, because "82% chance of rain" and "moderate flood risk" are two very
 * different kinds of statement and the user deserves to know which is which.
 */
export default function SourceBadge({ source, basis, at, detail }) {
  const meta = SOURCE_META[source] || SOURCE_META.unavailable;
  const title = [meta.label, basis, at ? `updated ${relativeTime(at)}` : null].filter(Boolean).join(' · ');
  return (
    <span className={`src src-${source}`} title={title}>
      <span className="src-dot" aria-hidden="true" />
      {meta.short}
      {detail && <span className="src-extra">{detail}</span>}
    </span>
  );
}

/** The long form, for the bottom of a card. */
export function SourceLine({ source, basis, at }) {
  const meta = SOURCE_META[source] || SOURCE_META.unavailable;
  return (
    <p className="src-line">
      <SourceBadge source={source} />
      <span>
        {meta.label}
        {basis ? ` · ${basis}` : ''}
        {at ? ` · updated ${relativeTime(at)}` : ''}
      </span>
    </p>
  );
}
