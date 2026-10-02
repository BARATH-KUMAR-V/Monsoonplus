/** Small formatting helpers. Kept in one place so units read the same everywhere. */

export const kmh = (v) => (v == null || Number.isNaN(v) ? '--' : `${Math.round(v)} km/h`);
export const mm = (v) => (v == null || Number.isNaN(v) ? '--' : `${v.toFixed(1)} mm/h`);
export const km = (v) => (v == null || Number.isNaN(v) ? '--' : `${v.toFixed(1)} km`);
export const mins = (v) => (v == null || Number.isNaN(v) ? '--' : `${Math.round(v)} min`);
export const pct = (v, digits = 0) =>
  v == null || Number.isNaN(v) ? '--' : `${v.toFixed(digits)}%`;

/** Minutes-since-midnight to a 12-hour clock. */
export function clockFromMinutes(minutes) {
  const total = ((Math.round(minutes) % 1440) + 1440) % 1440;
  const h24 = Math.floor(total / 60);
  const minute = total % 60;
  const suffix = h24 >= 12 ? 'PM' : 'AM';
  const h12 = h24 % 12 || 12;
  return `${h12}:${String(minute).padStart(2, '0')} ${suffix}`;
}

export function hhmm(date = new Date()) {
  return `${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;
}

/** "3 min ago" / "just now". */
export function relativeTime(iso) {
  if (!iso) return 'unknown';
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return 'unknown';
  const seconds = Math.max(0, Math.round((Date.now() - then) / 1000));
  if (seconds < 45) return 'just now';
  if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)} h ago`;
  return `${Math.round(seconds / 86400)} d ago`;
}

export function offsetLabel(minutes) {
  if (minutes === 0) return 'Now';
  return `+${minutes} min`;
}

/** Escape text destined for a CSV cell. */
export function csvCell(value) {
  const text = value == null ? '' : String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function toCsv(rows) {
  return rows.map((row) => row.map(csvCell).join(',')).join('\n');
}

/** Trigger a browser download of text content, no server involved. */
export function downloadText(filename, text, mime = 'text/plain;charset=utf-8') {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  // Revoke on the next tick; revoking synchronously can cancel the download in Safari.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
