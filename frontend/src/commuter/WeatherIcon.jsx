/**
 * Weather glyphs. Drawn inline rather than loaded as a font or sprite sheet so the page
 * stays self-contained and works offline once cached.
 */
export default function WeatherIcon({ name, size = 28, night = false }) {
  const p = { width: size, height: size, viewBox: '0 0 32 32', fill: 'none', 'aria-hidden': true };
  const s = { stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round', strokeLinejoin: 'round' };
  const cloud = <path d="M9 23a5 5 0 0 1 .6-10 7 7 0 0 1 13.2 1.6A4.6 4.6 0 0 1 22.5 23z" {...s} />;

  switch (name) {
    case 'sun':
      return night ? (
        <svg {...p}><path d="M22.5 19.6A8.4 8.4 0 0 1 12.4 9.5a8.6 8.6 0 1 0 10.1 10.1z" {...s} /></svg>
      ) : (
        <svg {...p}>
          <circle cx="16" cy="16" r="6" {...s} />
          <path d="M16 3v3M16 26v3M29 16h-3M6 16H3M25.2 6.8l-2.1 2.1M8.9 23.1l-2.1 2.1M25.2 25.2l-2.1-2.1M8.9 8.9 6.8 6.8" {...s} />
        </svg>
      );
    case 'cloud-sun':
      return (
        <svg {...p}>
          <circle cx="11" cy="11" r="4" {...s} />
          <path d="M11 4v2M4 11h2M16.2 5.8l-1.4 1.4M5.8 16.2l-1.4 1.4" {...s} />
          {cloud}
        </svg>
      );
    case 'cloud':
      return <svg {...p}>{cloud}</svg>;
    case 'drizzle':
      return <svg {...p}>{cloud}<path d="M12 26v2M18 26v2" {...s} /></svg>;
    case 'rain':
      return <svg {...p}>{cloud}<path d="M11 25.5 10 29M16 25.5 15 29M21 25.5 20 29" {...s} /></svg>;
    case 'rain-heavy':
      return (
        <svg {...p}>{cloud}
          <path d="M10 25 8.5 30M14 25l-1.5 5M18 25l-1.5 5M22 25l-1.5 5" {...s} strokeWidth="2.4" />
        </svg>
      );
    case 'storm':
      return <svg {...p}>{cloud}<path d="M17 24.5 13 29h4l-1.5 3.5" {...s} /><path d="M10 26l-1 3" {...s} /></svg>;
    case 'snow':
      return <svg {...p}>{cloud}<path d="M12 27h.01M17 27h.01M14.5 30h.01" {...s} strokeWidth="3" /></svg>;
    case 'fog':
      return <svg {...p}>{cloud}<path d="M7 26h18M9 29.5h14" {...s} /></svg>;
    case 'wind':
      return <svg {...p}><path d="M4 11h14a4 4 0 1 0-4-4M4 17h20a4 4 0 1 1-4 4M4 23h10" {...s} /></svg>;
    default:
      return <svg {...p}>{cloud}</svg>;
  }
}

/** Arrow showing where wind is blowing TO, which is what a traveller cares about. */
export function WindArrow({ degrees, size = 18 }) {
  if (degrees == null) return null;
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" style={{ transform: `rotate(${degrees + 180}deg)` }}>
      <path d="M12 3 L18 21 L12 17 L6 21 Z" fill="currentColor" />
    </svg>
  );
}
