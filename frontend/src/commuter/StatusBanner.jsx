export function Glyph({ tier }) {
  const p = { width: 22, height: 22, viewBox: '0 0 22 22', fill: 'none', 'aria-hidden': true, stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round', strokeLinejoin: 'round' };
  if (tier === 'ok') return <svg {...p}><circle cx="11" cy="11" r="9" /><path d="M6.8 11.4 9.7 14.3 15.4 8" /></svg>;
  if (tier === 'critical') return <svg {...p}><path d="M11 2.5 20.5 19H1.5z" /><path d="M11 8.5v5M11 16h.01" /></svg>;
  return <svg {...p}><path d="M11 2.5 20.5 19H1.5z" /><path d="M11 8.5v4.5M11 16h.01" /></svg>;
}

/** The one bold sentence people read first. */
export default function StatusBanner({ tier, title, detail, children }) {
  return (
    <div className={`cm-banner ${tier}`} role={tier === 'critical' ? 'alert' : 'status'}>
      <span className="cm-banner-ic"><Glyph tier={tier} /></span>
      <div>
        <div className="cm-banner-title">{title}</div>
        {detail && <div className="cm-banner-detail">{detail}</div>}
        {children}
      </div>
    </div>
  );
}
