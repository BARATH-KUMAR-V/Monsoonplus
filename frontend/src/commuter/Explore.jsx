import { useState } from 'react';
import { useStore } from '@/data/store';
import { rainBucket } from '@/lib/model';
import StatusBanner from './StatusBanner';
import { HAZARD_TONE, HAZARD_WORD, delayMinutes, hazardOf, headline, roundDelay, whyFlagged } from './plain';

const TIMES = [
  { m: 0, label: 'Now' },
  { m: 15, label: 'In 15 min' },
  { m: 30, label: 'In 30 min' },
  { m: 60, label: 'In 1 hour' },
];

/** The "what if" corner: time travel and rain sliders live here, not on the main screens. */
export default function Explore() {
  const { rain, setRain, trend, setTrend, statesAt } = useStore();
  const [minutes, setMinutes] = useState(0);
  const states = statesAt(minutes);
  const banner = headline(states);
  const sorted = [...states].sort((a, b) => hazardOf(b) - hazardOf(a) || b.impact - a.impact);

  return (
    <div className="stack cm-page">
      <h1 className="cm-h1">Explore</h1>
      <p className="cm-muted">See how roads could look later, or if the rain gets heavier or lighter. Nothing here changes your saved routes.</p>

      <section className="card pad">
        <div className="field">
          <label className="field-label" htmlFor="cm-rain">Rain: <strong>{rain < 3 ? 'None' : rainBucket(rain)}</strong></label>
          <input id="cm-rain" type="range" min="0" max="100" step="1" value={rain} onChange={(e) => setRain(Number(e.target.value))} aria-valuetext={`${rainBucket(rain)} rain`} />
          <div className="ticks"><span>None</span><span>Light</span><span>Heavy</span><span>Extreme</span></div>
        </div>
        <div className="field" style={{ marginTop: 12 }}>
          <div className="field-label">Rain is</div>
          <div className="seg" role="group" aria-label="Rain trend">
            {['rising', 'steady', 'easing'].map((t) => (
              <button key={t} type="button" aria-pressed={trend === t} onClick={() => setTrend(t)}>{t[0].toUpperCase() + t.slice(1)}</button>
            ))}
          </div>
        </div>
        <div className="field" style={{ marginTop: 12 }}>
          <div className="field-label">When</div>
          <div className="seg" role="group" aria-label="Time">
            {TIMES.map((t) => (
              <button key={t.m} type="button" aria-pressed={minutes === t.m} onClick={() => setMinutes(t.m)}>{t.label}</button>
            ))}
          </div>
        </div>
        <p className="cm-muted" style={{ marginTop: 10 }}>The rain level is sample data in this prototype. Moving it here updates every screen.</p>
      </section>

      <StatusBanner {...banner} />
      <section className="card pad">
        <ul className="cm-list">
          {sorted.map((s) => (
            <li key={s.segmentId}>
              <span className={`cm-dot ${HAZARD_TONE[hazardOf(s)]}`} aria-hidden="true" />
              <div>
                <div className="cm-strong">{s.segment.name}</div>
                <div className="cm-muted">{whyFlagged(s)}{hazardOf(s) ? `. Expect ${roundDelay(delayMinutes(s))} extra.` : '.'}</div>
              </div>
              <span className={`chip ${HAZARD_TONE[hazardOf(s)]}`}>{HAZARD_WORD[hazardOf(s)]}</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
