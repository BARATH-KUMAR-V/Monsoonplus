import { useState } from 'react';
import { useStore } from '@/data/store';
import NetworkMap from '@/components/NetworkMap';
import StatusBanner from './StatusBanner';
import { HAZARD_TONE, HAZARD_WORD, delayMinutes, hazardOf, headline, roundDelay, whyFlagged } from './plain';

export default function HazardMap() {
  const { states } = useStore();
  const [selected, setSelected] = useState(null);
  const banner = headline(states);
  const sorted = [...states].sort((a, b) => hazardOf(b) - hazardOf(a) || b.impact - a.impact);

  return (
    <div className="stack cm-page">
      <StatusBanner {...banner} />
      <NetworkMap states={states} layer="hazard" height={420} onSelect={setSelected} selectedId={selected} />
      <section className="card pad">
        <h2 className="card-title">All roads</h2>
        <ul className="cm-list">
          {sorted.map((s) => (
            <li key={s.segmentId} className={selected === s.segmentId ? 'sel' : ''}>
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
