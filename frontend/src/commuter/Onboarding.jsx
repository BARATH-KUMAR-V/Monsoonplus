/**
 * First launch. One question, because that is all we need to make the rest of the app
 * fit the person: the answer sets the weights in the route score, which page they land
 * on, and which warnings get priority.
 */
import { useState } from 'react';
import { PROFILES } from '@/engine/profiles';
import { useApp } from '@/data/consumer';

function Glyph({ name }) {
  const p = { width: 28, height: 28, viewBox: '0 0 24 24', fill: 'none', 'aria-hidden': true };
  const s = { stroke: 'currentColor', strokeWidth: 1.6, strokeLinecap: 'round', strokeLinejoin: 'round' };
  switch (name) {
    case 'car': return <svg {...p}><path d="M3 13.5 4.8 8A2 2 0 0 1 6.7 6.6h10.6A2 2 0 0 1 19.2 8L21 13.5" {...s} /><rect x="2.5" y="13.5" width="19" height="5.5" rx="1.6" {...s} /><circle cx="7" cy="19" r="1.4" {...s} /><circle cx="17" cy="19" r="1.4" {...s} /></svg>;
    case 'bike': return <svg {...p}><circle cx="5.5" cy="16.5" r="3.5" {...s} /><circle cx="18.5" cy="16.5" r="3.5" {...s} /><path d="M5.5 16.5h5l4-8h-2M14.5 8.5h3l1.5 8M9 8.5h4" {...s} /></svg>;
    case 'walk': return <svg {...p}><circle cx="13" cy="4.2" r="1.8" {...s} /><path d="M11 21l1.5-5.5-2.5-2.5 1-4.5 3 2 2.5 1M10 10.5 7.5 13M13.5 15.5 16 21" {...s} /></svg>;
    case 'school': return <svg {...p}><path d="M2.5 8.5 12 4l9.5 4.5L12 13z" {...s} /><path d="M6 10.5v5c0 1.4 2.7 2.5 6 2.5s6-1.1 6-2.5v-5" {...s} /></svg>;
    case 'family': return <svg {...p}><circle cx="8" cy="7" r="2.6" {...s} /><circle cx="16.5" cy="9" r="2" {...s} /><path d="M3.5 20v-3a4.5 4.5 0 0 1 9 0v3M14.5 20v-2.4a3.5 3.5 0 0 1 6-2.4" {...s} /></svg>;
    case 'delivery': return <svg {...p}><rect x="2.5" y="7" width="11" height="9" rx="1.4" {...s} /><path d="M13.5 10h3.8l3.2 3.3V16h-7z" {...s} /><circle cx="6.5" cy="18" r="1.6" {...s} /><circle cx="17" cy="18" r="1.6" {...s} /></svg>;
    default: return <svg {...p}><path d="M6.5 17a4.2 4.2 0 0 1 .5-8.4 5.4 5.4 0 0 1 10.2 1.2A3.8 3.8 0 0 1 17 17z" {...s} /></svg>;
  }
}

export default function Onboarding() {
  const { setProfileId, setOnboarded, profileId } = useApp();
  const [picked, setPicked] = useState(profileId);

  return (
    <div className="onboard">
      <div className="onboard-card">
        <div className="onboard-brand">
          <span className="brand-mark" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 16 16" fill="none">
              <path d="M4 10c0-2.6 4-7 4-7s4 4.4 4 7a4 4 0 1 1-8 0z" fill="currentColor" />
            </svg>
          </span>
          <div>
            <h1>MonsoonPlus</h1>
            <p>Weather and safer routes for Chennai</p>
          </div>
        </div>

        <h2>What will you use it for?</h2>
        <p className="cm-muted">This changes what we warn you about and how we pick routes. You can change it any time.</p>

        <div className="onboard-grid" role="radiogroup" aria-label="How you travel">
          {PROFILES.map((profile) => (
            <button
              key={profile.id}
              type="button"
              role="radio"
              aria-checked={picked === profile.id}
              className={`onboard-opt ${picked === profile.id ? 'on' : ''}`}
              onClick={() => setPicked(profile.id)}
            >
              <span className="onboard-ic"><Glyph name={profile.icon} /></span>
              <b>{profile.label}</b>
              <i>{profile.blurb}</i>
            </button>
          ))}
        </div>

        <button
          type="button"
          className="btn cm-primary onboard-go"
          onClick={() => { setProfileId(picked); setOnboarded(true); }}
        >
          Continue
        </button>

        <p className="onboard-note">
          MonsoonPlus is a student project and a planning aid. It forecasts conditions; it cannot see the road in front
          of you. Never use it to decide whether to cross flood water.
        </p>
      </div>
    </div>
  );
}
