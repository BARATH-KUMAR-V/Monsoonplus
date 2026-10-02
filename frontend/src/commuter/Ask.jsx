/**
 * Ask MonsoonPlus.
 *
 * The answers are composed from facts the app has already computed -- the official
 * forecast, the trip scoring, the flood estimate, the reports. The assistant does not
 * forecast anything itself, and when it lacks a fact it says so instead of filling the
 * gap. See engine/assistant.js for why that boundary is drawn where it is.
 */
import { useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { useApp } from '@/data/consumer';
import { answer, SUGGESTED } from '@/engine/assistant';
import SourceBadge from './SourceBadge';

export default function Ask() {
  const app = useApp();
  const [question, setQuestion] = useState('');
  const [thread, setThread] = useState([]);
  const endRef = useRef(null);

  const ask = (text) => {
    const q = String(text || '').trim();
    if (!q) return;
    const facts = {
      weather: app.weather.data,
      air: app.air.data,
      place: app.focus,
      profile: app.profile,
      reports: app.reports,
      trip: null, // the trip lives on the Trip page; the assistant says so when asked
    };
    const result = answer(q, facts);
    setThread((prev) => [...prev, { q, ...result, at: Date.now() }]);
    setQuestion('');
    setTimeout(() => endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }), 50);
  };

  return (
    <div className="stack cm-page">
      <h1 className="cm-h1">Ask MonsoonPlus</h1>
      <p className="cm-muted">
        Ask about the weather, your trip, or whether it is a good time to go. Answers come from the forecast and your
        settings, and the app says when it does not know something.
      </p>

      {thread.length === 0 && (
        <section className="card pad">
          <h2 className="card-title">Try asking</h2>
          <div className="quick-row">
            {SUGGESTED.map((s) => (
              <button key={s} type="button" className="btn ghost small" onClick={() => ask(s)}>{s}</button>
            ))}
          </div>
        </section>
      )}

      <div className="thread">
        {thread.map((entry) => (
          <div key={entry.at} className="qa">
            <p className="qa-q">{entry.q}</p>
            <div className="qa-a">
              <p className="cm-big">{entry.answer}</p>
              {entry.points.length > 0 && (
                <ul className="qa-points">
                  {entry.points.map((point, i) => (
                    <li key={i}>
                      <span>{point.text}</span>
                      <SourceBadge source={point.source} />
                    </li>
                  ))}
                </ul>
              )}
              {entry.missing.includes('trip') && (
                <p className="cm-muted"><Link to="/trip">Plan a trip</Link> and ask again for a route-specific answer.</p>
              )}
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <form
        className="ask-bar"
        onSubmit={(event) => { event.preventDefault(); ask(question); }}
      >
        <label className="visually-hidden" htmlFor="ask-input">Your question</label>
        <input
          id="ask-input"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask about the weather or your trip"
        />
        <button type="submit" className="btn cm-primary" disabled={!question.trim()}>Ask</button>
      </form>

      <p className="note">
        This assistant explains numbers the app already has. It is not a language model and it does not make its own
        weather predictions. If it does not have the information, it will tell you.
      </p>
    </div>
  );
}
