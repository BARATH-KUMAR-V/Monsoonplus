/**
 * Plan a Trip + Route Results.
 *
 * One page rather than two: a traveller wants to tweak the destination and immediately
 * see what changed, and bouncing between a form page and a results page breaks that.
 */
import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useApp } from '@/data/consumer';
import { useTrip } from './useTrip';
import PlaceInput from './PlaceInput';
import TripMap from './TripMap';
import StatusBanner from './StatusBanner';
import SourceBadge, { SourceLine } from './SourceBadge';
import { WindArrow } from './WeatherIcon';
import { reverseGeocode } from '@/services/geocode';
import { routeVerdict } from '@/engine/score';
import { decisionBreakdown, profileAdvice, routeFactors } from '@/engine/explain';
import { clock } from '@/engine/departure';
import { rainWord } from '@/services/weather';
import { Loading } from '@/components/ui';

const TONE_WORD = { smooth: 'ok', slow: 'warning', disruption: 'warning', jammed: 'critical' };

function Factor({ factor }) {
  return (
    <div className="factor">
      <span className="factor-label">{factor.label}</span>
      <span className={`factor-value t-${factor.tone}`}>{factor.value}</span>
      {factor.note && <span className="factor-note">{factor.note}</span>}
    </div>
  );
}

function RouteCard({ scored, selected, onSelect, isRecommended }) {
  const verdict = routeVerdict(scored);
  const factors = routeFactors(scored);
  const [showWhy, setShowWhy] = useState(false);
  const breakdown = decisionBreakdown(scored);

  return (
    <section className={`route-card ${isRecommended ? 'safe' : ''} ${selected ? 'on' : ''}`}>
      <button type="button" className="route-head" onClick={() => onSelect(scored.route.id)} aria-pressed={selected}>
        <div>
          <h3>
            Route {scored.route.letter}
            {isRecommended && <span className="pill">Recommended</span>}
          </h3>
          <div className="route-time">
            {Math.round(scored.time.minutes)} min
            <span>{scored.route.distanceKm.toFixed(1)} km</span>
          </div>
        </div>
        <span className={`chip ${verdict.tone}`}>{verdict.label}</span>
      </button>

      {scored.route.roads.length > 0 && (
        <p className="route-via">via {scored.route.roads.slice(0, 3).map((r) => r.name).join(', ')}</p>
      )}

      <div className="factors">
        {factors.map((f) => <Factor key={f.key} factor={f} />)}
      </div>

      <button type="button" className="btn ghost small" aria-expanded={showWhy} onClick={() => setShowWhy((v) => !v)}>
        {showWhy ? 'Hide how this was worked out' : 'How was this worked out?'}
      </button>

      {showWhy && (
        <div className="why">
          <p className="cm-muted">
            For a <b>{scored.profile.label.toLowerCase()}</b> trip, these are the things that counted against this route,
            largest first. Your travel mode sets how much each one matters.
          </p>
          <ul className="why-list">
            {breakdown.map((item) => (
              <li key={item.label}>
                <span>{item.label}</span>
                <span className="why-bar" aria-hidden="true">
                  <i style={{ width: `${Math.min(100, item.share * 260)}%` }} />
                </span>
                <span className="why-pct">{Math.round(item.weight * 100)}% weight</span>
              </li>
            ))}
          </ul>
          <SourceLine
            source={scored.flood?.source || 'estimate'}
            basis={scored.flood?.basis}
          />
        </div>
      )}
    </section>
  );
}

export default function Trip() {
  const { profile, profileId, reports, noteRecent, savePlace, home, work } = useApp();
  const navigate = useNavigate();

  const [from, setFrom] = useState(null);
  const [to, setTo] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [arriveByTime, setArriveByTime] = useState('');
  const [copied, setCopied] = useState(false);

  const arriveBy = useMemo(() => {
    if (!arriveByTime) return null;
    const [h, m] = arriveByTime.split(':').map(Number);
    const date = new Date();
    date.setHours(h, m, 0, 0);
    // A time already past means tomorrow.
    if (date < new Date()) date.setDate(date.getDate() + 1);
    return date;
  }, [arriveByTime]);

  const trip = useTrip({ from, to, profileId, reports, arriveBy });

  const selected = trip.routes.find((r) => r.route.id === selectedId) || trip.recommended;
  const advice = trip.plan || trip.departure;
  const tips = selected ? profileAdvice(selected) : [];

  const setPoint = async (which, coords) => {
    const place = await reverseGeocode(coords.lat, coords.lon);
    if (which === 'from') setFrom(place);
    else setTo(place);
    noteRecent(place);
  };

  const onPickPoint = (coords) => setPoint(from ? 'to' : 'from', coords);

  const choose = (setter) => (place) => {
    setter(place);
    setSelectedId(null);
    if (place) noteRecent(place);
  };

  const share = async () => {
    if (!selected) return;
    const text =
      `MonsoonPlus: ${from.name} to ${to.name}\n` +
      `Route ${selected.route.letter}, about ${Math.round(selected.time.minutes)} min (${selected.route.distanceKm.toFixed(1)} km)\n` +
      `Flooding risk: ${selected.flood?.band.label ?? 'unknown'}. Traffic: ${selected.time.band.label}.\n` +
      (advice ? `${advice.headline} ${advice.detail}\n` : '') +
      `Forecast, not an observation. Never drive into standing water.`;
    try {
      if (navigator.share) await navigator.share({ title: 'MonsoonPlus trip', text });
      else {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2500);
      }
    } catch {
      /* the user dismissed the share sheet */
    }
  };

  return (
    <div className="stack cm-page wide">
      <h1 className="cm-h1">Plan a trip</h1>

      <section className="card pad cm-form">
        <PlaceInput label="From" value={from} onChange={choose(setFrom)} autoFocus />
        <PlaceInput label="To" value={to} onChange={choose(setTo)} placeholder="Where are you going?" />

        <div className="quick-row">
          {home && <button type="button" className="btn ghost small" onClick={() => choose(setTo)(home)}>To Home</button>}
          {work && <button type="button" className="btn ghost small" onClick={() => choose(setTo)(work)}>To Work</button>}
          {from && to && (
            <button type="button" className="btn ghost small" onClick={() => { const a = from; setFrom(to); setTo(a); }}>
              Swap
            </button>
          )}
        </div>

        <div className="field" style={{ marginTop: 14 }}>
          <label className="field-label" htmlFor="arrive-by">Need to be there by a certain time? (optional)</label>
          <div className="inline">
            <input id="arrive-by" type="time" value={arriveByTime} onChange={(e) => setArriveByTime(e.target.value)} />
            {arriveByTime && (
              <button type="button" className="btn ghost small" onClick={() => setArriveByTime('')}>Clear</button>
            )}
          </div>
        </div>
      </section>

      <TripMap
        from={from}
        to={to}
        routes={trip.routes}
        selectedId={selected?.route.id}
        onSelectRoute={setSelectedId}
        onPickPoint={onPickPoint}
        onMovePin={setPoint}
        reports={reports}
        floodPoints={(selected?.flood?.points || []).filter((p) => p.flood.score >= 0.4)}
        height={360}
      />

      {!from || !to ? (
        <p className="cm-muted">Choose where you are starting and where you are going. You can search, use your location, or tap the map.</p>
      ) : trip.status === 'loading' ? (
        <Loading label="Finding routes and checking the weather along them" rows={3} />
      ) : trip.status === 'error' ? (
        <>
          <StatusBanner tier="warning" title="We could not plan this trip." detail={trip.error} />
          <button type="button" className="btn" onClick={trip.retry}>Try again</button>
        </>
      ) : trip.recommended ? (
        <>
          <StatusBanner
            tier={TONE_WORD[routeVerdict(trip.recommended).tone] || 'ok'}
            title={trip.explanation.headline}
            detail={trip.explanation.detail}
          />

          {trip.warnings.map((warning) => (
            <p key={warning} className="note">{warning}</p>
          ))}

          {advice && (
            <section className="card pad">
              <h2 className="card-title">{arriveBy ? 'When to leave' : 'Leave now or wait?'}</h2>
              <p className="cm-big">{advice.headline}</p>
              <p className="cm-muted">{advice.detail}</p>
              {advice.options && (
                <div className="depart-strip" role="list">
                  {advice.options.filter((o) => o.offset % 30 === 0).slice(0, 5).map((option) => (
                    <div key={option.offset} role="listitem" className={`depart ${option === advice.best ? 'best' : ''}`}>
                      <b>{option.offset === 0 ? 'Now' : clock(option.departAt)}</b>
                      <span>{Math.round(option.minutes)} min</span>
                      <i>{option.rainPeak > 0.2 ? rainWord(option.rainPeak) : 'dry'}</i>
                    </div>
                  ))}
                </div>
              )}
              <SourceLine source="estimate" basis="typical traffic for the time of day, plus the official rain forecast" />
            </section>
          )}

          <div className="stack">
            <RouteCard
              scored={trip.recommended}
              isRecommended
              selected={selected?.route.id === trip.recommended.route.id}
              onSelect={setSelectedId}
            />
            {trip.alternatives.map((alt) => (
              <RouteCard
                key={alt.route.id}
                scored={alt}
                selected={selected?.route.id === alt.route.id}
                onSelect={setSelectedId}
              />
            ))}
          </div>

          {tips.length > 0 && (
            <section className="card pad tips">
              <h2 className="card-title">Before you go, as a {profile.label.toLowerCase()} traveller</h2>
              <ul className="cm-bullets">
                {tips.map((tip) => <li key={tip}>{tip}</li>)}
              </ul>
            </section>
          )}

          {selected?.samples?.length > 0 && (
            <section className="card pad">
              <h2 className="card-title">Rain while you are on Route {selected.route.letter}</h2>
              <div className="along" role="list">
                {selected.samples.map((sample, index) => (
                  <div key={index} role="listitem" className="along-step">
                    <b>{index === 0 ? 'Start' : index === selected.samples.length - 1 ? 'End' : `${Math.round(sample.fraction * 100)}%`}</b>
                    <span>{(selected.exposure?.perPoint?.[index] ?? 0) > 0.05 ? `${selected.exposure.perPoint[index].toFixed(1)} mm/h` : 'dry'}</span>
                    <i>
                      {sample.wind?.speed != null ? `${Math.round(sample.wind.speed)} km/h ` : ''}
                      {sample.wind?.direction != null && <WindArrow degrees={sample.wind.direction} size={12} />}
                    </i>
                  </div>
                ))}
              </div>
              <SourceLine source="official" basis="Open-Meteo forecast at points along the route" at={new Date().toISOString()} />
            </section>
          )}

          <div className="inline wrap">
            <button type="button" className="btn cm-primary" onClick={share}>
              {copied ? 'Copied to clipboard' : 'Share this trip'}
            </button>
            <button type="button" className="btn ghost" onClick={() => { savePlace(to, 'custom'); navigate('/places'); }}>
              Save destination
            </button>
            <button type="button" className="btn ghost" onClick={() => navigate('/report')}>
              Report a hazard here
            </button>
          </div>

          <p className="note">
            Travel times and flooding risk are forecasts, not observations of the road. MonsoonPlus cannot see what is
            actually in front of you. If you reach standing water, turn back.
          </p>
        </>
      ) : null}
    </div>
  );
}
