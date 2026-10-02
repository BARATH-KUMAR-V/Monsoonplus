/**
 * Report a hazard. The Waze idea, and the piece that fills the gap no model can: a
 * person standing in the water knows something the forecast does not.
 *
 * Reports are stored in this browser only. There is no server in this build, so they
 * cannot reach other users -- the page says that plainly rather than implying a
 * community that does not exist yet. They do affect this user's own routing.
 */
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { MapContainer, Marker, TileLayer, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { useApp, REPORT_TYPES, REPORT_TTL_HOURS } from '@/data/consumer';
import { reverseGeocode } from '@/services/geocode';
import StatusBanner from './StatusBanner';

const pin = L.divIcon({
  className: 'trip-pin',
  html: '<svg width="30" height="38" viewBox="0 0 30 38" aria-hidden="true"><path d="M15 37C15 37 28 22.5 28 14A13 13 0 1 0 2 14c0 8.5 13 23 13 23z" fill="#b87d04" stroke="#fff" stroke-width="2.5"/><circle cx="15" cy="14" r="5" fill="#fff"/></svg>',
  iconSize: [30, 38],
  iconAnchor: [15, 37],
});

function Picker({ onPick }) {
  useMapEvents({ click: (e) => onPick({ lat: e.latlng.lat, lon: e.latlng.lng }) });
  return null;
}

export default function Report() {
  const { addReport, focus, located, locate } = useApp();
  const navigate = useNavigate();

  const [type, setType] = useState(null);
  const [point, setPoint] = useState(null);
  const [placeName, setPlaceName] = useState('');
  const [note, setNote] = useState('');
  const [done, setDone] = useState(false);
  const [busy, setBusy] = useState(false);

  const choosePoint = async (coords) => {
    setPoint(coords);
    setPlaceName('');
    const place = await reverseGeocode(coords.lat, coords.lon);
    setPlaceName(place.name);
  };

  const useMyLocation = async () => {
    setBusy(true);
    const place = await locate({ moveFocus: false });
    setBusy(false);
    if (place) {
      setPoint({ lat: place.lat, lon: place.lon });
      setPlaceName(place.name);
    }
  };

  const submit = () => {
    if (!type || !point) return;
    addReport({ type, lat: point.lat, lon: point.lon, placeName, note: note.trim().slice(0, 200) });
    setDone(true);
  };

  if (done) {
    return (
      <div className="stack cm-page">
        <StatusBanner tier="ok" title="Thanks, your report is saved." detail={`It will show on your map and affect your routes for the next ${REPORT_TTL_HOURS} hours.`} />
        <p className="cm-muted">
          Reports are kept in this browser only. This build has no server, so other people will not see it. The map and
          your route suggestions will take it into account straight away.
        </p>
        <div className="inline">
          <button type="button" className="btn cm-primary" onClick={() => navigate('/map')}>See it on the map</button>
          <button type="button" className="btn ghost" onClick={() => { setDone(false); setType(null); setPoint(null); setNote(''); }}>
            Report something else
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="stack cm-page">
      <h1 className="cm-h1">Report a hazard</h1>
      <p className="cm-muted">Tell the app what you can see. It takes two taps.</p>

      <section className="card pad">
        <h2 className="card-title">1. What is it?</h2>
        <div className="report-grid" role="radiogroup" aria-label="Type of hazard">
          {REPORT_TYPES.map((item) => (
            <button
              key={item.id}
              type="button"
              role="radio"
              aria-checked={type === item.id}
              className={`report-opt ${type === item.id ? 'on' : ''} ${item.severity}`}
              onClick={() => setType(item.id)}
            >
              <b>{item.label}</b>
              <i>{item.blurb}</i>
            </button>
          ))}
        </div>
      </section>

      <section className="card pad">
        <h2 className="card-title">2. Where?</h2>
        <div className="inline" style={{ marginBottom: 10 }}>
          <button type="button" className="btn ghost small" onClick={useMyLocation} disabled={busy}>
            {busy || located.state === 'locating' ? 'Finding you…' : 'Use my location'}
          </button>
          <span className="cm-muted">or tap the map</span>
        </div>
        {located.state === 'error' && <p className="cm-muted">{located.error}</p>}

        <div className="map-wrap" style={{ height: 280 }}>
          <MapContainer center={[point?.lat ?? focus.lat, point?.lon ?? focus.lon]} zoom={14} style={{ height: '100%', width: '100%' }}>
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution="&copy; OpenStreetMap contributors" maxZoom={19} />
            <Picker onPick={choosePoint} />
            {point && <Marker position={[point.lat, point.lon]} icon={pin} />}
          </MapContainer>
        </div>
        {point && <p className="cm-muted" style={{ marginTop: 8 }}>Selected: {placeName || `${point.lat.toFixed(4)}, ${point.lon.toFixed(4)}`}</p>}
      </section>

      <section className="card pad">
        <h2 className="card-title">3. Anything to add? (optional)</h2>
        <label className="field-label" htmlFor="report-note">Description</label>
        <textarea
          id="report-note"
          rows={3}
          maxLength={200}
          value={note}
          onChange={(event) => setNote(event.target.value)}
          placeholder="For example: knee-deep near the subway, cars turning back"
        />
        <p className="cm-muted small">{note.length}/200 characters. Please do not include names or vehicle numbers.</p>
      </section>

      <button type="button" className="btn cm-primary" disabled={!type || !point} onClick={submit}>
        {!type ? 'Choose what you are reporting' : !point ? 'Choose where it is' : 'Send report'}
      </button>

      <p className="note">
        Reports expire after {REPORT_TTL_HOURS} hours so the map does not fill up with conditions that have passed. If
        you are somewhere dangerous, move to safety first and report afterwards.
      </p>
    </div>
  );
}
