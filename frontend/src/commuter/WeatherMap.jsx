/**
 * The weather and hazard map.
 *
 * Open-Meteo serves point forecasts, not map tiles, so the layers here are built by
 * sampling a grid of real forecast points across the visible area and shading the
 * cells. That means what you see is actual forecast data at roughly 7 km spacing --
 * coarser than a radar image, and the legend says so, but every cell is a real number
 * rather than a decorative gradient.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import L from 'leaflet';
import { MapContainer, Marker, Popup, Rectangle, TileLayer, Tooltip, useMap, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { useApp, REPORT_BY_ID } from '@/data/consumer';
import { fetchWeatherAtPoints, compass, describeCode } from '@/services/weather';
import { estimatePointFlood, floodBand } from '@/engine/flood';
import { SourceLine } from './SourceBadge';
import { WindArrow } from './WeatherIcon';

const LAYERS = [
  { id: 'rain', label: 'Rain', unit: 'mm/h' },
  { id: 'flood', label: 'Flood risk', unit: '' },
  { id: 'wind', label: 'Wind', unit: 'km/h' },
  { id: 'temperature', label: 'Temperature', unit: '°C' },
  { id: 'clouds', label: 'Cloud', unit: '%' },
  { id: 'hazards', label: 'Reports only', unit: '' },
];

const SCALES = {
  // Sequential, single hue, light to dark -- magnitude, so never a rainbow.
  rain: [
    { max: 0.1, color: null, label: 'none' },
    { max: 2.5, color: '#cfe0ec', label: 'light' },
    { max: 7.6, color: '#8fb6d4', label: 'moderate' },
    { max: 25, color: '#3f7fae', label: 'heavy' },
    { max: Infinity, color: '#1b4578', label: 'very heavy' },
  ],
  wind: [
    { max: 15, color: null, label: 'calm' },
    { max: 30, color: '#dfe6e2', label: 'breezy' },
    { max: 45, color: '#a9c0b4', label: 'windy' },
    { max: 60, color: '#6e9381', label: 'strong' },
    { max: Infinity, color: '#2f5a46', label: 'very strong' },
  ],
  temperature: [
    { max: 24, color: '#cfe0ec', label: 'under 24' },
    { max: 28, color: '#e7e2d2', label: '24-28' },
    { max: 32, color: '#f0d9b4', label: '28-32' },
    { max: 36, color: '#e8b184', label: '32-36' },
    { max: Infinity, color: '#cf7a52', label: 'over 36' },
  ],
  clouds: [
    { max: 20, color: null, label: 'clear' },
    { max: 50, color: '#eae6dc', label: 'some cloud' },
    { max: 80, color: '#cdc8ba', label: 'cloudy' },
    { max: Infinity, color: '#a9a396', label: 'overcast' },
  ],
  flood: [
    { max: 0.18, color: null, label: 'low' },
    { max: 0.4, color: '#e9d9b0', label: 'watch' },
    { max: 0.65, color: '#dba36f', label: 'moderate' },
    { max: Infinity, color: '#b4584a', label: 'high' },
  ],
};

const shade = (layer, value) => (SCALES[layer] || []).find((s) => value < s.max)?.color ?? null;

const hazardIcon = (type) => {
  const severe = REPORT_BY_ID[type]?.severity === 'severe';
  return L.divIcon({
    className: 'hazard-pin',
    html:
      `<svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">` +
      `<path d="M13 1.5 25 23H1z" fill="${severe ? '#a3231b' : '#b87d04'}" stroke="#fff" stroke-width="2"/>` +
      `<path d="M13 9v6M13 18h.01" stroke="#fff" stroke-width="2.2" stroke-linecap="round"/></svg>`,
    iconSize: [26, 26],
    iconAnchor: [13, 20],
  });
};

/** Reports the visible bounds upward, debounced, so we only refetch when it matters. */
function BoundsWatcher({ onChange }) {
  const map = useMap();
  const timer = useRef(null);
  const emit = useCallback(() => {
    clearTimeout(timer.current);
    timer.current = setTimeout(() => {
      const b = map.getBounds();
      onChange({ south: b.getSouth(), west: b.getWest(), north: b.getNorth(), east: b.getEast(), zoom: map.getZoom() });
    }, 600);
  }, [map, onChange]);

  useMapEvents({ moveend: emit, zoomend: emit });
  useEffect(() => { emit(); return () => clearTimeout(timer.current); }, [emit]);
  return null;
}

const GRID = 7; // 49 points, inside Open-Meteo's multi-coordinate limit

export default function WeatherMap() {
  const { focus, reports, weather } = useApp();
  const [layer, setLayer] = useState('rain');
  const [hour, setHour] = useState(0);
  const [bounds, setBounds] = useState(null);
  const [cells, setCells] = useState({ state: 'idle', data: [], error: null, at: null });
  const runId = useRef(0);

  useEffect(() => {
    if (!bounds || layer === 'hazards') return;
    const id = ++runId.current;
    const latStep = (bounds.north - bounds.south) / GRID;
    const lonStep = (bounds.east - bounds.west) / GRID;
    const points = [];
    for (let i = 0; i < GRID; i += 1) {
      for (let j = 0; j < GRID; j += 1) {
        points.push({
          lat: bounds.south + latStep * (i + 0.5),
          lon: bounds.west + lonStep * (j + 0.5),
          box: [
            [bounds.south + latStep * i, bounds.west + lonStep * j],
            [bounds.south + latStep * (i + 1), bounds.west + lonStep * (j + 1)],
          ],
        });
      }
    }

    setCells((prev) => ({ ...prev, state: 'loading' }));
    fetchWeatherAtPoints(points)
      .then((result) => {
        if (id !== runId.current) return;
        setCells({ state: 'ready', data: result, error: null, at: new Date().toISOString() });
      })
      .catch((error) => {
        if (id !== runId.current) return;
        setCells({ state: 'error', data: [], error: error.message, at: null });
      });
  }, [bounds, layer]);

  /** Pull the value for the chosen hour out of each cell. */
  const shaded = useMemo(() => {
    if (layer === 'hazards') return [];
    const target = Date.now() + hour * 3600000;
    return cells.data
      .map((cell) => {
        const rows = cell.hourly || [];
        if (!rows.length) return null;
        const row = rows.reduce((best, r) => (Math.abs(r.date - target) < Math.abs(best.date - target) ? r : best), rows[0]);
        const index = rows.indexOf(row);

        let value;
        if (layer === 'rain') value = row.precipitation ?? 0;
        else if (layer === 'wind') value = row.wind_speed_10m ?? 0;
        else if (layer === 'temperature') value = row.temperature_2m ?? 0;
        else if (layer === 'clouds') value = row.cloud_cover ?? 0;
        else if (layer === 'flood') {
          const recent = rows.slice(Math.max(0, index - 6), index).reduce((s, r) => s + (r.precipitation ?? 0), 0);
          const next3 = Math.max(0, ...rows.slice(index, index + 3).map((r) => r.precipitation ?? 0));
          value = estimatePointFlood(
            cell,
            { nowMm: row.precipitation ?? 0, next3hMm: next3, recent6hMm: recent },
            null,
            reports,
          ).score;
        }
        return { ...cell, row, value, color: shade(layer, value) };
      })
      .filter(Boolean);
  }, [cells.data, layer, hour, reports]);

  const legend = SCALES[layer] || [];
  const active = LAYERS.find((l) => l.id === layer);

  return (
    <div className="stack cm-page wide">
      <h1 className="cm-h1">Map</h1>

      <div className="seg wrap" role="group" aria-label="What to show on the map">
        {LAYERS.map((item) => (
          <button key={item.id} type="button" aria-pressed={layer === item.id} onClick={() => setLayer(item.id)}>
            {item.label}
          </button>
        ))}
      </div>

      {layer !== 'hazards' && (
        <div className="field">
          <label className="field-label" htmlFor="map-hour">
            {hour === 0 ? 'Now' : `In ${hour} hour${hour > 1 ? 's' : ''}`}
          </label>
          <input id="map-hour" type="range" min="0" max="24" step="1" value={hour} onChange={(e) => setHour(Number(e.target.value))} />
          <div className="ticks"><span>Now</span><span>+6 h</span><span>+12 h</span><span>+18 h</span><span>+24 h</span></div>
        </div>
      )}

      <div className="map-wrap" style={{ height: 460 }}>
        <MapContainer center={[focus.lat, focus.lon]} zoom={11} scrollWheelZoom style={{ height: '100%', width: '100%' }}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution="&copy; OpenStreetMap contributors" maxZoom={19} />
          <BoundsWatcher onChange={setBounds} />

          {shaded.filter((cell) => cell.color).map((cell, index) => (
            <Rectangle
              key={index}
              bounds={cell.box}
              pathOptions={{ color: cell.color, weight: 0, fillColor: cell.color, fillOpacity: 0.5 }}
            >
              <Tooltip sticky>
                {layer === 'flood'
                  ? `${floodBand(cell.value).label} flooding risk`
                  : `${cell.value.toFixed(layer === 'rain' ? 1 : 0)} ${active.unit}`}
              </Tooltip>
            </Rectangle>
          ))}

          {layer === 'wind' && shaded.filter((_, i) => i % 2 === 0).map((cell, index) => (
            <Marker
              key={`w${index}`}
              position={[cell.lat, cell.lon]}
              icon={L.divIcon({
                className: 'wind-pin',
                html: `<div style="transform:rotate(${(cell.row.wind_direction_10m ?? 0) + 180}deg)"><svg width="16" height="16" viewBox="0 0 24 24"><path d="M12 3 L18 21 L12 17 L6 21 Z" fill="#2f5a46"/></svg></div>`,
                iconSize: [16, 16],
                iconAnchor: [8, 8],
              })}
            >
              <Tooltip direction="top">
                {Math.round(cell.row.wind_speed_10m)} km/h from the {compass(cell.row.wind_direction_10m)}
              </Tooltip>
            </Marker>
          ))}

          {reports.map((report) => (
            <Marker key={report.id} position={[report.lat, report.lon]} icon={hazardIcon(report.type)}>
              <Popup>
                <strong>{report.label}</strong>
                <div className="small muted" style={{ marginTop: 4 }}>
                  {report.note || 'No description given.'}<br />
                  {report.placeName ? `${report.placeName} · ` : ''}
                  reported {new Date(report.at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}
                </div>
                <div className="small muted" style={{ marginTop: 6 }}>Community report, not verified.</div>
              </Popup>
            </Marker>
          ))}
        </MapContainer>

        {layer !== 'hazards' && (
          <div className="map-overlay bl">
            {legend.filter((s) => s.color).map((s) => (
              <span className="legend-item" key={s.label}>
                <span className="legend-swatch" style={{ background: s.color, height: 10, borderRadius: 2 }} aria-hidden="true" />
                {s.label}
              </span>
            ))}
          </div>
        )}

        {cells.state === 'loading' && <div className="map-overlay tl">Loading forecast for this area…</div>}
        {cells.state === 'error' && <div className="map-overlay tl">Forecast data unavailable here. {cells.error}</div>}
      </div>

      {layer === 'flood' ? (
        <SourceLine source="estimate" basis="rainfall forecast and past flooding patterns, at roughly 7 km spacing. Ground height is not used at this zoom." at={cells.at} />
      ) : layer === 'hazards' ? (
        <SourceLine source="community" basis={`${reports.length} active report${reports.length === 1 ? '' : 's'} from this browser`} />
      ) : (
        <SourceLine source="official" basis="Open-Meteo forecast sampled on a grid across the visible area, roughly 7 km spacing" at={cells.at} />
      )}

      <p className="note">
        These layers are built from point forecasts on a grid, not from weather radar. They show the broad pattern well
        and small local detail poorly. Move or zoom the map and the grid is recalculated for the new area.
      </p>
    </div>
  );
}
