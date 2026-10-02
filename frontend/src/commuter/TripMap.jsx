/**
 * The trip map: route lines, draggable start and end pins, tap-to-drop, hazard markers.
 *
 * Accessibility note carried over from the rest of the app: a Leaflet map is not
 * usefully keyboard-navigable, so every screen that shows one also lists the same
 * information as text. The map is the pleasant way to read it, never the only way.
 */
import { useEffect, useMemo, useState } from 'react';
import L from 'leaflet';
import { CircleMarker, MapContainer, Marker, Polyline, Popup, TileLayer, Tooltip, useMap, useMapEvents } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { boundsOf } from '@/services/routing';
import { REPORT_BY_ID } from '@/data/consumer';

const ROUTE_COLORS = {
  recommended: '#1f5c8c',
  alternative: '#7a8780',
  risky: '#a3231b',
};

function pinIcon(kind) {
  const fill = kind === 'from' ? '#1f5c47' : '#a3231b';
  return L.divIcon({
    className: 'trip-pin',
    html:
      `<svg width="30" height="38" viewBox="0 0 30 38" aria-hidden="true">` +
      `<path d="M15 37C15 37 28 22.5 28 14A13 13 0 1 0 2 14c0 8.5 13 23 13 23z" fill="${fill}" stroke="#fff" stroke-width="2.5"/>` +
      `<circle cx="15" cy="14" r="5" fill="#fff"/></svg>`,
    iconSize: [30, 38],
    iconAnchor: [15, 37],
    popupAnchor: [0, -32],
  });
}

const hazardIcon = (type) => {
  const severe = REPORT_BY_ID[type]?.severity === 'severe';
  const bg = severe ? '#a3231b' : '#b87d04';
  return L.divIcon({
    className: 'hazard-pin',
    html:
      `<svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">` +
      `<path d="M13 1.5 25 23H1z" fill="${bg}" stroke="#fff" stroke-width="2"/>` +
      `<path d="M13 9v6M13 18h.01" stroke="#fff" stroke-width="2.2" stroke-linecap="round"/></svg>`,
    iconSize: [26, 26],
    iconAnchor: [13, 20],
  });
};

/** Fits the view whenever the routes change, but leaves the user alone after that. */
function FitBounds({ bounds }) {
  const map = useMap();
  useEffect(() => {
    if (!bounds) return;
    map.fitBounds(bounds, { padding: [44, 44], maxZoom: 14 });
  }, [map, JSON.stringify(bounds)]); // eslint-disable-line react-hooks/exhaustive-deps
  return null;
}

function ClickToSet({ onPick }) {
  useMapEvents({
    click(event) {
      onPick?.({ lat: event.latlng.lat, lon: event.latlng.lng });
    },
  });
  return null;
}

function TileWatcher({ onFail }) {
  const map = useMap();
  useEffect(() => {
    let ok = false;
    const mark = () => { ok = true; };
    map.on('tileload', mark);
    const timer = setTimeout(() => !ok && onFail(), 7000);
    return () => { map.off('tileload', mark); clearTimeout(timer); };
  }, [map, onFail]);
  return null;
}

export default function TripMap({
  from,
  to,
  routes = [],
  selectedId,
  onSelectRoute,
  onPickPoint,
  onMovePin,
  reports = [],
  floodPoints = [],
  height = 380,
  center = [13.0827, 80.2707],
  zoom = 11,
}) {
  const [tilesFailed, setTilesFailed] = useState(false);

  // Both the routes AND the pins: fitting to the lines alone can push a pin off screen.
  const bounds = useMemo(() => {
    const all = [
      ...routes.flatMap((r) => r.route.coordinates),
      ...[from, to].filter(Boolean).map((p) => [p.lat, p.lon]),
    ];
    return all.length >= 2 ? boundsOf(all) : null;
  }, [routes, from, to]);

  return (
    <div className="map-wrap trip-map" style={{ height }}>
      <MapContainer center={center} zoom={zoom} scrollWheelZoom style={{ height: '100%', width: '100%' }}>
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; OpenStreetMap contributors"
          maxZoom={19}
        />
        <TileWatcher onFail={() => setTilesFailed(true)} />
        <FitBounds bounds={bounds} />
        {onPickPoint && <ClickToSet onPick={onPickPoint} />}

        {/* Draw non-selected routes first so the chosen one sits on top. */}
        {[...routes].sort((a, b) => (a.route.id === selectedId ? 1 : -1)).map((scored) => {
          const isSelected = scored.route.id === selectedId;
          const risky = (scored.flood?.score ?? 0) >= 0.5;
          const color = isSelected ? (risky ? ROUTE_COLORS.risky : ROUTE_COLORS.recommended) : ROUTE_COLORS.alternative;
          return (
            <Polyline
              key={scored.route.id}
              positions={scored.route.coordinates}
              pathOptions={{
                color,
                weight: isSelected ? 7 : 5,
                opacity: isSelected ? 0.95 : 0.55,
                dashArray: risky && isSelected ? '14 7' : null,
                lineCap: 'round',
              }}
              eventHandlers={{ click: () => onSelectRoute?.(scored.route.id) }}
            >
              <Tooltip sticky>
                Route {scored.route.letter} · {Math.round(scored.time.minutes)} min
                {scored.recommended ? ' · recommended' : ''}
              </Tooltip>
            </Polyline>
          );
        })}

        {/* Where water is expected along the selected route. */}
        {floodPoints.map((point, index) => (
          <CircleMarker
            key={`fp${index}`}
            center={[point.lat, point.lon]}
            radius={9}
            pathOptions={{ color: '#1f5c8c', weight: 2, fillColor: '#1f5c8c', fillOpacity: 0.35 }}
          >
            <Tooltip direction="top">{point.flood.band.label} flooding risk</Tooltip>
            <Popup>
              <strong>{point.flood.band.label} flooding risk</strong>
              <div className="small muted" style={{ marginTop: 4 }}>
                {point.flood.reasons.length ? point.flood.reasons.join(', ') : 'Estimated from rainfall and terrain.'}
              </div>
              <div className="small muted" style={{ marginTop: 6 }}>MonsoonPlus estimate, not a measurement.</div>
            </Popup>
          </CircleMarker>
        ))}

        {reports.map((report) => (
          <Marker key={report.id} position={[report.lat, report.lon]} icon={hazardIcon(report.type)}>
            <Popup>
              <strong>{report.label}</strong>
              <div className="small muted" style={{ marginTop: 4 }}>
                {report.note || 'No description given.'}
                <br />
                Reported {new Date(report.at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}
                {report.placeName ? ` near ${report.placeName}` : ''}
              </div>
              <div className="small muted" style={{ marginTop: 6 }}>Community report. Not verified.</div>
            </Popup>
          </Marker>
        ))}

        {from && (
          <Marker
            position={[from.lat, from.lon]}
            icon={pinIcon('from')}
            draggable={Boolean(onMovePin)}
            eventHandlers={{ dragend: (e) => onMovePin?.('from', { lat: e.target.getLatLng().lat, lon: e.target.getLatLng().lng }) }}
          >
            <Tooltip direction="top">Start: {from.name}</Tooltip>
          </Marker>
        )}
        {to && (
          <Marker
            position={[to.lat, to.lon]}
            icon={pinIcon('to')}
            draggable={Boolean(onMovePin)}
            eventHandlers={{ dragend: (e) => onMovePin?.('to', { lat: e.target.getLatLng().lat, lon: e.target.getLatLng().lng }) }}
          >
            <Tooltip direction="top">Destination: {to.name}</Tooltip>
          </Marker>
        )}
      </MapContainer>

      {tilesFailed && <div className="map-overlay tl">Map images could not load. Routes are still shown.</div>}
      {onPickPoint && !from && <div className="map-overlay tl">Tap the map to set your starting point</div>}
      {onPickPoint && from && !to && <div className="map-overlay tl">Tap the map to set your destination</div>}
    </div>
  );
}
