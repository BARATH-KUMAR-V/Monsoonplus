/**
 * The Leaflet map. One component serves every page; pages pass a `layer` and the
 * segment states to draw.
 *
 * Accessibility: a Leaflet map is not keyboard-navigable in any useful way, so every
 * map on the site is accompanied by the same information as a list or table. The map
 * is an enhancement, never the only route to the data.
 *
 * Offline: OpenStreetMap tiles are the single external request the app makes. When
 * they fail the map still renders roads, junctions and overlays over a plain
 * background, and a notice says tiles are unavailable.
 */
import { useEffect, useMemo, useState } from 'react';
import L from 'leaflet';
import { CircleMarker, MapContainer, Marker, Polygon, Polyline, Popup, TileLayer, Tooltip, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { CITY, JUNCTIONS, SEGMENT_BY_ID, latlon } from '@/config/network';
import { STATUS_COLOR, STATUS_META } from '@/components/ui';
import { kmh, mm } from '@/lib/format';

/** Commuter hazard layer: one colour per road, flood-prone roads get a wave marker. */
const hazardIndex = (state) =>
  state.flood.index === 2 || state.level.index >= 3 ? 2 : state.flood.index === 1 || state.level.index >= 1 ? 1 : 0;
const HAZARD_STYLE = [
  { color: 'var(--smooth)', dash: null, label: 'Clear' },
  { color: 'var(--slow)', dash: '10 6', label: 'Caution' },
  { color: 'var(--jammed)', dash: '14 5 3 5', label: 'Avoid' },
];

const WAVE_ICON = L.divIcon({
  className: 'flood-pin',
  html: '<svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true"><circle cx="13" cy="13" r="12" fill="#1f5c8c" stroke="#fff" stroke-width="2"/><path d="M5 12.5c2-2 3.5-2 5.5 0s3.5 2 5.5 0 3.5-2 5 0M5 17c2-2 3.5-2 5.5 0s3.5 2 5.5 0 3.5-2 5 0" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/></svg>',
  iconSize: [26, 26],
  iconAnchor: [13, 13],
});

function FloodMarkers({ states }) {
  return states
    .filter((state) => state.flood.index >= 1)
    .map((state) => {
      const a = latlon(state.segment.a);
      const b = latlon(state.segment.b);
      return (
        <Marker key={`flood-${state.segmentId}`} position={[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]} icon={WAVE_ICON} keyboard={false}>
          <Tooltip direction="top">{state.flood.index === 2 ? 'Waterlogging likely' : 'Water may collect'}</Tooltip>
        </Marker>
      );
    });
}

export const LAYERS = [
  { id: 'traffic', label: 'Traffic' },
  { id: 'rain', label: 'Rain' },
  { id: 'flood', label: 'Flood' },
  { id: 'prediction', label: 'Prediction' },
  { id: 'combined', label: 'Combined' },
];

const RAIN_SCALE = [
  { max: 3, color: '#cfe3ee', label: 'Dry' },
  { max: 15, color: '#9cc6e2', label: 'Light' },
  { max: 40, color: '#5b96c9', label: 'Moderate' },
  { max: 70, color: '#2d6ba8', label: 'Heavy' },
  { max: Infinity, color: '#1b4578', label: 'Extreme' },
];

const FLOOD_SCALE = [
  { color: '#7f9a8c', label: 'Low' },
  { color: 'var(--slow)', label: 'Moderate' },
  { color: 'var(--jammed)', label: 'High' },
];

const IMPACT_SCALE = [
  { max: 20, color: 'var(--smooth)', label: '0-19' },
  { max: 40, color: 'var(--slow)', label: '20-39' },
  { max: 60, color: 'var(--disruption)', label: '40-59' },
  { max: Infinity, color: 'var(--jammed)', label: '60+' },
];

const rainColor = (value) => RAIN_SCALE.find((s) => value < s.max).color;
const impactColor = (value) => IMPACT_SCALE.find((s) => value < s.max).color;

/** Prediction layer: colour by how fast the road is deteriorating over the hour. */
function trendColor(state) {
  const delta = state.speedIn60 - state.speedNow;
  const relative = delta / Math.max(state.segment.free_flow_kmh, 1);
  if (relative < -0.18) return { color: 'var(--jammed)', label: 'Worsening fast' };
  if (relative < -0.05) return { color: 'var(--disruption)', label: 'Worsening' };
  if (relative > 0.05) return { color: 'var(--smooth)', label: 'Recovering' };
  return { color: '#7f9a8c', label: 'Steady' };
}

function styleFor(state, layer) {
  switch (layer) {
    case 'hazard': {
      const h = HAZARD_STYLE[hazardIndex(state)];
      return { color: h.color, dashArray: h.dash, label: h.label };
    }
    case 'rain':
      return { color: rainColor(state.rain), dashArray: null, label: `${mm(state.rain)}` };
    case 'flood':
      return {
        color: FLOOD_SCALE[state.flood.index].color,
        dashArray: STATUS_META[state.flood.key]?.dash ?? null,
        label: `Flood ${state.flood.label}`,
      };
    case 'prediction': {
      const trend = trendColor(state);
      return { color: trend.color, dashArray: null, label: trend.label };
    }
    case 'combined':
      return {
        color: impactColor(state.impact),
        dashArray: null,
        label: `Impact ${state.impact}`,
      };
    case 'traffic':
    default:
      return {
        color: STATUS_COLOR[state.level.key],
        // Dash pattern is the non-colour channel for traffic status.
        dashArray: STATUS_META[state.level.key]?.dash ?? null,
        label: state.level.label,
      };
  }
}

function legendFor(layer) {
  switch (layer) {
    case 'hazard':
      return HAZARD_STYLE.map((h) => ({ color: h.color, label: h.label, dash: h.dash }));
    case 'rain':
      return RAIN_SCALE.map((s) => ({ color: s.color, label: s.label }));
    case 'flood':
      return FLOOD_SCALE;
    case 'prediction':
      return [
        { color: 'var(--jammed)', label: 'Worsening fast' },
        { color: 'var(--disruption)', label: 'Worsening' },
        { color: '#7f9a8c', label: 'Steady' },
        { color: 'var(--smooth)', label: 'Recovering' },
      ];
    case 'combined':
      return IMPACT_SCALE.map((s) => ({ color: s.color, label: s.label }));
    case 'traffic':
    default:
      return Object.entries(STATUS_META).map(([key, meta]) => ({
        color: STATUS_COLOR[key],
        label: meta.label,
        dash: meta.dash,
      }));
  }
}

/** Detects whether OSM tiles actually loaded, so we can be honest when they have not. */
function TileWatcher({ onFail }) {
  const map = useMap();
  useEffect(() => {
    let loaded = false;
    const markLoaded = () => {
      loaded = true;
    };
    map.on('tileload', markLoaded);
    const timer = setTimeout(() => {
      if (!loaded) onFail();
    }, 6000);
    return () => {
      map.off('tileload', markLoaded);
      clearTimeout(timer);
    };
  }, [map, onFail]);
  return null;
}

/**
 * The satellite wetness overlay. Each segment gets a translucent polygon around its
 * midpoint, shaded by NDWI. This is the layer that makes the satellite modality
 * visible rather than merely claimed -- and it is clickable for the raw numbers.
 */
function WetnessOverlay({ states, onSelect }) {
  return states.map((state) => {
    const segment = SEGMENT_BY_ID[state.segmentId];
    const a = latlon(segment.a);
    const b = latlon(segment.b);
    const midLat = (a[0] + b[0]) / 2;
    const midLon = (a[1] + b[1]) / 2;
    // A small diamond around the midpoint; size grows a little with wetness so the
    // signal is legible even in greyscale.
    const r = 0.004 + state.wetness * 0.004;
    const corners = [
      [midLat + r, midLon],
      [midLat, midLon + r],
      [midLat - r, midLon],
      [midLat, midLon - r],
    ];
    return (
      <Polygon
        key={`wet-${state.segmentId}`}
        positions={corners}
        pathOptions={{
          color: 'var(--flood)',
          weight: 1,
          opacity: 0.5,
          fillColor: 'var(--flood)',
          fillOpacity: 0.1 + state.wetness * 0.45,
        }}
        eventHandlers={{ click: () => onSelect?.(state.segmentId) }}
      >
        <Popup>
          <div>
            <strong>{segment.name}</strong>
            <div className="small muted" style={{ marginTop: 4 }}>
              Rain intensity here: {mm(state.rain)}
              <br />
              NDWI (wetness): {state.ndwi.toFixed(3)}
              <br />
              Wetness index: {(state.wetness * 100).toFixed(0)}%
            </div>
            <div className="small muted" style={{ marginTop: 6 }}>
              Satellite signal. Synthetic NDWI unless a Sentinel-2 export is configured.
            </div>
          </div>
        </Popup>
      </Polygon>
    );
  });
}

export default function NetworkMap({
  states,
  layer = 'traffic',
  onSelect,
  selectedId,
  height = 440,
  showWetness = false,
  children,
  topLeft,
}) {
  const [tilesFailed, setTilesFailed] = useState(false);
  const center = CITY.center;
  const legend = useMemo(() => legendFor(layer), [layer]);

  return (
    <div className="map-wrap" style={{ height }}>
      <MapContainer
        center={center}
        zoom={CITY.map_zoom}
        scrollWheelZoom={false}
        style={{ height: '100%', width: '100%' }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; OpenStreetMap contributors"
          maxZoom={19}
        />
        <TileWatcher onFail={() => setTilesFailed(true)} />

        {showWetness && <WetnessOverlay states={states} onSelect={onSelect} />}

        {layer === 'hazard' && <FloodMarkers states={states} />}

        {states.map((state) => {
          const segment = state.segment;
          const style = styleFor(state, layer);
          const isSelected = selectedId === state.segmentId;
          return (
            <Polyline
              key={state.segmentId}
              positions={[latlon(segment.a), latlon(segment.b)]}
              pathOptions={{
                color: style.color,
                weight: isSelected ? 10 : 7,
                opacity: isSelected ? 1 : 0.88,
                dashArray: style.dashArray,
                lineCap: 'round',
              }}
              eventHandlers={{ click: () => onSelect?.(state.segmentId) }}
            >
              <Tooltip sticky>
                <strong>{segment.name}</strong>
                <br />
                {layer === 'hazard' ? style.label : `${kmh(state.speed)} - ${style.label}`}
              </Tooltip>
              <Popup>
                {layer === 'hazard' ? (
                  <div>
                    <strong>{segment.name}</strong>
                    <div className="small muted" style={{ marginTop: 4 }}>
                      {style.label}. {state.flood.index >= 1 ? 'Flooding is possible on this road. ' : ''}{state.level.label} traffic.
                    </div>
                  </div>
                ) : (
                <div>
                  <strong>{segment.name}</strong>
                  <div style={{ fontSize: 17, fontWeight: 800, marginTop: 4 }}>
                    {kmh(state.speed)}
                  </div>
                  <div className="small muted">
                    {state.level.label} - free flow {kmh(segment.free_flow_kmh)}
                    <br />
                    Rain {mm(state.rain)} - flood {state.flood.label}
                    <br />
                    Impact score {state.impact}
                  </div>
                </div>
                )}
              </Popup>
            </Polyline>
          );
        })}

        {JUNCTIONS.map((junction) => (
          <CircleMarker
            key={junction.id}
            center={[junction.lat, junction.lon]}
            radius={4}
            pathOptions={{ color: 'var(--ink)', weight: 1.5, fillColor: '#fff', fillOpacity: 1 }}
          >
            <Tooltip permanent direction="top" className="jtip" offset={[0, -4]}>
              {junction.name}
            </Tooltip>
          </CircleMarker>
        ))}
      </MapContainer>

      {topLeft && <div className="map-overlay tl">{topLeft}</div>}

      <div className="map-overlay bl">
        {legend.map((item) => (
          <span className="legend-item" key={item.label}>
            <span
              className="legend-swatch"
              style={{
                background: item.dash ? 'transparent' : item.color,
                borderTop: item.dash ? `4px dashed ${item.color}` : undefined,
                height: item.dash ? 0 : 4,
              }}
              aria-hidden="true"
            />
            {item.label}
          </span>
        ))}
      </div>

      {tilesFailed && (
        <div className="map-overlay tl" style={{ top: topLeft ? 56 : 12 }}>
          Map tiles unavailable offline - roads still shown
        </div>
      )}

      {children}
    </div>
  );
}
