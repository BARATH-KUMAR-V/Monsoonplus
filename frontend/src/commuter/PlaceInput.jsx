/**
 * Search-any-place field.
 *
 * This is the component that removes the old restriction to eight fixed junctions.
 * It offers, in order: the user's saved places, their recent trips, their current
 * location, and then live search results from OpenStreetMap.
 *
 * Search is debounced and the in-flight request is aborted on each keystroke, which
 * keeps the app comfortably inside the free geocoder's fair-use policy.
 */
import { useCallback, useEffect, useId, useRef, useState } from 'react';
import { searchPlaces } from '@/services/geocode';
import { useApp } from '@/data/consumer';

const ROLE_LABEL = { home: 'Home', work: 'Work', college: 'College', custom: 'Saved' };

function Pin({ kind }) {
  const common = { width: 16, height: 16, viewBox: '0 0 16 16', fill: 'none', 'aria-hidden': true };
  const s = { stroke: 'currentColor', strokeWidth: 1.5, strokeLinecap: 'round', strokeLinejoin: 'round' };
  if (kind === 'me') return <svg {...common}><circle cx="8" cy="8" r="2.5" {...s} /><circle cx="8" cy="8" r="6" {...s} /><path d="M8 .8v1.6M8 13.6v1.6M15.2 8h-1.6M2.4 8H.8" {...s} /></svg>;
  if (kind === 'home') return <svg {...common}><path d="M2 7l6-5 6 5v6a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1z" {...s} /></svg>;
  if (kind === 'work') return <svg {...common}><rect x="2" y="5" width="12" height="9" rx="1" {...s} /><path d="M6 5V3h4v2" {...s} /></svg>;
  if (kind === 'recent') return <svg {...common}><circle cx="8" cy="8" r="6" {...s} /><path d="M8 4.8V8l2.2 1.6" {...s} /></svg>;
  return <svg {...common}><path d="M8 14.5s5-4.3 5-8a5 5 0 0 0-10 0c0 3.7 5 8 5 8z" {...s} /><circle cx="8" cy="6.4" r="1.7" {...s} /></svg>;
}

export default function PlaceInput({ label, value, onChange, placeholder = 'Search any place or address', allowMyLocation = true, autoFocus = false }) {
  const { places, recentPlaces, located, locate } = useApp();
  const id = useId();
  const [text, setText] = useState('');
  const [open, setOpen] = useState(false);
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState('idle'); // idle | searching | empty | error
  const [highlight, setHighlight] = useState(-1);
  const boxRef = useRef(null);
  const abortRef = useRef(null);

  // Show the chosen place's name when not actively typing.
  const display = open ? text : value?.name || '';

  useEffect(() => {
    const onDown = (event) => {
      if (boxRef.current && !boxRef.current.contains(event.target)) setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    const query = text.trim();
    if (query.length < 3) {
      setResults([]);
      setStatus('idle');
      return undefined;
    }
    const timer = setTimeout(async () => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      setStatus('searching');
      try {
        const found = await searchPlaces(query, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setResults(found);
        setStatus(found.length ? 'idle' : 'empty');
      } catch {
        if (!controller.signal.aborted) {
          setResults([]);
          setStatus('error');
        }
      }
    }, 350);
    return () => clearTimeout(timer);
  }, [text, open]);

  const pick = useCallback(
    (place) => {
      onChange(place);
      setText('');
      setOpen(false);
      setHighlight(-1);
    },
    [onChange],
  );

  const useMyLocation = async () => {
    const place = await locate({ moveFocus: false });
    if (place) pick(place);
  };

  // Shortcuts shown before the user types anything.
  const shortcuts = [
    ...places.map((p) => ({ ...p, badge: ROLE_LABEL[p.role] || 'Saved', pin: p.role })),
    ...recentPlaces.filter((r) => !places.some((p) => p.id === r.id)).map((r) => ({ ...r, badge: 'Recent', pin: 'recent' })),
  ].slice(0, 6);

  const list = text.trim().length >= 3 ? results.map((r) => ({ ...r, pin: 'place' })) : shortcuts;

  const onKeyDown = (event) => {
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setOpen(true);
      setHighlight((h) => Math.min(list.length - 1, h + 1));
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      setHighlight((h) => Math.max(-1, h - 1));
    } else if (event.key === 'Enter' && highlight >= 0 && list[highlight]) {
      event.preventDefault();
      pick(list[highlight]);
    } else if (event.key === 'Escape') {
      setOpen(false);
    }
  };

  return (
    <div className="place" ref={boxRef}>
      <label className="field-label" htmlFor={id}>{label}</label>
      <div className="place-row">
        <span className="place-ic" aria-hidden="true"><Pin kind={value?.role || 'place'} /></span>
        <input
          id={id}
          type="text"
          autoComplete="off"
          autoFocus={autoFocus}
          value={display}
          placeholder={placeholder}
          onFocus={() => { setText(''); setOpen(true); }}
          onChange={(event) => { setText(event.target.value); setOpen(true); setHighlight(-1); }}
          onKeyDown={onKeyDown}
          role="combobox"
          aria-expanded={open}
          aria-controls={`${id}-list`}
          aria-autocomplete="list"
        />
        {value && !open && (
          <button type="button" className="place-clear" onClick={() => onChange(null)} aria-label={`Clear ${label}`}>
            <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" /></svg>
          </button>
        )}
      </div>

      {value?.context && !open && <p className="place-sub">{value.context}</p>}

      {open && (
        <div className="place-pop" id={`${id}-list`} role="listbox">
          {allowMyLocation && (
            <button type="button" className="place-opt me" onClick={useMyLocation} role="option" aria-selected="false">
              <span className="place-ic"><Pin kind="me" /></span>
              <span>
                <b>{located.state === 'locating' ? 'Finding you…' : 'Use my current location'}</b>
                {located.state === 'error' && <i className="place-err">{located.error}</i>}
              </span>
            </button>
          )}

          {status === 'searching' && <p className="place-note">Searching…</p>}
          {status === 'empty' && <p className="place-note">No places found for “{text.trim()}”. Try a landmark or an area name.</p>}
          {status === 'error' && <p className="place-note">Place search is not responding. You can also tap the map to drop a pin.</p>}

          {list.map((item, index) => (
            <button
              key={item.id}
              type="button"
              className={`place-opt ${index === highlight ? 'on' : ''}`}
              onClick={() => pick(item)}
              onMouseEnter={() => setHighlight(index)}
              role="option"
              aria-selected={index === highlight}
            >
              <span className="place-ic"><Pin kind={item.pin} /></span>
              <span className="place-txt">
                <b>{item.name}</b>
                {item.context && <i>{item.context}</i>}
              </span>
              {item.badge && <span className="place-badge">{item.badge}</span>}
            </button>
          ))}

          {!list.length && status === 'idle' && text.trim().length < 3 && (
            <p className="place-note">Start typing a place name, or tap anywhere on the map below.</p>
          )}
        </div>
      )}
    </div>
  );
}
