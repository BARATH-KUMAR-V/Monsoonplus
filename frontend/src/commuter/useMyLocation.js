import { useCallback, useEffect, useState } from 'react';
import { nearestJunction } from './plain';

const MAX_KM = 15; // beyond this we are not in the monitored zone

/**
 * Finds the junction nearest to the user. Runs once on mount (that is the "default to
 * my location" behaviour) and again on demand. The position never leaves the browser.
 */
export function useMyLocation({ auto = true } = {}) {
  const [loc, setLoc] = useState({ state: 'idle', junction: null });

  const locate = useCallback(() => {
    if (!('geolocation' in navigator)) {
      setLoc({ state: 'unsupported', junction: null });
      return;
    }
    setLoc({ state: 'locating', junction: null });
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const near = nearestJunction(pos.coords.latitude, pos.coords.longitude);
        setLoc(near && near.km <= MAX_KM ? { state: 'ok', junction: near } : { state: 'outside', junction: null });
      },
      () => setLoc({ state: 'denied', junction: null }),
      { enableHighAccuracy: false, timeout: 8000, maximumAge: 120000 },
    );
  }, []);

  useEffect(() => {
    if (auto) locate();
  }, [auto, locate]);

  return { ...loc, locate };
}
