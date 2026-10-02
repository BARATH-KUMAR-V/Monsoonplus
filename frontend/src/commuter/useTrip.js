/**
 * The trip pipeline, in one hook.
 *
 *   routes (OSRM)  ->  sample points  ->  weather at those points (Open-Meteo)
 *                                     ->  ground elevation (Open-Meteo)
 *                                     ->  score + explain + departure advice
 *
 * Weather and elevation are fetched in parallel and elevation is allowed to fail: the
 * flood estimate degrades to rainfall and history, and says so, rather than the whole
 * trip failing because one optional service was slow.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { findRoutes, samplePoints } from '@/services/routing';
import { fetchWeatherAtPoints } from '@/services/weather';
import { fetchElevations } from '@/services/elevation';
import { scoreRoutes } from '@/engine/score';
import { explainChoice } from '@/engine/explain';
import { arriveByAdvice, departureAdvice } from '@/engine/departure';

/** Rain at a point, for the hour the traveller actually reaches it. */
function rainForPoint(point, departAt) {
  const rows = point.hourly || [];
  if (!rows.length) return { ...point, rain: { nowMm: 0, next3hMm: 0, recent6hMm: 0 }, wind: {}, visibility: null, noForecast: true };

  const arrival = departAt.getTime() + (point.fraction || 0) * 40 * 60000;
  const nearest = rows.reduce((best, row) => (Math.abs(row.date - arrival) < Math.abs(best.date - arrival) ? row : best), rows[0]);
  const index = rows.indexOf(nearest);

  const next3h = rows.slice(index, index + 3);
  const recent6h = rows.slice(Math.max(0, index - 6), index);

  return {
    ...point,
    rain: {
      nowMm: nearest.precipitation ?? 0,
      next3hMm: Math.max(0, ...next3h.map((r) => r.precipitation ?? 0)),
      recent6hMm: recent6h.reduce((sum, r) => sum + (r.precipitation ?? 0), 0),
      chance: nearest.precipitation_probability ?? null,
    },
    wind: { speed: nearest.wind_speed_10m, gust: nearest.wind_gusts_10m, direction: nearest.wind_direction_10m },
    visibility: nearest.visibility ?? null,
    weatherCode: nearest.weather_code,
    temperature: nearest.temperature_2m,
  };
}

export function useTrip({ from, to, profileId, reports, departAt, arriveBy }) {
  const [state, setState] = useState({ status: 'idle', routes: [], error: null, warnings: [] });
  const runId = useRef(0);

  const run = useCallback(async () => {
    if (!from || !to) {
      setState({ status: 'idle', routes: [], error: null, warnings: [] });
      return;
    }
    const id = ++runId.current;
    setState((prev) => ({ ...prev, status: 'loading', error: null }));

    try {
      const found = await findRoutes(from, to);
      if (id !== runId.current) return;
      if (!found.length) throw new Error('No route was found between those two points.');

      const warnings = [];
      const sampled = found.map((route) => ({ route, points: samplePoints(route.coordinates, 8) }));
      const allPoints = sampled.flatMap((s) => s.points);

      const [weatherResult, elevationResult] = await Promise.allSettled([
        fetchWeatherAtPoints(allPoints),
        fetchElevations(allPoints),
      ]);
      if (id !== runId.current) return;

      if (weatherResult.status === 'rejected') {
        warnings.push('Weather along the route could not be loaded, so flooding risk and rain are not shown for this trip.');
      }
      if (elevationResult.status === 'rejected' || !elevationResult.value) {
        warnings.push('Ground height data was not available, so the flooding estimate uses rainfall and past flooding only.');
      }

      const weatherPoints = weatherResult.status === 'fulfilled' ? weatherResult.value : allPoints;
      const elevations = elevationResult.status === 'fulfilled' ? elevationResult.value : null;

      // Put the flat per-point results back onto their routes.
      let cursor = 0;
      const when = departAt || new Date();
      let short = false;
      const candidates = sampled.map(({ route, points }) => {
        // A provider that answers with fewer points than we asked for must not leave a
        // route silently un-assessed: fall back to the bare coordinate, which still
        // carries lat/lon so terrain and flood history can be read, and flag it.
        const slice = points.map((point, i) => weatherPoints[cursor + i] || point);
        if (slice.some((p) => !p.hourly?.length)) short = true;
        const elevationSlice = elevations ? elevations.slice(cursor, cursor + points.length) : null;
        cursor += points.length;
        return {
          route,
          samples: slice.map((p) => rainForPoint(p, when)),
          elevations: elevationSlice?.length === points.length ? elevationSlice : null,
        };
      });
      if (short && weatherResult.status === 'fulfilled') {
        warnings.push('The weather service returned fewer points than we asked for, so rain along part of the route is unknown.');
      }

      const scored = scoreRoutes(candidates, { profileId, departAt: when, reports });
      if (id !== runId.current) return;

      setState({ status: 'ready', routes: scored, error: null, warnings });
    } catch (error) {
      if (id !== runId.current) return;
      setState({ status: 'error', routes: [], error: error.message || 'Something went wrong planning this trip.', warnings: [] });
    }
  }, [from, to, profileId, reports, departAt]);

  useEffect(() => {
    run();
  }, [run]);

  const recommended = state.routes.find((r) => r.recommended) || state.routes[0] || null;
  const alternatives = state.routes.filter((r) => r !== recommended);

  const explanation = recommended ? explainChoice(recommended, alternatives) : null;
  const departure = recommended && !arriveBy ? departureAdvice(recommended, { reports }) : null;
  const plan = recommended && arriveBy ? arriveByAdvice(recommended, arriveBy, { reports }) : null;

  return { ...state, recommended, alternatives, explanation, departure, plan, retry: run };
}
