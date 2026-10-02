/**
 * "Why this route?" -- explainable AI, built from the scoring evidence rather than
 * written after the decision.
 *
 * The rule throughout: say the trade-off in the user's terms (minutes, roads, rain),
 * never the score. A number like 0.37 explains nothing to a person deciding whether to
 * leave now.
 */
import { rainWord } from '@/services/weather';

const minutes = (m) => `${Math.round(m)} min`;

const COMPONENT_WORDS = {
  time: 'travel time',
  traffic: 'traffic',
  flood: 'flooding risk',
  weather: 'rain and wind',
  hazard: 'reported hazards',
};

/** The headline sentence for the recommended route. */
export function explainChoice(recommended, alternatives) {
  const faster = alternatives.find((alt) => alt.time.minutes < recommended.time.minutes - 1.5);

  // Case 1: the recommendation is also the fastest.
  //
  // Careful with the wording here. "The safest route" is only true when the route is
  // actually clear -- saying it over a route with heavy rain and water pooling on it
  // would be the single most misleading sentence in the app.
  if (!faster) {
    const flood = recommended.flood?.band?.key;
    const rain = recommended.weather.peakRainMm;
    const gust = recommended.weather.peakGust;

    if (flood === 'high' || flood === 'moderate') {
      return {
        headline: 'This is the best of the routes we found, but none of them is dry.',
        detail: 'Every option passes roads where water is likely. This one has the least of it.',
      };
    }
    if (flood === 'watch' || rain >= 7.6 || gust >= 45) {
      const notes = [];
      if (rain >= 7.6) notes.push(`${rainWord(rain)} rain along the way`);
      if (gust >= 45) notes.push(`gusts to ${Math.round(gust)} km/h`);
      if (flood === 'watch') notes.push('shallow water possible at the usual spots');
      return {
        headline: 'This is the quickest route, and no alternative is safer.',
        detail: `Conditions on it are still rough: ${notes.join(', ')}.`,
      };
    }
    return {
      headline: 'This is both the quickest route and the clearest one.',
      detail: 'No flooding, hazards or heavy rain are expected along it right now.',
    };
  }

  // Case 2: we are recommending something slower. The trade-off IS the explanation.
  const cost = recommended.time.minutes - faster.time.minutes;
  const reasons = [];

  const floodGap = (faster.flood?.score ?? 0) - (recommended.flood?.score ?? 0);
  if (floodGap > 0.12) {
    const risky = faster.flood?.riskyCount ?? 0;
    const where = faster.flood?.worst;
    const area = where?.flood?.reasons?.find((r) => r.includes('floods often'));
    reasons.push(
      risky > 1
        ? `it avoids ${risky} stretches on Route ${faster.route.letter} where water is likely`
        : `it avoids a stretch on Route ${faster.route.letter} where water is likely${area ? ` (${area.replace(' floods often', '')})` : ''}`,
    );
  }

  if (faster.hazard.hits.length && !recommended.hazard.hits.length) {
    const hit = faster.hazard.hits[0];
    reasons.push(`Route ${faster.route.letter} has ${faster.hazard.hits.length > 1 ? `${faster.hazard.hits.length} hazards` : 'a hazard'} reported on it${hit.label ? ` (${hit.label.toLowerCase()})` : ''}`);
  }

  const weatherGap = faster.weather.risk - recommended.weather.risk;
  if (weatherGap > 0.15) {
    reasons.push(`there is less rain and wind on the way`);
  }

  const trafficGap = faster.components.traffic - recommended.components.traffic;
  if (!reasons.length && trafficGap > 0.15) {
    reasons.push(`traffic is lighter on it`);
  }

  if (!reasons.length) {
    return {
      headline: `Route ${recommended.route.letter} is the balanced choice.`,
      detail: `It is ${minutes(cost)} slower than the quickest option but scores better overall on your settings.`,
    };
  }

  return {
    headline: `${minutes(cost)} slower, but ${reasons[0]}.`,
    detail: reasons.length > 1 ? `Also, ${reasons.slice(1).join(', and ')}.` : null,
  };
}

/** Per-route plain-language bullet points, used on the comparison cards. */
export function routeFactors(scored) {
  const out = [];
  const { time, flood, weather, hazard, components } = scored;

  out.push({
    key: 'time',
    label: 'Travel time',
    value: minutes(time.minutes),
    note:
      time.delayFromRain > 2
        ? `about ${minutes(time.delayFromRain)} of that is the rain`
        : time.delayFromTraffic > 3
          ? `includes about ${minutes(time.delayFromTraffic)} of traffic delay`
          : 'close to free-flowing',
    tone: components.time > 0.5 ? 'slow' : 'smooth',
  });

  out.push({
    key: 'traffic',
    label: 'Traffic',
    value: time.band.label,
    note: time.basis,
    tone: time.band.tone,
  });

  if (flood) {
    out.push({
      key: 'flood',
      label: 'Flooding risk',
      value: flood.band.label,
      note: flood.reasons.length
        ? flood.reasons.slice(0, 2).join(', ')
        : flood.band.key === 'low'
          ? 'no pooling expected on this route'
          : flood.band.advice,
      tone: flood.band.tone,
    });
  }

  const noForecast = (scored.samples || []).every((s) => s.noForecast);
  out.push({
    key: 'weather',
    label: 'Rain on the way',
    value: noForecast
      ? 'not available'
      : weather.peakRainMm > 0.2
        ? `${rainWord(weather.peakRainMm)} (${weather.peakRainMm.toFixed(1)} mm/h)`
        : 'none expected',
    note: weather.peakGust > 35 ? `gusts to ${Math.round(weather.peakGust)} km/h` : null,
    tone: components.weather > 0.5 ? 'disruption' : components.weather > 0.2 ? 'slow' : 'smooth',
  });

  if (hazard.hits.length) {
    out.push({
      key: 'hazard',
      label: 'Reported on this route',
      value: `${hazard.hits.length} report${hazard.hits.length > 1 ? 's' : ''}`,
      note: hazard.hits.map((h) => h.label).slice(0, 2).join(', '),
      tone: 'jammed',
    });
  }

  return out;
}

/** What tipped the balance, for the expandable "how this was decided" panel. */
export function decisionBreakdown(scored) {
  return scored.contributions
    .filter((c) => c.weighted > 0.005)
    .map((c) => ({
      label: COMPONENT_WORDS[c.key],
      share: c.weighted,
      raw: c.value,
      weight: scored.profile.weights[c.key],
    }));
}

/** Safety advice drawn from the user's profile, only where it applies. */
export function profileAdvice(scored) {
  const tips = scored.profile.tips || {};
  const notes = [];
  if ((scored.flood?.score ?? 0) >= 0.4 && tips.flood) notes.push(tips.flood);
  if (scored.weather.peakRainMm >= 7.6 && tips.heavyRain) notes.push(tips.heavyRain);
  if (scored.weather.peakGust >= 45 && tips.wind) notes.push(tips.wind);
  return notes;
}
