/**
 * The ask-anything layer.
 *
 * ARCHITECTURE NOTE, and the point of the whole module: the assistant never predicts
 * anything. It matches the question to an intent, pulls the already-computed facts out
 * of the app (official forecast, model output, estimates, reports), and puts them into
 * a sentence. If the facts are not there, it says so.
 *
 * That ordering -- facts first, language second -- is what makes it safe. Dropping a
 * hosted LLM in later means replacing `compose()` with a call that receives this same
 * `facts` object as its only source of truth; `answer()` would not change. There is no
 * LLM key in this build, so the composition is written out by hand.
 */
import { rainWord, describeCode, compass, nextRain, severeConditions } from '@/services/weather';
import { clock } from './departure';

const has = (text, ...words) => words.some((w) => text.includes(w));

/** Work out what is being asked. Order matters: the first match wins. */
export function classify(question) {
  const q = String(question || '').toLowerCase();

  if (has(q, 'safe to ride', 'safe to drive', 'should i ride', 'should i drive', 'is it safe', 'safe to go', 'safe to travel'))
    return 'safety';
  if (has(q, 'when should i leave', 'what time should i leave', 'leave now', 'should i wait', 'when to leave'))
    return 'departure';
  if (has(q, 'which route', 'best route', 'safer route', 'what route', 'which way'))
    return 'route';
  if (has(q, 'will it rain', 'is it going to rain', 'rain today', 'rain now', 'raining'))
    return 'rain';
  if (has(q, 'flood', 'waterlog', 'water on the road'))
    return 'flood';
  if (has(q, 'wind', 'gust', 'storm', 'thunder', 'cyclone'))
    return 'severe';
  if (has(q, 'tomorrow', 'this week', 'next week', 'weekend', 'forecast'))
    return 'forecast';
  if (has(q, 'temperature', 'how hot', 'how cold', 'weather', 'humid', 'uv', 'air quality', 'aqi'))
    return 'weather';
  if (has(q, 'how long', 'how far', 'eta', 'travel time'))
    return 'duration';
  return 'unknown';
}

const bullet = (text, source) => ({ text, source });

/**
 * @param question
 * @param facts  { weather, air, severe, anomalies, trip, place, profile, reports }
 * @returns { intent, answer, points, missing }
 */
export function answer(question, facts) {
  const intent = classify(question);
  const { weather, air, trip, place, profile, reports } = facts || {};
  const current = weather?.current;
  const where = place?.name || 'your area';

  // Nothing loaded yet is a real answer, not an excuse to improvise one.
  if (!weather && ['rain', 'weather', 'forecast', 'severe', 'safety'].includes(intent)) {
    return {
      intent,
      answer: 'I do not have the weather loaded yet. Open the Weather page once and ask me again.',
      points: [],
      missing: ['weather'],
    };
  }

  switch (intent) {
    case 'rain': {
      const soon = nextRain(weather.hourly);
      if (!soon) return unknown(intent, 'I could not read the hourly forecast.');
      const nowMm = current?.precipitation ?? 0;
      const raining = nowMm > 0.2;
      const lead = raining
        ? `It is raining in ${where} right now, ${rainWord(nowMm)}, ${nowMm.toFixed(1)} mm in the last hour.`
        : `It is not raining in ${where} right now.`;
      const next = !soon.willRain
        ? `Rain in the next three hours looks unlikely, with a peak chance of ${Math.round(soon.chance)}%.`
        : raining
          ? `It is expected to keep going, with around ${soon.total.toFixed(1)} mm over the next three hours.`
          : soon.startsInMinutes < 10
            ? 'Rain is about to start.'
            : `Rain is likely to start in about ${soon.startsInMinutes} minutes.`;
      return {
        intent,
        answer: `${lead} ${next}`,
        points: [bullet(`Chance of rain in the next 3 hours: ${Math.round(soon.chance)}%`, 'official'),
          bullet(`Expected total: ${soon.total.toFixed(1)} mm`, 'official')],
        missing: [],
      };
    }

    case 'weather': {
      const code = describeCode(current?.weather_code);
      const points = [
        bullet(`Feels like ${Math.round(current?.apparent_temperature ?? current?.temperature_2m)}°C`, 'official'),
        bullet(`Humidity ${Math.round(current?.relative_humidity_2m)}%`, 'official'),
        bullet(`Wind ${Math.round(current?.wind_speed_10m)} km/h from the ${compass(current?.wind_direction_10m)}`, 'official'),
      ];
      if (air?.us_aqi != null) points.push(bullet(`Air quality index ${Math.round(air.us_aqi)}`, 'official'));
      return {
        intent,
        answer: `${where} is ${Math.round(current?.temperature_2m)}°C and ${code.text.toLowerCase()}.`,
        points,
        missing: air ? [] : ['air quality'],
      };
    }

    case 'forecast': {
      const days = (weather.daily || []).slice(0, 5);
      if (!days.length) return unknown(intent, 'The daily forecast did not load.');
      const wettest = days.reduce((a, b) => ((b.precipitation_sum ?? 0) > (a.precipitation_sum ?? 0) ? b : a));
      const dayName = (d) => d.date.toLocaleDateString([], { weekday: 'long' });
      return {
        intent,
        answer:
          (wettest.precipitation_sum ?? 0) > 2
            ? `The wettest day in the next five is ${dayName(wettest)}, with about ${wettest.precipitation_sum.toFixed(0)} mm expected.`
            : 'The next five days look mostly dry, with no day above 2 mm of rain.',
        points: days.map((d) =>
          bullet(`${dayName(d)}: ${Math.round(d.temperature_2m_min)} to ${Math.round(d.temperature_2m_max)}°C, ${(d.precipitation_sum ?? 0).toFixed(1)} mm`, 'official'),
        ),
        missing: [],
      };
    }

    case 'severe': {
      const conditions = severeConditions(weather);
      if (!conditions.length) {
        return { intent, answer: `No storms, strong winds or extreme heat are showing in the forecast for ${where} over the next 12 hours.`, points: [], missing: [] };
      }
      return {
        intent,
        answer: conditions[0].title + '. ' + conditions[0].detail,
        points: conditions.slice(1).map((c) => bullet(`${c.title}: ${c.detail}`, 'official')),
        missing: [],
      };
    }

    case 'flood': {
      if (trip?.recommended?.flood) {
        const flood = trip.recommended.flood;
        return {
          intent,
          answer: `On your planned route the flooding risk is ${flood.band.label.toLowerCase()}. ${flood.band.advice}`,
          points: flood.reasons.map((r) => bullet(r, flood.source)),
          missing: [],
        };
      }
      const active = (reports || []).filter((r) => ['flood', 'waterlogging'].includes(r.type));
      if (active.length) {
        return {
          intent,
          answer: `${active.length} ${active.length === 1 ? 'person has' : 'people have'} reported water on roads recently.`,
          points: active.slice(0, 4).map((r) => bullet(`${r.label} near ${r.placeName || 'a reported location'}`, 'community')),
          missing: [],
        };
      }
      return { intent, answer: 'No flooding has been reported, and I have no trip planned to check. Plan a trip and I can estimate the risk on it.', points: [], missing: ['trip'] };
    }

    case 'route': {
      if (!trip?.recommended) return unknown(intent, 'Plan a trip first and I can compare the routes for you.');
      const r = trip.recommended;
      return {
        intent,
        answer: `${trip.explanation.headline} Route ${r.route.letter} takes about ${Math.round(r.time.minutes)} minutes over ${r.route.distanceKm.toFixed(1)} km.`,
        points: [
          bullet(`Traffic: ${r.time.band.label}`, 'estimate'),
          bullet(`Flooding risk: ${r.flood?.band.label ?? 'unknown'}`, r.flood?.source ?? 'estimate'),
          bullet(`Rain on the way: ${r.weather.peakRainMm > 0.2 ? rainWord(r.weather.peakRainMm) : 'none expected'}`, 'official'),
        ],
        missing: [],
      };
    }

    case 'departure': {
      if (!trip?.departure) return unknown(intent, 'Plan a trip first and I can work out when to leave.');
      return {
        intent,
        answer: `${trip.departure.headline} ${trip.departure.detail}`,
        points: trip.departure.options
          .filter((o) => o.offset % 30 === 0)
          .slice(0, 4)
          .map((o) => bullet(`Leave ${o.offset === 0 ? 'now' : `at ${clock(o.departAt)}`}: about ${Math.round(o.minutes)} min`, 'estimate')),
        missing: [],
      };
    }

    case 'duration': {
      if (!trip?.recommended) return unknown(intent, 'Plan a trip and I can estimate how long it will take.');
      const r = trip.recommended;
      return {
        intent,
        answer: `About ${Math.round(r.time.minutes)} minutes for ${r.route.distanceKm.toFixed(1)} km, arriving around ${clock(new Date(Date.now() + r.time.minutes * 60000))}.`,
        points: [bullet(`Free-flowing it would be ${Math.round(r.route.baseMinutes)} min; the rest is traffic and rain`, 'estimate')],
        missing: [],
      };
    }

    case 'safety': {
      const conditions = severeConditions(weather);
      const soon = nextRain(weather.hourly);
      const floodRisk = trip?.recommended?.flood?.score ?? 0;
      const severe = conditions.some((c) => c.severity === 'severe');

      let verdict;
      if (severe || floodRisk >= 0.65) verdict = 'I would put this trip off if you can.';
      else if (floodRisk >= 0.4 || (soon?.peakMm ?? 0) >= 7.6) verdict = 'It is doable, but take care.';
      else verdict = 'Nothing in the forecast says you should not go.';

      const points = [];
      if (conditions.length) points.push(bullet(conditions[0].title, 'official'));
      if (soon?.willRain) {
        points.push(bullet(
          soon.startsInMinutes < 10 ? 'Rain is about to start' : `Rain likely in about ${soon.startsInMinutes} minutes`,
          'official',
        ));
      }
      if (trip?.recommended?.flood) points.push(bullet(`Flooding risk on your route: ${trip.recommended.flood.band.label}`, trip.recommended.flood.source));
      (profile?.cares || []).slice(0, 1).forEach((c) => points.push(bullet(`For your travel mode, the thing to watch is: ${c.toLowerCase()}`, 'estimate')));

      return {
        intent,
        answer: `${verdict} This is a forecast, not an observation of the road. If you reach water, turn back.`,
        points,
        missing: trip ? [] : ['trip'],
      };
    }

    default:
      return {
        intent: 'unknown',
        answer:
          'I can answer questions about the weather now and ahead, whether it will rain, storms and wind, flooding risk, which route to take, how long a trip will take, and when to leave. Try "will it rain in the next hour?" or "when should I leave for OMR?".',
        points: [],
        missing: [],
      };
  }
}

const unknown = (intent, why) => ({ intent, answer: why, points: [], missing: ['context'] });

export const SUGGESTED = [
  'Will it rain in the next hour?',
  'Is it safe to ride right now?',
  'When should I leave?',
  'Which route is safer?',
  'What is the weather this week?',
  'Are there any storms coming?',
];
