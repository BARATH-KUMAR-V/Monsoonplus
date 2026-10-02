/**
 * Who the user is, and what that changes.
 *
 * A profile is not cosmetic: it sets the weights in the route score, so a rider and a
 * car driver genuinely get different recommendations from identical data. The weights
 * are declared here in one place so they can be read, argued with and tuned, rather
 * than being buried in the scoring code.
 *
 * Weights are applied to normalised 0-1 risk components and must sum to 1 within each
 * profile, which `assertWeights` checks at module load in development.
 */

export const PROFILES = [
  {
    id: 'car',
    label: 'Car',
    blurb: 'Driving a car or taxi',
    icon: 'car',
    weights: { time: 0.34, traffic: 0.16, flood: 0.34, weather: 0.08, hazard: 0.08 },
    cares: ['Deep water that can stall the engine', 'Flooded underpasses', 'Blocked roads'],
    tips: {
      flood: 'Do not drive into standing water. A car can float in 30 cm, and you cannot see a washed-out road under it.',
      heavyRain: 'Keep extra distance. Braking takes much longer on a wet road.',
    },
  },
  {
    id: 'bike',
    label: 'Two-wheeler',
    blurb: 'Riding a bike or scooter',
    icon: 'bike',
    // A rider is exposed, so weather and wind matter far more than a few minutes.
    weights: { time: 0.2, traffic: 0.14, flood: 0.3, weather: 0.24, hazard: 0.12 },
    cares: ['Getting caught in heavy rain', 'Strong side winds', 'Slippery and waterlogged roads'],
    tips: {
      flood: 'Water hides potholes and open drains. Do not ride through standing water.',
      wind: 'Strong gusts push a two-wheeler sideways, especially on flyovers and open stretches.',
      heavyRain: 'Painted road markings and metal covers get very slippery in the first minutes of rain.',
    },
  },
  {
    id: 'walk',
    label: 'Walking',
    blurb: 'On foot',
    icon: 'walk',
    weights: { time: 0.16, traffic: 0.1, flood: 0.3, weather: 0.3, hazard: 0.14 },
    cares: ['Getting soaked', 'Water on the footpath', 'Poor visibility at crossings'],
    tips: {
      flood: 'Do not wade through moving water. Open drains and manholes are invisible under it.',
      heavyRain: 'Drivers see you late in heavy rain. Cross where you can be seen.',
    },
  },
  {
    id: 'student',
    label: 'College or school',
    blurb: 'Getting to class on time',
    icon: 'school',
    weights: { time: 0.38, traffic: 0.2, flood: 0.26, weather: 0.1, hazard: 0.06 },
    cares: ['Arriving on time', 'Knowing when to leave', 'Buses stuck in traffic'],
    tips: { heavyRain: 'Leave earlier than usual. Everyone travels slower in rain, so the whole road is slower.' },
  },
  {
    id: 'family',
    label: 'Family or school run',
    blurb: 'Travelling with children',
    icon: 'family',
    weights: { time: 0.22, traffic: 0.14, flood: 0.38, weather: 0.16, hazard: 0.1 },
    cares: ['Avoiding risk entirely', 'Predictable timing', 'Not getting stuck'],
    tips: { flood: 'With children in the car, turn back rather than attempt any flooded stretch.' },
  },
  {
    id: 'delivery',
    label: 'Delivery or cab',
    blurb: 'Driving for work, many trips a day',
    icon: 'delivery',
    weights: { time: 0.42, traffic: 0.22, flood: 0.22, weather: 0.08, hazard: 0.06 },
    cares: ['Fastest realistic route', 'Which areas to avoid today', 'Live changes'],
    tips: { heavyRain: 'Allow more time per drop. Rain adds time to every trip, not only the flooded ones.' },
  },
  {
    id: 'weather',
    label: 'Just checking the weather',
    blurb: 'No trip planned',
    icon: 'cloud',
    weights: { time: 0.3, traffic: 0.15, flood: 0.25, weather: 0.2, hazard: 0.1 },
    cares: ['Current conditions', 'Will it rain', 'The week ahead'],
    tips: {},
  },
];

export const PROFILE_BY_ID = Object.fromEntries(PROFILES.map((p) => [p.id, p]));
export const DEFAULT_PROFILE = 'car';

export const getProfile = (id) => PROFILE_BY_ID[id] || PROFILE_BY_ID[DEFAULT_PROFILE];

/** Which page a profile should land on. Someone checking the weather wants the weather. */
export const homeRouteFor = (id) => (id === 'weather' ? '/weather' : '/');

if (import.meta.env?.DEV) {
  PROFILES.forEach((p) => {
    const total = Object.values(p.weights).reduce((a, b) => a + b, 0);
    if (Math.abs(total - 1) > 0.001) {
      console.warn(`[profiles] weights for "${p.id}" sum to ${total.toFixed(3)}, not 1`);
    }
  });
}
