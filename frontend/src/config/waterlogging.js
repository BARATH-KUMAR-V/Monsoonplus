/**
 * Areas of Chennai that are repeatedly reported as waterlogging-prone.
 *
 * WHAT THIS IS: a hand-compiled list of neighbourhoods that appear again and again in
 * public reporting of Chennai monsoon flooding. It is a PRIOR -- it tells the estimate
 * "this general area has a history" -- and nothing more.
 *
 * WHAT THIS IS NOT: an official register, a survey, a measurement, or model output. It
 * has no street-level precision and no depth information. Anything derived from it is
 * labelled ESTIMATE in the UI, never AI PREDICTION.
 *
 * Each entry is a centre point and a radius in kilometres, with `weight` 0-1 for how
 * consistently the area is reported. Edit freely: this file is meant to be replaced by
 * a proper corporation dataset if one becomes available.
 */
export const WATERLOGGING_PRIORS = [
  { name: 'Velachery', lat: 12.9791, lon: 80.2210, km: 2.4, weight: 1.0 },
  { name: 'Pallikaranai marsh edge', lat: 12.9340, lon: 80.2120, km: 3.0, weight: 1.0 },
  { name: 'Madipakkam', lat: 12.9617, lon: 80.1960, km: 2.0, weight: 0.9 },
  { name: 'Mudichur', lat: 12.9090, lon: 80.0640, km: 2.6, weight: 1.0 },
  { name: 'Tambaram', lat: 12.9249, lon: 80.1000, km: 2.6, weight: 0.85 },
  { name: 'Perungudi', lat: 12.9630, lon: 80.2430, km: 2.0, weight: 0.85 },
  { name: 'Taramani', lat: 12.9880, lon: 80.2430, km: 1.8, weight: 0.8 },
  { name: 'Kotturpuram', lat: 13.0170, lon: 80.2430, km: 1.6, weight: 0.85 },
  { name: 'Saidapet', lat: 13.0210, lon: 80.2230, km: 1.8, weight: 0.8 },
  { name: 'Ashok Nagar and KK Nagar', lat: 13.0380, lon: 80.2110, km: 2.2, weight: 0.75 },
  { name: 'Mylapore and Mandaveli', lat: 13.0330, lon: 80.2670, km: 1.8, weight: 0.7 },
  { name: 'Thiruvanmiyur', lat: 12.9830, lon: 80.2590, km: 1.8, weight: 0.7 },
  { name: 'West Tambaram and Peerkankaranai', lat: 12.9180, lon: 80.1150, km: 2.4, weight: 0.8 },
  { name: 'Mugalivakkam and Manapakkam', lat: 13.0170, lon: 80.1660, km: 2.4, weight: 0.85 },
  { name: 'Nungambakkam and Chetpet', lat: 13.0580, lon: 80.2420, km: 1.8, weight: 0.65 },
  { name: 'Anna Nagar West', lat: 13.0870, lon: 80.2000, km: 2.0, weight: 0.65 },
  { name: 'Villivakkam and Ambattur', lat: 13.1030, lon: 80.1640, km: 2.8, weight: 0.7 },
  { name: 'Korattur and Padi', lat: 13.1110, lon: 80.1860, km: 2.2, weight: 0.7 },
  { name: 'Porur and Ramapuram', lat: 13.0330, lon: 80.1570, km: 2.4, weight: 0.8 },
  { name: 'OMR Thoraipakkam', lat: 12.9360, lon: 80.2320, km: 2.2, weight: 0.8 },
  { name: 'Semmancheri and Navalur', lat: 12.8700, lon: 80.2270, km: 3.0, weight: 0.75 },
  { name: 'Ennore and Manali', lat: 13.1950, lon: 80.2800, km: 3.2, weight: 0.7 },
  { name: 'Pulianthope and Pattalam', lat: 13.1000, lon: 80.2580, km: 1.8, weight: 0.65 },
  { name: 'Triplicane and Chepauk', lat: 13.0600, lon: 80.2770, km: 1.6, weight: 0.6 },
];

export const WATERLOGGING_NOTE =
  'Compiled from public reporting of past Chennai monsoon flooding. Not an official register and not street-level.';
