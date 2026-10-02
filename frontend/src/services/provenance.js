/**
 * Provenance labels.
 *
 * Every number MonsoonPlus shows carries one of these, because the product makes three
 * very different kinds of claim and conflating them would be dishonest:
 *
 *   OFFICIAL    measured or forecast by a meteorological / traffic provider
 *   MODEL       output of a MonsoonPlus neural network, inside its trained area
 *   ESTIMATE    a transparent rule over real inputs -- NOT a learned prediction
 *   COMMUNITY   reported by a user, unverified
 *   DEMO        sample data shipped with the repo
 *
 * The rule enforced throughout the codebase: an ESTIMATE is never relabelled MODEL just
 * because it looks confident, and DEMO never borrows an OFFICIAL badge.
 */
export const SOURCE = {
  OFFICIAL: 'official',
  MODEL: 'model',
  ESTIMATE: 'estimate',
  COMMUNITY: 'community',
  DEMO: 'demo',
  UNAVAILABLE: 'unavailable',
};

export const SOURCE_META = {
  official: { label: 'Official forecast', tone: 'rain', short: 'Official' },
  model: { label: 'MonsoonPlus AI prediction', tone: 'flood', short: 'AI model' },
  estimate: { label: 'MonsoonPlus estimate', tone: 'slow', short: 'Estimate' },
  community: { label: 'Community report', tone: 'neutral', short: 'Reported' },
  demo: { label: 'Sample data', tone: 'neutral', short: 'Sample' },
  unavailable: { label: 'Not available', tone: 'neutral', short: 'N/A' },
};

/** Attach provenance to a value so a component can render it without guessing. */
export function fact(value, source, { basis = null, at = null, unit = null } = {}) {
  return { value, source, basis, at: at || new Date().toISOString(), unit };
}

export const unavailable = (why) => ({ value: null, source: SOURCE.UNAVAILABLE, basis: why, at: null });
