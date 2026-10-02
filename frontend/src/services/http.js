/**
 * One fetch wrapper for every external call.
 *
 * Everything MonsoonPlus talks to is a free, keyless, CORS-enabled public service, so
 * the browser calls it directly and the site deploys as static files with no server.
 * That makes failure handling the important part: a public demo endpoint WILL be slow
 * or rate-limited sometimes, and the rule for this app is that a failure must surface
 * as "unavailable", never as a plausible-looking made-up number.
 */

export class ServiceError extends Error {
  constructor(service, message, { status = null, cause = null } = {}) {
    super(message);
    this.name = 'ServiceError';
    this.service = service;
    this.status = status;
    this.cause = cause;
  }
}

const memory = new Map();

/** Session-scoped response cache. Keeps us well inside every provider's fair-use policy. */
function cacheGet(key, maxAgeMs) {
  const hit = memory.get(key);
  if (!hit) return null;
  if (Date.now() - hit.at > maxAgeMs) {
    memory.delete(key);
    return null;
  }
  return hit.value;
}

function cacheSet(key, value) {
  // Bounded so a long session cannot grow the cache without limit.
  if (memory.size > 200) memory.delete(memory.keys().next().value);
  memory.set(key, { at: Date.now(), value });
}

export function clearServiceCache() {
  memory.clear();
}

/**
 * @param url      absolute URL
 * @param service  short name used in error messages and provenance labels
 * @param options  { timeout, cacheMs, retries, signal }
 */
export async function getJson(url, service, { timeout = 12000, cacheMs = 0, retries = 1, signal } = {}) {
  if (cacheMs > 0) {
    const cached = cacheGet(url, cacheMs);
    if (cached) return cached;
  }

  let lastError = null;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeout);
    const onAbort = () => controller.abort();
    signal?.addEventListener('abort', onAbort);
    try {
      const response = await fetch(url, { signal: controller.signal, headers: { Accept: 'application/json' } });
      if (!response.ok) {
        // 4xx will not fix itself on retry; 429/5xx might.
        const retryable = response.status === 429 || response.status >= 500;
        lastError = new ServiceError(service, `${service} returned ${response.status}`, { status: response.status });
        if (!retryable || attempt === retries) throw lastError;
      } else {
        const data = await response.json();
        if (cacheMs > 0) cacheSet(url, data);
        return data;
      }
    } catch (error) {
      if (error instanceof ServiceError) lastError = error;
      else if (error.name === 'AbortError') lastError = new ServiceError(service, `${service} timed out`, { cause: error });
      else lastError = new ServiceError(service, `${service} could not be reached`, { cause: error });
      if (attempt === retries) throw lastError;
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener('abort', onAbort);
    }
    // Back off before the one retry we allow.
    await new Promise((resolve) => setTimeout(resolve, 600 * (attempt + 1)));
  }
  throw lastError;
}

export const qs = (params) =>
  Object.entries(params)
    .filter(([, v]) => v !== undefined && v !== null && v !== '')
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(Array.isArray(v) ? v.join(',') : v)}`)
    .join('&');
