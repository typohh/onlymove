// No-op @sentry/browser stand-in for headless runs (aliased in by build.mjs).
// model.js / population-constants.js call a few of these; none need a real
// transport offline.
const noop = () => {};
export const init = noop;
export const captureException = noop;
export const captureMessage = noop;
export const addBreadcrumb = noop;
export const setUser = noop;
export const setTag = noop;
export const setContext = noop;
export const withScope = (fn) => { try { fn({ setTag: noop, setContext: noop, setLevel: noop }); } catch (_e) {} };
export default { init, captureException, captureMessage, addBreadcrumb, setUser, setTag, setContext, withScope };
