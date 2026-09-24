// Replaces the retired onlymove.pages.dev service worker. The browser fetches
// this on the next navigation, and it takes over only to remove itself: it
// drops every cache the old app stored, unregisters, and reloads open tabs so
// they reach the moved-notice page from the network.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    for (const name of await caches.keys()) await caches.delete(name);
    await self.registration.unregister();
    for (const client of await self.clients.matchAll({ type: 'window' })) client.navigate(client.url);
  })());
});
