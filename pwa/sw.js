const CACHE_NAME = 'yose-tsumego-cache-v1';
const urlsToCache = [
    '/',
    '/index.html',
    '/app.js',
    'http://wgo.waltheri.net/css/wgo.player.css',
    'http://wgo.waltheri.net/js/wgo.min.js',
    'http://wgo.waltheri.net/js/wgo.player.min.js'
];

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                return cache.addAll(urlsToCache);
            })
    );
});

self.addEventListener('fetch', event => {
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                if (response) {
                    return response;
                }
                return fetch(event.request);
            })
    );
});
