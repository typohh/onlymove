# TODO

## Decide: `viewport-fit=cover`

`pwa/index.html:5` ships the viewport meta *without* `viewport-fit=cover`:

```html
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
```

Without it, every `env(safe-area-inset-*)` resolves to `0px`, so the whole safe-area half of the layout basis in `pwa/style.css` (`--safe-t/r/b/l` at lines 53–56, feeding `--app-h`/`--app-w`/`--u` and the `body` padding at line 143) is present but **inert**. It was written that way deliberately so enabling cover is a one-attribute change rather than a reflow — but the change has never actually been made or tested, so it's a decision, not a pending edit.

What has to be settled before flipping it:

- **What we gain.** Under cover the page paints edge-to-edge and the shell then pulls itself back inside the notch/home-indicator via the existing padding; the decorative full-bleed layers (`#corner-art`, overlay/modal backdrops — all `position: fixed`) stay edge-to-edge, which is the point. Without cover those layers stop at the browser's own inset letterboxing.
- **What it costs.** Turning insets from 0 to non-zero shrinks `--app-w`/`--app-h`, hence `--u`, hence the board, the type scale, `--col-w` and `--mb-tab-reserve` — on an inset device every shell length moves at once. Needs a real look on a notched phone in portrait *and* landscape (landscape is where left/right insets bite the column gutters and the parked mobile buttons, `style.css:621–624`).
- **Where to check it.** Three surfaces behave differently and all three matter: the installed PWA, the Android TWA (`android/`), and in-app browsers hitting share links (`?id`) — the last is the one the dvh basis was written for and the one with the most toolbar chrome.

Decide: enable cover (and re-check the shell on inset hardware), or drop the inert safe-area plumbing and say so in the `style.css` header comment.
