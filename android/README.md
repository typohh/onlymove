# Only Move TWA shell (Bubblewrap project)

Generated with `bubblewrap-cli` from `twa-manifest.json`, then **hand-modified** — do not run `bubblewrap update` blindly, it regenerates the project from the template and would clobber:

- `app/src/main/java/net/onlymove/twa/NotificationPermissionBridgeActivity.java` — the in-app notification permission bridge. Chrome's TWA notification delegation is broken (android-browser-helper #563), so the PWA fires `intent://notifications#Intent;scheme=onlymove-bridge;package=net.onlymove.twa;end` after a Notify opt-in and this activity raises the native POST_NOTIFICATIONS prompt (or deep-links the app's notification settings screen when the prompt would be silently auto-denied). The PWA side lives in `pwa/js/ui.js` (`launchTwaNotificationBridge`); the intent URL and the `onlymove-bridge://notifications` filter must stay in sync.
- `app/src/main/AndroidManifest.xml` — carries the bridge activity's `<activity>` + intent-filter block.

If you do regenerate with Bubblewrap, re-apply those pieces (git diff against the previous commit shows exactly what to restore) before building.

Build: `./gradlew bundleRelease` (AAB for Play) — signing config in `twa-manifest.json` points at `android.keystore`. Bump `versionCode`/`versionName` in `app/build.gradle` **and** the `appVersion*` fields in `twa-manifest.json` together.
