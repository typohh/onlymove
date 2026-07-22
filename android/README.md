# Only Move TWA shell (Bubblewrap project)

Generated with `bubblewrap-cli` from `twa-manifest.json`, then **hand-modified** — do not run `bubblewrap update` blindly, it regenerates the project from the template and would clobber:

- `app/src/main/java/net/onlymove/twa/NotificationPermissionBridgeActivity.java` — the in-app notification permission bridge. Chrome's TWA notification delegation is broken (android-browser-helper #563), so the PWA fires `intent://notifications#Intent;scheme=onlymove-bridge;package=net.onlymove.twa;end` after a Notify opt-in and this activity raises the native POST_NOTIFICATIONS prompt (or deep-links the app's notification settings screen when the prompt would be silently auto-denied). The PWA side lives in `pwa/js/ui.js` (`launchTwaNotificationBridge`); the intent URL and the `onlymove-bridge://notifications` filter must stay in sync.
- `app/src/main/AndroidManifest.xml` — carries the bridge activity's `<activity>` + intent-filter block.

If you do regenerate with Bubblewrap, re-apply those pieces (git diff against the previous commit shows exactly what to restore) before building.

## Building a release

`@bubblewrap/cli` is a version-pinned local devDependency (deliberately not global/npx — the CLI's template-regeneration behavior must not drift under a hand-maintained project):

```bash
cd android
npm install    # first time only
npm run build
```

The build uses the JDK/SDK paths in `~/.bubblewrap/config.json`, runs Gradle (`assembleRelease` + `bundleRelease`), and signs with `android.keystore` — it prompts for the keystore and key passwords, or reads them from `BUBBLEWRAP_KEYSTORE_PASSWORD` / `BUBBLEWRAP_KEY_PASSWORD` env vars (`npm run build -- --skipSigning` for an unsigned smoke build). Outputs land here: `app-release-signed.apk` (local testing) and `app-release-bundle.aab` (upload this to Play).

**The update prompt is a clobber trap.** `bubblewrap build` sha1-compares `twa-manifest.json` against `manifest-checksum.txt`, and on mismatch asks "the manifest was updated, apply the changes to the project?" — **defaulting to Yes, which regenerates the project from the template** and wipes the hand-maintained pieces listed above. If the prompt ever appears, answer **No** (the build then proceeds on the current project files). After editing `twa-manifest.json` deliberately, silence the prompt instead by re-syncing the checksum: `sha1sum twa-manifest.json | cut -d' ' -f1 | tr -d '\n' > manifest-checksum.txt` (the build rewrites this file itself after a successful run).

Version bumps: `versionCode`/`versionName` in `app/build.gradle` **and** the `appVersion*` fields in `twa-manifest.json` together.
