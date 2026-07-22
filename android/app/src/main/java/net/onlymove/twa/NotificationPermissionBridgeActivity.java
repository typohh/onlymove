package net.onlymove.twa;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.Settings;

import androidx.core.app.NotificationManagerCompat;

/**
 * In-app notification permission bridge for the PWA.
 *
 * Chrome's TWA notification delegation is broken (android-browser-helper #563):
 * when the site subscribes to push, the native POST_NOTIFICATIONS prompt that
 * Chrome should raise through this app never appears, so on Android 13+ the
 * app stays blocked and reminders can't display. The web app works around it
 * by navigating to intent://notifications#Intent;scheme=onlymove-bridge;...
 * right after a successful Notify opt-in, which lands here.
 *
 * Behavior, cheapest path first:
 *  - notifications already deliverable -> finish immediately (no UI at all;
 *    the activity is transparent, so this is invisible).
 *  - promptable (Android 13+, first ask ever or a previous soft denial) ->
 *    raise the native permission dialog, then finish whatever the outcome.
 *    An explicit denial is respected — no settings shove; the PWA's Notify
 *    button resets on its own once Chrome's revalidation clears the orphaned
 *    subscription.
 *  - not promptable (permanently denied, notifications toggled off in system
 *    settings, or pre-13 with notifications disabled) -> the dialog would be
 *    silently auto-denied, so open the app's notification settings screen
 *    directly instead — the exact toggle the old in-web instructions modal
 *    walked the user to by hand.
 */
public class NotificationPermissionBridgeActivity extends Activity {
    private static final int REQUEST_CODE = 1;
    private static final String PREFS = "notification_bridge";
    // Whether the native POST_NOTIFICATIONS dialog has ever been requested.
    // Needed because shouldShowRequestPermissionRationale() is false both
    // before the first ask and after a permanent denial — this flag tells the
    // two states apart.
    private static final String KEY_ASKED = "post_notifications_asked";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (NotificationManagerCompat.from(this).areNotificationsEnabled()) {
            finish();
            return;
        }

        if (Build.VERSION.SDK_INT >= 33 && canPrompt()) {
            getPrefs().edit().putBoolean(KEY_ASKED, true).apply();
            requestPermissions(new String[]{Manifest.permission.POST_NOTIFICATIONS}, REQUEST_CODE);
            return;
        }

        openNotificationSettings();
        finish();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        finish();
    }

    private boolean canPrompt() {
        // Permission granted but notifications still not deliverable means the
        // app's notifications are toggled off in system settings — only the
        // settings screen can fix that.
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED) {
            return false;
        }
        return !getPrefs().getBoolean(KEY_ASKED, false)
                || shouldShowRequestPermissionRationale(Manifest.permission.POST_NOTIFICATIONS);
    }

    private void openNotificationSettings() {
        Intent intent;
        if (Build.VERSION.SDK_INT >= 26) {
            intent = new Intent(Settings.ACTION_APP_NOTIFICATION_SETTINGS)
                    .putExtra(Settings.EXTRA_APP_PACKAGE, getPackageName());
        } else {
            intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                    Uri.fromParts("package", getPackageName(), null));
        }
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        try {
            startActivity(intent);
        } catch (Exception e) {
            // No settings activity to handle it (heavily customized ROM) —
            // nothing more the bridge can do.
        }
    }

    private SharedPreferences getPrefs() {
        return getSharedPreferences(PREFS, MODE_PRIVATE);
    }
}
