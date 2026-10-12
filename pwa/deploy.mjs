import { execSync } from 'child_process';

const args = process.argv.slice(2);
const force = args.includes('--force') || args.includes('--allow-dirty') || args.includes('-f');

let isDirty = false;
try {
  // Check if there are any uncommitted changes
  const status = execSync('git status --porcelain', { encoding: 'utf8' }).trim();
  isDirty = status.length > 0;
} catch (error) {
  console.error('Error checking git status:', error.message);
  process.exit(1);
}

if (isDirty && !force) {
  console.error('❌ Error: Git working tree has uncommitted changes.');
  console.error('Please commit or stash your changes before deploying.');
  console.error('To override this check and deploy anyway, use: npm run deploy -- --force');
  process.exit(1);
}

console.log(isDirty ? '⚠️ Deploying dirty tree (override active)...' : '✨ Git working tree is clean. Proceeding to deploy...');

// Deploys go out from more than one machine, and a deploy replaces the whole
// Worker: a HEAD that is BEHIND the remote silently rolls back every commit
// another machine has already deployed, and one that is AHEAD ships work the
// other machine's next deploy will roll back in turn. So the deployed commit must
// be exactly the remote's — pull, push, then deploy.
let ahead = 0, behind = 0;
try {
  execSync('git fetch --quiet', { stdio: 'inherit' });
  const counts = execSync('git rev-list --left-right --count HEAD...@{upstream}', { encoding: 'utf8' }).trim();
  [ahead, behind] = counts.split(/\s+/).map(Number);
} catch (error) {
  console.error('❌ Error: could not compare HEAD with its upstream:', error.message);
  if (!force) process.exit(1);
}

if ((ahead || behind) && !force) {
  console.error(`❌ Error: HEAD is ${ahead} commit(s) ahead and ${behind} behind its upstream.`);
  if (behind) console.error('Deploying now would roll back commits already on the remote — pull (and rebase) first.');
  if (ahead) console.error('Push first, so the next deploy from another machine includes these commits.');
  console.error('To override this check and deploy anyway, use: npm run deploy -- --force');
  process.exit(1);
}

try {
  console.log('📦 Running build...');
  execSync('npm run build', { stdio: 'inherit' });

  console.log('🚀 Running wrangler deploy...');
  execSync('npx wrangler deploy', { stdio: 'inherit' });
} catch (error) {
  console.error('❌ Deploy failed.');
  process.exit(1);
}

// Post-deploy verification. Separate try//catch from the deploy above on purpose:
// the upload already succeeded, so a failing probe is a "verify this" signal, not
// a failed deploy, and saying so is the difference between rolling back and
// re-probing. The checker retries internally, since a fresh deploy takes a few
// seconds to propagate and the previous version answers until it does.
try {
  console.log('\n🔎 Verifying the deployment...');
  execSync('node ../auxiliary/check_live_status.mjs', { stdio: 'inherit' });
} catch {
  console.error('\n⚠️  Deploy SUCCEEDED but post-deploy checks did not pass.');
  console.error('   Re-run `npm run status` — a just-deployed Worker can still be propagating.');
  process.exit(2);
}
