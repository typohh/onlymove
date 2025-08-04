#!/bin/sh

rm -rf dist
mkdir dist
cd dist
cp -r ../* . 2>/dev/null

# Update APP_VERSION in app.js with current date and time in UTC
CURRENT_DATETIME=$(date -u +"%Y-%m-%d %H:%M")
sed -i "s/const APP_VERSION = \".*\";/const APP_VERSION = \"$CURRENT_DATETIME\";/" app.js

# Step 1: Search for references to app.js inside .html and .js files
REF=$(grep -hoE 'app\.js' *.html *.js 2>/dev/null | sort | uniq | head -n 1)

if [ -z "$REF" ]; then
  echo "No app.js reference found."
  exit 1
fi

# Step 2: Compute hash of app.js
if [ ! -f app.js ]; then
  echo "app.js not found."
  exit 1
fi

YYYY=$(sha256sum app.js | cut -c1-8)

# Step 3: Replace all occurrences of app.js with appYYYY.js and count replacements
REPLACEMENTS=0
for f in *.html *.js; do
  [ "$f" = "app.js" ] && continue
  [ -f "$f" ] || continue
  COUNT=$(sed -n -E "s/app\.js/app$YYYY.js/gp" "$f" | wc -l)
  if [ "$COUNT" -gt 0 ]; then
    sed -i -E "s/app\.js/app$YYYY.js/g" "$f"
    REPLACEMENTS=$((REPLACEMENTS + COUNT))
  fi
done

# Step 5: Copy app.js to appYYYY.js
mv app.js "app$YYYY.js"

echo "Replaced $REPLACEMENTS app.js references with app$YYYY.js"