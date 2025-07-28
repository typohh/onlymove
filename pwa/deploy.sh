#!/bin/sh

rm -rf dist
mkdir dist
cd dist
cp -r ../* .

# Update APP_VERSION in app.js with current date and time
CURRENT_DATETIME=$(date +"%Y-%m-%d %H:%M")
sed -i "s/const APP_VERSION = \".*\";/const APP_VERSION = \"$CURRENT_DATETIME\";/" app.js

# Step 1: Search for references to app.js or appXXXX.js inside .html and .js files
REF=$(grep -hoE 'app([0-9a-zA-Z]{4,})?\.js' *.html *.js 2>/dev/null | sort | uniq | head -n 1)

if [ -z "$REF" ]; then
  echo "No app.js or appXXXX.js reference found."
  exit 1
fi

# Extract XXXX if present
XXXX=$(echo "$REF" | sed -E 's/^app([0-9a-zA-Z]{4,})?\.js$/\1/')

# Step 2: Compute hash of app.js
if [ ! -f app.js ]; then
  echo "app.js not found."
  exit 1
fi

YYYY=$(sha256sum app.js | cut -c1-8)

# Step 3: Replace all occurrences of app.js or appXXXX.js with appYYYY.js
for f in *.html *.js; do
  [ "$f" = "app.js" ] && continue
  [ -f "$f" ] || continue
  sed -i -E "s/app([0-9a-zA-Z]{4,})?\.js/app$YYYY.js/g" "$f"
done

# Step 5: Copy app.js to appYYYY.js
mv app.js "app$YYYY.js"

echo "Replaced all app.js/appXXXX.js references with app$YYYY.js"