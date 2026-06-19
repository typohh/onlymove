#!/bin/bash
# Production chain (2026-06-12): M0b is now the zero-flag default of train_rating_v2.
# fetch fresh attempts -> full M0b train into ratings_v2/ (live backend backed up first)
# -> EGF curve fit -> population constants -> package for PWA. After this: cd pwa && npm run dev.
set -e
cd /home/ootakamoku/Repositories/onlymove
P=./.venv/bin/python3
[ -d ratings_v2_live_backup ] || cp -r ratings_v2 ratings_v2_live_backup
$P -m pipeline.fetch_database --rebuild-sqlite   > experiments_4pl/prod_1_fetch.log 2>&1
$P -m auxiliary.rating_dev.train_rating_v2_research                   > experiments_4pl/prod_2_train.log 2>&1
$P -m pipeline.map_rating_v2_to_egf              > experiments_4pl/prod_3_egf.log 2>&1
$P -m pipeline.fit_population_constants          > experiments_4pl/prod_4_popconst.log 2>&1
$P -m pipeline.package_for_pwa                   > experiments_4pl/prod_5_package.log 2>&1
echo PRODUCTION_CHAIN_DONE
