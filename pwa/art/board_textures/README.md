# Board textures

Candidate wood textures for the board background, `pwa/lib/wgo/wood_1024.jpg`. The shipped file is a plain copy of one of the `board<N>_1024.jpg` here; swapping it is `cp` plus a deploy (the service worker precache is rebuilt per app version, so installed clients pick the new one up on their next update).

`sources/` holds the photographs each texture was cut from: product photos of real kaya boards with the grid on them. `extract_grain.py <photo> <tag>` turns one into `<tag>_1024.png`: it segments the board from the backdrop, rectifies the perspective, finds the 19×19 lines from their darkness profiles and traces each one's real path, then refills them grain-aware (horizontal lines by interpolating every column along the grain, vertical lines by mirror-blending the neighbouring columns, star points column-wise), flattens the photo's lighting falloff and crops to the grid extent before scaling to 1024. A generic inpaint is not a substitute: it smooths the fine grain and the grid ghosts back as a lattice of smooth strips. Needs `opencv-python-headless` and scipy in the venv.

The grain runs vertically in every file, which is the orientation the board draws it in (player to player).

| File | Source board px | Notes |
| --- | --- | --- |
| original | – | the texture the app shipped with before any of these; no photo behind it |
| board3 | 1175 | straight masame, clean |
| board4 | 1101 | mild figure, knot near one edge |
| board6 | 1010 | strongest figure, cleanest fill (black backdrop gave the lines no halo) |
| board7 | 1077 | same board as 8 under flatter light; faint lattice residue at native resolution |
| board8 | 1156 | most dramatic figure; faint lattice residue at native resolution |
