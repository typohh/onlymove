We are working on a PWA. Its about showing users pre designed tsumegos (context baduk/weiqi/go). The relevant entry points are @index.html and @app.js. We use wGo.js library to handle game state, board graphics and parse SGF. DO NOT READ metadata.json, YOU ARE TO NEVER EVER READ metadata.json. you READING metadata.json is illegal and has severe consequences. you can deduce its structure from the code and use it in code.

TODO:
- calculate exponential decay on lost game frequency. as it deviates from expected 1/6, apply a scaling factor to rank adustments.
- flash card memorization. track tsumegos attempted, for each apply a correction factor for probability. if fails drop it to very high, if success drop significantly.