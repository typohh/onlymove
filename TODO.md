Work on one task or sub task at a time from TODO list, testing it as needed, then moving it to COMPLETED, after which you should move to the next TODO issue if any remaining. If a task turns out to be complex, split it into multiple new TODOs

### ** TODO **
- during game difficulty evaluation, if katago evaluation is that winrate is below 1% we can stop evaluatoin and just accept that human lost.
- during game difficulty evaluation, if a weaker player manages to get above 50% winrate, lets just assume all stornger players will as well.

### ** COMPLETED **

- update statistics
  - number of tsumego creations attempts.
  - avg score adjustment exchange per attempted tsumego creation.
  - avg move eliminations exchange per attempted tsumego creation.
  - change games per minute into to tsumegos per minute.
  - add statistics on each reason why tsumego creation fails or is rejected during normal operations.
- include running time in statistics print.
- change from determing rank via likeliest move, to where probability of playing move exceeds 50%
- evalute also what rank is required to have 50% chance of winning the game (against full strength kata), not just the next move. the human network is move is based on weighted random sample from policy network.
- use preaz_20k in addition to rank_20k.
- start each game with 4 random moves, and 2 moves to try to get the game back to even.
- when removing alterantive moves, instead of white playing the perfect response, have it play a response that aims at the targets we have (same as in score adjustment)
- remove all worker print statements, they only collect statistics.
- track various statistics and print them in the main thread at regular intervals.
  - average score and winrate deviation from even.
  - average number of score adjustments, and move elimiations.
  - how often we fail for each reason.

