## **Task List: Go Problem Generation**
- for each position we consider for a tsumego, its only valid if there is only one correct next move, detect this by running widerootnoise with say 9*9 playouts. 
  - if multiple moves are valid, keep playing the second best moves, and opponent resonds with their best move, until either the original best move is played, or the best move is the only reamining move with above 50% winrate before and after, this is our new corrected tsumego.
- test how internally consistent human ranks order is at predicting correct move trough out the game, create an .md file on the findings.


