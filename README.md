# Simple Reinforcement Learning

This demonstrates reinforcement learning. Specifically, it
uses [Q-learning](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node65.html)
to move a player (`@`) around a fixed maze and avoid traps (`^`) while getting
treasure (`$`) as fast as possible.

There are three ways to run the grid.py program:

1. `grid.py --interactive`: Use the arrow keys to walk around the maze. The
   episode ends when you reach a trap or the treasure. Press space to restart or
   Q to quit. No learning happens in this mode.
1. `grid.py --q`: An &epsilon;-greedy Q-learner repeatedly runs the maze. The
   parameters are not tuned to learn quickly. Over the course of several minutes
   the player first learns to avoid spikes, then reach the treasure, and
   eventually reach the treasure in the minimum number of steps. Learning is not
   saved between runs.
1. `grid.py --test`: Run the unit tests.

Here are some ideas in ways to extend grid.py. These are increasingly difficult.
Some early steps may be useful for later steps.

* Try different mazes and see how features like long hallways affect learning.
* Change the learning rate &alpha; and future reward discount &gamma; to try to
  improve the effectiveness of the learner.
* Display summaries of results. For example, graph the distribution of rewards
  (Simulation.score) over repeated runs. It may be useful to separate the game
  simulation loop and the display so that you can simulate faster.
* Implement TD(&lambda;)
  and [eligibility traces](https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node75.html). How do features like long hallways affect learning now?
* Q-learning is an "off-policy" learning algorithm, which means the policy
  controlling the player and the policy being learned can be different. Adapt
  the HumanPlayer to permit a learner to learn by observing a human play.
* Save policies to disk and load them later. This will let you checkpoint and
  restart learning.
* Implement a maze generator. The state space will be too large for a QTable to
  be useful, so implement a value function approximator such as a neural
  network. The QTable memorizes the fixed map; with multiple maps you will need
  to feed the maze as input to the neural network so it can "see" the map
  instead of memorizing it.
* Connect the learner to an actual roguelike such
  as [NetHack](http://www.nethack.org/) to speed run for dungeon depth.
* Change the problem from one with discrete states (a grid) and actions (up,
  down, left, right) to continuous states (the player is at fine-grained x-y
  coordinates with a certain heading and velocity) and actions (acceleration and
  turning.) How will you relate states to each other in a continuous space?
