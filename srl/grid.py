#!/usr/bin/env python3

# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO:
# - Decouple learning from the animated display
# - Implement random maps and approximate value functions

import collections
import curses
import random
import sys
import time

from srl import movement
from srl import simulation
from srl import world


# There is also an interactive version of the game. These are keycodes
# for interacting with it.
KEY_Q = ord('q')
KEY_ESC = 27
KEY_SPACE = ord(' ')
KEY_UP = 259
KEY_DOWN = 258
KEY_LEFT = 260
KEY_RIGHT = 261
KEY_ACTION_MAP = {
  KEY_UP: movement.ACTION_UP,
  KEY_DOWN: movement.ACTION_DOWN,
  KEY_LEFT: movement.ACTION_LEFT,
  KEY_RIGHT: movement.ACTION_RIGHT
}
QUIT_KEYS = set([KEY_Q, KEY_ESC])


class Game(object):
  '''A simulation that uses curses.'''
  def __init__(self, the_world, driver):
    '''Creates a new game in world where driver will interact with the game.'''
    self._world = the_world
    self._sim = simulation.Simulation(self._world)
    self._driver = driver

  def start(self):
    '''Sets up and starts the game and runs it until the driver quits.'''
    curses.initscr()
    curses.wrapper(self._loop)

  # The game loop.
  def _loop(self, window):
    while not self._driver.should_quit:
      # Paint
      self._draw(window)
      window.addstr(self._world.h, 0, 'Score: %d' % self._sim.score)
      window.move(self._sim.y, self._sim.x)
      window.refresh()

      # Get input, etc.
      self._driver.interact(self._sim, window)

  # Paints the window.
  def _draw(self, window):
    window.erase()
    # Draw the environment
    for y, line in enumerate(self._world._lines):
      window.addstr(y, 0, line)
    # Draw the player
    window.addstr(self._sim.y, self._sim.x, '@')


class Player(object):
  '''A Player provides input to the game as a simulation evolves.'''
  def interact(self, sim, window):
    # All players have the same interface
    # pylint: disable=unused-argument
    pass


class HumanPlayer(Player):
  '''A game driver that reads input from the keyboard.'''
  def __init__(self):
    super(HumanPlayer, self).__init__()
    self._ch = 0

  @property
  def should_quit(self):
    return self._ch in QUIT_KEYS

  def interact(self, sim, window):
    self._ch = window.getch()
    if self._ch in KEY_ACTION_MAP and not sim.in_terminal_state:
      sim.act(KEY_ACTION_MAP[self._ch])
    elif self._ch == KEY_SPACE and sim.in_terminal_state:
      sim.reset()


class MachinePlayer(Player):
  '''A game driver which applies a policy, observed by a learner.
The learner can adjust the policy.'''
  def __init__(self, policy, learner):
    super(MachinePlayer, self).__init__()
    self._policy = policy
    self._learner = learner

  @property
  def should_quit(self):
    return False

  def interact(self, sim, window):
    super(MachinePlayer, self).interact(sim, window)
    if sim.in_terminal_state:
      time.sleep(1)
      sim.reset()
    else:
      old_state = sim.state
      action = self._policy.pick_action(sim.state)
      reward = sim.act(action)
      self._learner.observe(old_state, action, reward, sim.state)
      time.sleep(0.05)


class StubFailure(Exception):
  pass


class StubWindow(object):
  '''A no-op implementation of the game display.'''
  def addstr(self, y, x, s):
    pass

  def erase(self):
    pass

  def getch(self):
    raise StubFailure('"getch" not implemented; use a mock')

  def move(self, y, x):
    pass

  def refresh(self):
    pass


class StubLearner(object):
  '''Plugs in as a learner but doesn't update anything.'''
  def observe(self, old_state, action, reward, new_state):
    pass


class RandomPolicy(object):
  '''A policy which picks actions at random.'''
  def pick_action(self, _):
    return random.choice(movement.ALL_ACTIONS)


class EpsilonPolicy(object):
  '''Pursues policy A, but uses policy B with probability epsilon.

Be careful when using a learned function for one of these policies;
the epsilon policy needs an off-policy learner.
  '''
  def __init__(self, policy_a, policy_b, epsilon):
    self._policy_a = policy_a
    self._policy_b = policy_b
    self._epsilon = epsilon

  def pick_action(self, state):
    if random.random() < self._epsilon:
      return self._policy_b.pick_action(state)
    else:
      return self._policy_a.pick_action(state)


class QTable(object):
  '''An approximation of the Q function based on a look-up table.
  As such it is only appropriate for discrete state-action spaces.'''
  def __init__(self, init_reward = 0):
    self._table = collections.defaultdict(lambda: init_reward)

  def get(self, state, action):
    return self._table[(state, action)]

  def set(self, state, action, value):
    self._table[(state, action)] = value

  def best(self, state):
    '''Gets the best predicted action and its value for |state|.'''
    best_value = -1e20
    best_action = None
    for action in movement.ALL_ACTIONS:
      value = self.get(state, action)
      if value > best_value:
        best_action, best_value = action, value
    return best_action, best_value


class GreedyQ(object):
  '''A policy which chooses the action with the highest reward estimate.'''
  def __init__(self, q):
    self._q = q

  @property
  def should_quit(self):
    return False

  def pick_action(self, state):
    return self._q.best(state)[0]


class QLearner(object):
  '''An off-policy learner which updates a QTable.'''
  def __init__(self, q, learning_rate, discount_rate):
    self._q = q
    self._alpha = learning_rate
    self._gamma = discount_rate

  def observe(self, old_state, action, reward, new_state):
    prev = self._q.get(old_state, action)
    self._q.set(old_state, action, prev + self._alpha * (
      reward + self._gamma * self._q.best(new_state)[1] - prev))


def main():
  if '--interactive' in sys.argv:
    player = HumanPlayer()
  elif '--q' in sys.argv:
    q = QTable()
    learner = QLearner(q, 0.05, 0.1)
    policy = EpsilonPolicy(GreedyQ(q), RandomPolicy(), 0.01)
    player = MachinePlayer(policy, learner)
  else:
    print('use --test, --interactive or --q')
    sys.exit(1)

  w = None
  if '--random' in sys.argv:
    w = world.Generator(25, 15).generate()
  else:
    w = world.World.parse('''\
  ########
  #..#...#
  #.@#.$.#
  #.##^^.#
  #......#
  ########
  ''')

  game = Game(w, player)
  game.start()


if __name__ == '__main__':
  main()
