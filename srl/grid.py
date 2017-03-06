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
# - Implement approximate value functions

import argparse
import collections
import curses
import random
import sys
import tensorflow as tf
import time

from srl import context
from srl import dqn
from srl import movement
from srl import player
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
  def __init__(self, ctx, generator, driver):
    '''Creates a new game in world where driver will interact with the game.'''
    self._context = ctx
    self._sim = simulation.Simulation(generator)
    self._driver = driver
    self._wins = 0
    self._losses = 0
    self._was_in_terminal_state = False

  # The game loop.
  def step(self):
    # Paint
    self._draw(self._context.window)
    # Get input, etc.
    self._driver.interact(self._context, self._sim)
    if self._sim.in_terminal_state and not self._was_in_terminal_state:
      self._context.window.clear()
      if self._sim.score < 0:
        self._losses += 1
      else:
        self._wins += 1
    self._was_in_terminal_state = self._sim.in_terminal_state

  # Paints the window.
  def _draw(self, window):
    window.erase()
    # Draw the environment
    for y, line in enumerate(self._sim.world._lines):
      window.addstr(y, 0, line)
    # Draw the player
    window.addstr(self._sim.y, self._sim.x, '@')
    # Draw status
    window.addstr(self._sim.world.h, 0,
                  'W/L: %d/%d   Score: %d' %
                  (self._wins, self._losses, self._sim.score))
    window.move(self._sim.y, self._sim.x)
    # TODO: Add a display so multiple things can contribute to the output.
    window.refresh()


class HumanPlayer(player.Player):
  '''A game driver that reads input from the keyboard.'''
  def __init__(self):
    super(HumanPlayer, self).__init__()
    self._ch = 0

  def interact(self, ctx, sim):
    self._ch = ctx.window.getch()
    if self._ch in KEY_ACTION_MAP and not sim.in_terminal_state:
      sim.act(KEY_ACTION_MAP[self._ch])
    elif self._ch == KEY_SPACE and sim.in_terminal_state:
      sim.reset()
    elif self._ch in QUIT_KEYS:
      ctx.run_loop.post_quit()


class MachinePlayer(player.Player):
  '''A game driver which applies a policy, observed by a learner.

  The learner can adjust the policy.
  '''

  def __init__(self, policy, learner):
    super(MachinePlayer, self).__init__()
    self._policy = policy
    self._learner = learner

  def interact(self, ctx, sim):
    super(MachinePlayer, self).interact(ctx, sim)
    if sim.in_terminal_state:
      sim.reset()
    else:
      old_state = sim.state
      action = self._policy.pick_action(sim.state)
      reward = sim.act(action)
      self._learner.observe(old_state, action, reward, sim.state)


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
  parser = argparse.ArgumentParser(description='Simple Reinforcement Learning.')
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--interactive', action='store_true',
                     help='use the keyboard arrow keys to play')
  group.add_argument('--q', action='store_true',
                     help='play automatically with Q-learning')
  group.add_argument('--dqn', action='store_true',
                     help='play automatically with a deep-Q network')
  parser.add_argument('--random', action='store_true',
                      help='generate a random map')

  args = parser.parse_args()

  ctx = context.Context()

  if args.random:
    generator = world.Generator(15, 12)
  else:
    generator = world.Static(world.World.parse('''\
  ########
  #..#...#
  #.@#.$.#
  #.##^^.#
  #......#
  ########
  '''))

  if args.interactive:
    player = HumanPlayer()
  elif args.q:
    q = QTable()
    learner = QLearner(q, 0.05, 0.1)
    policy = EpsilonPolicy(GreedyQ(q), RandomPolicy(), 0.01)
    player = MachinePlayer(policy, learner)
  elif args.dqn:
    g = tf.Graph()
    s = tf.Session(graph=g)
    player = dqn.DeepQPlayer(g, s, generator.size)
    with g.as_default():
      init = tf.global_variables_initializer()
      s.run(init)
  else:
    sys.exit(1)

  is_automatic = args.q or args.dqn
  if is_automatic:
    # Slow the game down to make it fun? to watch.
    ctx.run_loop.post_task(lambda: time.sleep(0.1), repeat=True)

  game = Game(ctx, generator, player)
  ctx.run_loop.post_task(game.step, repeat=True)

  ctx.start()


if __name__ == '__main__':
  main()
