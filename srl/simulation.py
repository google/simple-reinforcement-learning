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

import numpy as np

from srl import movement


class Simulation(object):
  '''Tracks the player in a world and implements the rules and rewards.

  score is the cumulative score of the player in this run of the
  simulation.
  '''
  def __init__(self, generator):
    self._generator = generator

    # Initialized by reset()
    self.state = None
    self.world = None
    self.steps = None

    self.reset()

  def reset(self):
    '''Resets the simulation to the initial state.'''
    self.world = self._generator.generate()
    self.state = self.world.init_state
    self.score = 0
    self.steps = 0

  @property
  def in_terminal_state(self):
    '''Whether the simulation is in a terminal state (stopped.)'''
    return self.world.at(self.state) in ['^', '$'] or self.steps > 300

  @property
  def x(self):
    '''The x coordinate of the player.'''
    return self.state[0]

  @property
  def y(self):
    '''The y coordinate of the player.'''
    return self.state[1]

  def act(self, action):
    '''Performs action and returns the reward from that step.'''
    reward = -1

    self.steps += 1
    # TODO: Encode the move limit it one place.
    if self.steps > 295:
      reward -= 1000  # Too slow penalty.

    delta = movement.MOVEMENT[action]
    new_state = self.x + delta[0], self.y + delta[1]

    if self._valid_move(new_state):
      ch = self.world.at(new_state)
      if ch == '^':
        reward = -10000
      elif ch == '$':
        reward = 10000
      else:
        self.world.update(new_state, ',')
      self.state = new_state
    else:
      # Penalty for hitting the walls.
      reward -= 3

    self.score += reward
    return reward

  def _valid_move(self, new_state):
    '''Gets whether movement to new_state is a valid move.'''
    new_x, new_y = new_state
    # TODO: Could check that there's no teleportation cheating.
    return (0 <= new_x and new_x < self.world.w and
            0 <= new_y and new_y < self.world.h and
            self.world.at(new_state) in ['.', ',', '^', '$'])

  def to_array(self):
    '''Converts the state of a simulation to numpy ndarray.

    The returned array has numpy.int8 units with the following mapping.
    This mapping has no special meaning because these indices are fed
    into an embedding layer.
        ' ' -> 0
        '#' -> 1
        '$' -> 2
        '.' -> 3
        ',' -> 4
        '@' -> 5
        '^' -> 6
    Args:
      sim: A simulation.Simulation to externalize the state of.
    Returns:
      The world map and player position represented as an numpy ndarray.
    '''
    key = ' #$.,@^'
    w = np.empty(shape=(self.world.h, self.world.w), dtype=np.int8)
    for v in range(self.world.h):
      for u in range(self.world.w):
        w[v, u] = key.index(self.world.at((u, v)))
    w[self.y, self.x] = key.index('@')
    return w
