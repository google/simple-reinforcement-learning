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

import movement
import world


class Simulation(object):
  '''Tracks the player in a world and implements the rules and rewards.
score is the cumulative score of the player in this run of the simulation.'''
  def __init__(self, world):
    '''Creates a new simulation in world.'''
    self._world = world
    self.reset()

  def reset(self):
    '''Resets the simulation to the initial state.'''
    self.state = self._world.init_state
    self.score = 0

  @property
  def in_terminal_state(self):
    '''Whether the simulation is in a terminal state (stopped.)'''
    return self._world.at(self.state) in ['^', '$']

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

    delta = movement.MOVEMENT[action]
    new_state = self.x + delta[0], self.y + delta[1]

    if self._valid_move(new_state):
      ch = self._world.at(new_state)
      if ch == '^':
        reward = -10000
      elif ch == '$':
        reward = 10000
      self.state = new_state
    else:
      # Penalty for hitting the walls.
      reward -= 5

    self.score += reward
    return reward

  def _valid_move(self, new_state):
    '''Gets whether movement to new_state is a valid move.'''
    new_x, new_y = new_state
    # TODO: Could check that there's no teleportation cheating.
    return (0 <= new_x and new_x < self._world.w and
            0 <= new_y and new_y < self._world.h and
            self._world.at(new_state) in ['.', '^', '$'])
