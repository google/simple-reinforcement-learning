#!/usr/bin/env python3

# Copyright 2017 Google Inc.
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

# This is modelled on [Deep Deterministic Policy Gradients in
# Tensorflow](http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)
# with some changes to use a discrete action space.

import numpy as np
import random
import tensorflow as tf
import unittest

import simulation
import world


class ReplayBuffer(object):
  '''Stores snapshots of states, actions and rewards.

  Sampling from a replay buffer reduces bias from the correlation
  between successive states.
  '''

  def __init__(self, max_size):
    '''Creates a ReplayBuffer which can hold up to max_size items.'''
    assert max_size > 0
    self._max_size = max_size
    self._buffer = []

  @property
  def size(self):
    '''The number of experiences entered into the buffer.'''
    return len(self._buffer)

  def add(self, old_state, action, reward, new_state, is_terminal):
    '''Adds one item to the experience buffer.

    Args:
      old_state: The np.array of state data.
      action: The one-hot four element np.array indicating which
          action was taken.
      reward: The float reward from the environment for taking action.
      new_state: The np.array of state data after action was taken.
          This should have the same shape as old_state.
      is_terminal: The bool indicating whether new_state is a terminal state.
    '''
    assert old_state.shape == new_state.shape
    assert old_state.dtype == new_state.dtype == np.int8
    assert action.shape == (4,)
    assert action.dtype == np.float32
    assert type(reward) == float
    assert type(is_terminal) == bool
    sample = (old_state, action, reward, new_state, is_terminal)
    if self.size < self._max_size:
      self._buffer.append(sample)
    else:
      self._buffer[random.randint(self._max_size)] = sample

  def sample_batch(self, batch_size):
    '''Samples up to batch_size items from the replay buffer.

    Args:
      batch_size: The maximum number of samples to return. Fewer may
          be returned. The outer dimension of the returned np.arrays
          will be up to this length.

    Returns:
      A tuple of 5 np.arrays: the original state, the action, the
      reward, the next state, and whether that state is a terminal
      state.
    '''
    assert self.size > 0
    buffer = random.sample(self._buffer, min(self.size, batch_size))
    old_state = np.array(map(lambda x: x[0], buffer))
    action = np.array(map(lambda x: x[1], buffer))
    reward = np.array(map(lambda x: x[2], buffer))
    new_state = np.array(map(lambda x: x[3], buffer))
    terminal = np.array(map(lambda x: x[4], buffer))
    return old_state, action, reward, new_state, terminal


def simulation_to_array(sim):
  '''Converts the state of a simulation to numpy ndarray.

  The returned array has numpy.int8 units with the following mapping.
  This mapping has no special meaning because these indices are fed
  into an embedding layer.

      ' ' -> 0
      '#' -> 1
      '$' -> 2
      '.' -> 3
      '@' -> 4
      '^' -> 5

  Args:
    sim: A simulation.Simulation to externalize the state of.

  Returns:
    The world map and player position represented as an numpy ndarray.
  '''
  key = ' #$.@^'
  w = np.empty(shape=(sim.world.h, sim.world.w), dtype=np.int8)
  for v in range(sim.world.h):
    for u in range(sim.world.w):
      w[v, u] = key.index(sim.world.at((u, v)))
  w[sim.y, sim.x] = key.index('@')
  return w


class TestSimulationToArray(unittest.TestCase):
  def test(self):
    w = world.World.parse('$.@^#')
    sim = simulation.Simulation(w)
    self.assertTrue(
      (np.array([[2, 3, 4, 5, 1]], dtype=np.int8) == simulation_to_array(sim))
      .all())

# TODO(dominicc):
# - implement a mapping from simulation state to np.array
# - implement a convolutional neural network over that state
# - implement the critic network
# - implement the actor network
# - implement target networks and snapshot updates
# - implement a training step for the critic network
# - implement a training step for the actor network
# - implement a loop which drives learning
# - examine whether # gets a similar embedding to ' '

if __name__ == '__main__':
  unittest.main()
