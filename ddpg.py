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


STATE_FEATURE_SIZE = 6
EMBEDDING_DIMENSION = 4


def neural_network(state, world_size_h_w):
  '''Makes a neural network for examining worlds.

  This network features:
  - An embedding layer.
  - Two layers of convolutions with max pooling.
  - A fully connected layer. The embedding layer is resupplied.
  - Output logits and softmax for up, right, down, left.

  Args:
    state: The map, a tensor of height * width.
    world_size_h_w: A tuple, the height and width of the world.

  Returns:
    logits: The -1 * 4 logits output tensor for up, right, down, left.
    softmax: The softmax for the same.
  '''
  h, w = world_size_h_w
  embedding = tf.Variable(
    tf.truncated_normal(shape=[STATE_FEATURE_SIZE, EMBEDDING_DIMENSION],
                        stddev=0.1))
  embedded = tf.nn.embedding_lookup(
    embedding,
    tf.reshape(state, [-1, h * w]))
  embedded = tf.reshape(embedded, [-1, h, w, EMBEDDING_DIMENSION])
  # First convolutional and pooling layer.
  conv_1_out_channels = 3
  conv_1_filter = tf.Variable(
      tf.truncated_normal((5, 5, EMBEDDING_DIMENSION, conv_1_out_channels),
                          stddev=0.1))
  conv_1_bias = tf.Variable(tf.constant(0.1, shape=[conv_1_out_channels]))
  conv_1 = tf.nn.conv2d(embedded, conv_1_filter, [1, 1, 1, 1], 'SAME')
  conv_1 = tf.nn.relu(conv_1 + conv_1_bias)
  pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')
  # Second convolution and pooling layer.
  conv_2_out_channels = 7
  conv_2_filter = tf.Variable(
    tf.truncated_normal((5, 5, conv_1_out_channels, conv_2_out_channels),
                        stddev=0.1))
  conv_2_bias = tf.Variable(tf.constant(0.1, shape=[conv_2_out_channels]))
  conv_2 = tf.nn.conv2d(pool_1, conv_2_filter, [1, 1, 1, 1], 'SAME')
  conv_2 = tf.nn.relu(conv_2 + conv_2_bias)
  pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')
  # Resupply the embedding layer.
  shrunk_w = (w+3)//4
  shrunk_h = (h+3)//4
  pool_2 = tf.reshape(pool_2, [-1, shrunk_h * shrunk_w * conv_2_out_channels])
  embedded = tf.reshape(embedded, [-1, h * w * EMBEDDING_DIMENSION])
  fc_input = tf.concat([pool_2, embedded], axis=1)
  fc_input_size = (shrunk_h * shrunk_w * conv_2_out_channels +
                   h * w * EMBEDDING_DIMENSION)
  # Fully connected layer.
  fc_output_size = 100
  w2 = tf.Variable(
    tf.truncated_normal([fc_input_size, fc_output_size], stddev=0.1))
  b2 = tf.Variable(tf.constant(0.1, shape=[fc_output_size]))
  fc_pre = tf.matmul(fc_input, w2) + b2
  fc = tf.nn.relu(fc_pre)
  # Softmax.
  action_size = 4
  w3 = tf.Variable(
    tf.truncated_normal([fc_output_size, action_size], stddev=0.1))
  b3 = tf.Variable(tf.constant(0.1, shape=[action_size]))
  logits_pre = tf.matmul(fc, w3) + b3
  logits = tf.sigmoid(logits_pre)
  softmax = tf.nn.softmax(logits)
  return logits, softmax


class TestNeuralNetwork(unittest.TestCase):
  def test(self):
    # This first because it is useful to have the world dimensions.
    w = world.World.parse('''
@.#$
^...
##^^
####
####
####''')

    # Make the graph.
    g = tf.Graph()
    with g.as_default():
      world_in = tf.placeholder(tf.int32, shape=[None, w.h, w.w])
      _, out = neural_network(world_in, (w.h, w.w))

    # Start a simulation.
    sim = simulation.Simulation(w)
    raw = simulation_to_array(sim)
    with tf.Session(graph=g) as session:
      # Could use tf.argmax for convenience.
      session.run(tf.global_variables_initializer())
      act = session.run(out, feed_dict={world_in: [raw]})
    print(act)

# TODO(dominicc):
# - implement the critic network
# - implement the actor network
# - implement target networks and snapshot updates
# - implement a training step for the critic network
# - implement a training step for the actor network
# - implement a loop which drives learning
# - examine whether # gets a similar embedding to ' '

if __name__ == '__main__':
  unittest.main()
