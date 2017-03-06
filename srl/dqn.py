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

'''A deep Q network-based maze runner in TensorFlow.

This is based on these papers: https://arxiv.org/pdf/1312.5602v1.pdf and
https://arxiv.org/pdf/1509.06461.pdf
'''

import collections
import datetime
import itertools
import numpy as np
import random
import sys
import tensorflow as tf

from srl import movement
from srl import player
from srl import world


_EMBEDDING_SIZE = 3
_FEATURE_SIZE = len(world.VALID_CHARS)
_ACTION_SIZE = len(movement.ALL_ACTIONS)


class ReplayBuffer(object):
  def __init__(self, max_size):
    self._max_size = max_size
    self._positive = collections.deque()
    self._negative = collections.deque()

  @property
  def size(self):
    return len(self._positive) + len(self._negative)

  def add(self, score, experience):
    if self.size >= self._max_size:
      if len(self._positive) > len(self._negative):
        self._positive.pop()
      else:
        self._negative.pop()
    if score > 0:
      self._positive.append(experience)
    else:
      self._negative.append(experience)

  def sample(self):
    assert self.size > 0
    source = None
    if len(self._negative) == 0:
      source = self._positive
    elif len(self._positive) == 0:
      source = self._negative
    # This sample is unbalanced to weight positive and negative
    # experiences equally. In practice there are more spikes than
    # rewards and more negative experiences.
    elif random.randint(0, 1) == 0:
      source = self._positive
    else:
      source = self._negative
    return source[random.randint(0, len(source) - 1)]


class DeepQNetwork(object):
  '''A deep Q-network.

  This has a number of properties for operating on the network with
  TensorFlow:

  Placeholders:
    state: Feed a world-sized array for selecting an action.
        [-1, h, w] int32

  Operations:
    action_out: Produces an action, fed state.

  For trainable networks (one provided a target):

  Placeholders:
    action_in: Feed the responsible actions for the rewards.
        [-1, 1] int32 0 <= x < len(movement.ALL_ACTIONS)
    reward: The reward of each given action. [-1, 1] float32
    is_terminal: Whether the next state is a terminal state.
        [-1, 1] bool.
    next_state: The state arrived at after action. [-1, h, w] int32.

  Operations:
    update: Given batches of experience, train the network.
    update_target: Copy trainable variables to the target network.
    loss: Training loss.
    summary: Merged summaries.
  '''

  def __init__(self, name, graph, world_size_h_w, target=None):
    '''Creates a DeepQNetwork.

    Args:
      name: The name of this network. TF variables are in a scope with
          this name.
      graph: The TF graph to build operations in.
      world_size_h_w: The size of the world, height by width.
      target: The target network to use for training.
    '''
    h, w = world_size_h_w
    self._h, self._w = h, w
    trainable = target != None
    with graph.as_default():
      variable_start_index = len(tf.model_variables())

      initializer = tf.contrib.layers.xavier_initializer()
      with tf.variable_scope(name) as var_scope:
        self.state = tf.placeholder(tf.int32, shape=[None, h, w], name='state')

        # Input embedding
        embedding = tf.get_variable(
            'embedding', shape=[_FEATURE_SIZE, _EMBEDDING_SIZE],
            initializer=initializer)
        tf.contrib.framework.add_model_variable(embedding)
        embedding_lookup = tf.nn.embedding_lookup(
            embedding, tf.reshape(self.state, [-1, h * w]),
            name='embedding_lookup')
        embedding_lookup = tf.reshape(embedding_lookup,
                                      [-1, h, w, _EMBEDDING_SIZE])

        # First convolution.
        conv_1_out_channels = 27
        conv_1 = tf.contrib.layers.conv2d(
            trainable=trainable,
            inputs=embedding_lookup,
            num_outputs=conv_1_out_channels,
            kernel_size=[5, 5],
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            # TODO: What's a good initializer for biases? Below too.
            biases_initializer=initializer)

        shrunk_h = h
        shrunk_w = w

        # Second convolution.
        conv_2_out_channels = 50
        conv_2_stride = 2
        conv_2 = tf.contrib.layers.conv2d(
            trainable=trainable,
            inputs=conv_1,
            num_outputs=conv_2_out_channels,
            kernel_size=[5, 5],
            stride=conv_2_stride,
            padding='SAME',
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            biases_initializer=initializer)
        shrunk_h = (h + conv_2_stride - 1) // conv_2_stride
        shrunk_w = (w + conv_2_stride - 1) // conv_2_stride

        # Third convolution.
        conv_3_out_channels = 100
        conv_3_stride = 2
        conv_3 = tf.contrib.layers.conv2d(
            trainable=trainable,
            inputs=conv_2,
            num_outputs=conv_3_out_channels,
            kernel_size=[5, 5],
            stride=conv_3_stride,
            padding='SAME',
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            biases_initializer=initializer)
        shrunk_h = (shrunk_h + conv_3_stride - 1) // conv_3_stride
        shrunk_w = (shrunk_w + conv_3_stride - 1) // conv_3_stride

        # Resupply the input.
        resupply = tf.concat([
              tf.reshape(conv_3,
                         [-1, shrunk_h * shrunk_w * conv_3_out_channels]),
              tf.reshape(embedding_lookup, [-1, h * w * _EMBEDDING_SIZE])
            ], 1, name='resupply')

        # First fully connected layer.
        connected_1 = tf.contrib.layers.fully_connected(
            trainable=trainable,
            inputs=resupply,
            num_outputs=16 * (h+w),
            activation_fn=tf.nn.relu,
            weights_initializer=initializer,
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            biases_initializer=initializer)

        # Second fully connected layer, steps down.
        connected_2 = tf.contrib.layers.fully_connected(
            trainable=trainable,
            inputs=connected_1,
            num_outputs=8 * (h+w),
            activation_fn=tf.nn.relu,
            weights_initializer=initializer,
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            biases_initializer=initializer)

        # Third fully connected layer, steps down.
        connected_3 = tf.contrib.layers.fully_connected(
            trainable=trainable,
            inputs=connected_2,
            num_outputs=16,
            activation_fn=tf.nn.relu,
            weights_initializer=initializer,
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            biases_initializer=initializer)

        # Output layer.
        self.action_value = tf.contrib.layers.fully_connected(
            trainable=trainable,
            inputs=connected_3,
            num_outputs=_ACTION_SIZE,
            activation_fn=tf.nn.relu,
            weights_initializer=initializer,
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            biases_initializer=initializer)

        self.action_out = tf.argmax(self.action_value, 1)

        self._variables = tf.model_variables()[variable_start_index:]
        assert len(self._variables) > 0

        if trainable:
          # This network is trainable.
          assert len(self._variables) == len(target._variables)

          self.next_state = target.state

          self.reward = tf.placeholder(tf.float32, shape=[None], name='reward')
          self.is_terminal = tf.placeholder(tf.bool, shape=[None],
                                            name='is_terminal')
          is_not_terminal = tf.cast(tf.logical_not(self.is_terminal),
                                    tf.float32)

          gamma = 0.97
          target_value = (self.reward + gamma * is_not_terminal *
                          tf.reduce_max(target.action_value, axis=1))

          self.action_in = tf.placeholder(tf.int32, shape=[None],
                                          name='action_in')
          action_one_hot = tf.one_hot(self.action_in, _ACTION_SIZE,
                                      dtype=tf.float32)
          policy_value = tf.reduce_sum(self.action_value * action_one_hot,
                                       axis=1)
          loss_policy = target_value - policy_value
          loss_policy *= loss_policy  # Squared error
          loss_policy = tf.reduce_mean(loss_policy, name='loss_policy')
          # TODO: Investigate whether regularization losses are sums or
          # means and consider removing the division.
          loss_regularization = (0.5 / tf.to_float(tf.shape(self.state)[0]) *
              sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                    scope=name)))
          # Encourage indecision.
          #loss_entropy = -tf.reduce_mean(
          #    self.action_softmax * (1.0 - self.action_softmax))
          # TODO: Consider removing entropy entirely.
          self.loss = loss_policy + loss_regularization # + loss_entropy

          tf.summary.scalar('loss_policy', loss_policy)
          tf.summary.scalar('loss_regularization', loss_regularization)
          # tf.summary.scalar('loss_entropy', loss_entropy)

          # TODO: Use a decaying learning rate
          optimizer = tf.train.AdamOptimizer(learning_rate=0.05)
          self.update = optimizer.minimize(self.loss)

          self.summary = tf.summary.merge_all()

          self.update_target = list(map(lambda t, s: tf.assign(t, s),
                                        target._variables, self._variables))

  def predict(self, session, states):
    '''Chooses actions for a list of states.

    Args:
      session: The TensorFlow session to run the net in.
      states: A list of simulation states which have been serialized
          to arrays.

    Returns:
      An array of actions, 0 .. 4 and an array of array of values.
    '''
    return session.run([self.action_out, self.action_value],
                       feed_dict={self.state: states})

  def train(self, session, experiences):
    '''Trains the network.

    Args:
      experiences: A list of tuples with the state, the chosen action,
          the reward, the next state, and whether the new state is a
          terminal state.

    Returns:
      Training loss summaries suitable for adding to a file for
      TensorBoard.

    '''
    size = len(experiences)
    state = np.empty([size, self._h, self._w])
    next_state = np.empty([size, self._h, self._w])
    action_in = np.empty([size])
    reward = np.empty([size])
    is_terminal = np.empty([size], dtype=np.bool)
    for i, (s, a, r, next_s, next_s_is_terminal) in enumerate(experiences):
      state[i,:,:] = s
      next_state[i,:,:] = next_s
      reward[i] = r / 1000.0  # Scale rewards
      action_in[i] = a
      is_terminal[i] = next_s_is_terminal

    summary, _ = session.run([self.summary, self.update], feed_dict={
        self.state: state,
        self.next_state: next_state,
        self.reward: reward,
        self.action_in: action_in,
        self.is_terminal: is_terminal
      })
    return summary


_EXPERIENCE_BUFFER_SIZE = 5000
_BATCH_SIZE = 100


class DeepQPlayer(player.Player):
  def __init__(self, graph, session, world_size_w_h):
    super(PolicyGradientPlayer, self).__init__()
    w, h = world_size_w_h
    self._net = PolicyGradientNetwork('net', graph, (h, w))
    self._experiences = ReplayBuffer(_EXPERIENCE_BUFFER_SIZE)
    self._experience = []
    self._session = session
    self._summary_writer = tf.summary.FileWriter(
        '/tmp/srlpg/%s' % datetime.datetime.now().isoformat())

  def interact(self, ctx, sim):
    if sim.in_terminal_state:
      self._experiences.add(sim.score, self._experience)
      self._experience = []
      summary = self._net.train(self._session,
                      [self._experiences.sample() for _ in range(_BATCH_SIZE)])
      self._summary_writer.add_summary(summary)
      sim.reset()
    else:
      state = sim.to_array()
      score = sim.score
      [[action], _] = self._net.predict(self._session, [state], [0])
      reward = sim.act(movement.ALL_ACTIONS[action])
      self._experience.append((state, score, action, reward))

  def visualize(self, ctx, sim, window):
    visitable = []
    for u in range(sim.world.w):
      for v in range(sim.world.h):
        if sim.world.at((u, v)) in '.,':
          visitable.append((v, u))
    state = sim.to_array()
    # TODO: This hard-codes indexes which must line up with Simulation.to_array.
    state[sim.state[1], sim.state[0]] = 3 # 3 == .

    states = np.empty((len(visitable), sim.world.h, sim.world.w))
    for i, (v, u) in enumerate(visitable):
      states[i,:,:] = state
      states[i,v,u] = 5  # 5 == @

    scores = [sim.score] * len(visitable)
    [moves, _] = self._net.predict(self._session, states, scores)

    window.erase()
    symbols = ['\N{UPWARDS SANS-SERIF ARROW}',
               '\N{RIGHTWARDS SANS-SERIF ARROW}',
               '\N{DOWNWARDS SANS-SERIF ARROW}',
               '\N{LEFTWARDS SANS-SERIF ARROW}']
    for i, (v, u) in enumerate(visitable):
      window.addstr(v, u, symbols[moves[i]])
