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

'''A policy gradient-based maze runner in TensorFlow.

This is based on
https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724
'''

import collections
import datetime
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

  def add(self, final_score, experience):
    if self.size >= self._max_size:
      if len(self._positive) > len(self._negative):
        self._positive.pop()
      else:
        self._negative.pop()
    if final_score > 0:
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


class PolicyGradientNetwork(object):
  '''A policy gradient network.

  This has a number of properties for operating on the network with
  TensorFlow:

    state: Feed a world-sized array for selecting an action.
        [-1, h, w] int32
    score: The score, also used for selecting an action. [-1, 1]
        int32.
    action_out: Produces an action, fed state.

    action_in: Feed the responsible actions for the rewards.
        [-1, 1] int32 0 <= x < len(movement.ALL_ACTIONS)
    advantage: Feed the "goodness" of each given state.
        [-1, 1] float32
    update: Given batches of experience, train the network.

    loss: Training loss.
    summary: Merged summaries.
  '''

  def __init__(self, name, graph, world_size_h_w):
    '''Creates a PolicyGradientNetwork.

    Args:
      name: The name of this network. TF variables are in a scope with
          this name.
      graph: The TF graph to build operations in.
      world_size_h_w: The size of the world, height by width.
    '''
    h, w = world_size_h_w
    self._h, self._w = h, w
    with graph.as_default():
      initializer = tf.contrib.layers.xavier_initializer()
      with tf.variable_scope(name) as self.variables:
        self.score = tf.placeholder(tf.int32, shape=[None])
        score = tf.expand_dims(tf.cast(self.score, dtype=tf.float32), axis=-1)
        self.state = tf.placeholder(tf.int32, shape=[None, h, w])

        # Input embedding
        embedding = tf.get_variable(
            'embedding', shape=[_FEATURE_SIZE, _EMBEDDING_SIZE],
            initializer=initializer)
        embedding_lookup = tf.nn.embedding_lookup(
            embedding, tf.reshape(self.state, [-1, h * w]),
            name='embedding_lookup')
        embedding_lookup = tf.reshape(embedding_lookup,
                                      [-1, h, w, _EMBEDDING_SIZE])

        # First convolution.
        conv_1_out_channels = 27
        conv_1 = tf.contrib.layers.conv2d(
            trainable=True,
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
            trainable=True,
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
            trainable=True,
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

        # Resupply the input, and introduce the score, at this point.
        resupply = tf.concat([
              tf.reshape(conv_3,
                         [-1, shrunk_h * shrunk_w * conv_3_out_channels]),
              tf.reshape(embedding_lookup, [-1, h * w * _EMBEDDING_SIZE]),
              score
            ], 1, name='resupply')

        # First fully connected layer.
        connected_1 = tf.contrib.layers.fully_connected(
            trainable=True,
            inputs=resupply,
            num_outputs=16 * (h+w),
            activation_fn=tf.nn.relu,
            weights_initializer=initializer,
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            biases_initializer=initializer)

        # Second fully connected layer, steps down.
        connected_2 = tf.contrib.layers.fully_connected(
            trainable=True,
            inputs=connected_1,
            num_outputs=8 * (h+w),
            activation_fn=tf.nn.relu,
            weights_initializer=initializer,
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            biases_initializer=initializer)

        # Third fully connected layer, steps down.
        connected_3 = tf.contrib.layers.fully_connected(
            trainable=True,
            inputs=connected_2,
            num_outputs=16,
            activation_fn=tf.nn.relu,
            weights_initializer=initializer,
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            biases_initializer=initializer)

        # Logits, softmax, random sample.
        connected_4 = tf.contrib.layers.fully_connected(
            trainable=True,
            inputs=connected_3,
            num_outputs=_ACTION_SIZE,
            activation_fn=tf.nn.sigmoid,
            weights_initializer=initializer,
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            biases_initializer=initializer)
        self.action_softmax = tf.nn.softmax(connected_4, name='action_softmax')

        # Sum the components of the softmax
        probability_histogram = tf.cumsum(self.action_softmax, axis=1)
        sample = tf.random_uniform(tf.shape(probability_histogram)[0:1])
        sample = tf.expand_dims(sample, axis=-1)
        filtered = tf.where(probability_histogram >= sample,
                            probability_histogram,
                            2.0 * tf.ones_like(probability_histogram))

        self.action_out = tf.argmin(filtered, 1)

        self.action_in = tf.placeholder(tf.int32, shape=[None, 1])
        self.advantage = tf.placeholder(tf.float32, shape=[None, 1])

        action_one_hot = tf.one_hot(self.action_in, _ACTION_SIZE,
                                    dtype=tf.float32)
        action_advantage = self.advantage * action_one_hot
        loss_policy = -100.0 * tf.reduce_mean(
            tf.reduce_sum(tf.log(self.action_softmax) * action_advantage, 1),
            name='loss_policy')
        # TODO: Investigate whether regularization losses are sums or
        # means and consider removing the division.
        loss_regularization = (0.5 / tf.to_float(tf.shape(self.state)[0]) *
            sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
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

  def predict(self, session, states, scores):
    '''Chooses actions for a list of states.

    Args:
      session: The TensorFlow session to run the net in.
      states: A list of simulation states which have been serialized
          to arrays.
      scores: A list of simulation scores for those states.

    Returns:
      An array of actions, 0 .. 4 and an array of array of
      probabilities.
    '''
    return session.run([self.action_out, self.action_softmax],
                       feed_dict={self.state: states, self.score: scores})

  def train(self, session, episodes):
    '''Trains the network.

    Args:
      episodes: A list of episodes. Each episode is a list of
          3-tuples with the state, the chosen action, and the
          reward.

    Returns:
      Training loss summaries suitable for adding to a file for
      TensorBoard.
    '''
    size = sum(map(len, episodes))
    state = np.empty([size, self._h, self._w])
    score = np.empty([size])
    action_in = np.empty([size, 1])
    advantage = np.empty([size, 1])
    i = 0
    for episode in episodes:
      r = 0.0
      for step_state, step_score, action, reward in reversed(episode):
        state[i,:,:] = step_state
        score[i] = step_score
        action_in[i,0] = action
        r = reward + 0.95 * r
        advantage[i,0] = r
        i += 1
    # Scale rewards to have unit variance
    variance = np.var(advantage)
    if variance > 1e-10:
      advantage /= variance

    summary, _ = session.run([self.summary, self.update], feed_dict={
        self.state: state,
        self.score: score,
        self.action_in: action_in,
        self.advantage: advantage
      })
    return summary


_EXPERIENCE_BUFFER_SIZE = 100
_BATCH_SIZE = 20


class PolicyGradientPlayer(player.Player):
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
