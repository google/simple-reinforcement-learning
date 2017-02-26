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

# Some useful tutorial references on implementing Actor-Critic variants
# in TensorFlow:
# * [Deep Deterministic Policy Gradients in Tensorflow](http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html)
# * [Asynchronous Actor-Critic Agents](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)

import numpy as np
import random
import tensorflow as tf
import unittest

import movement
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
      old_state: The np.ndarray of state data.
      action: The np.int64, 0 <= action < 4, indicating which action
          was taken.
      reward: The float reward from the environment for taking action.
      new_state: The np.ndarray of state data after action was taken.
          This should have the same shape as old_state.
      is_terminal: The bool indicating whether new_state is a terminal state.

    '''
    assert old_state.shape == new_state.shape
    assert old_state.dtype == new_state.dtype == np.int8
    assert type(action) == np.int64
    assert type(reward) == float
    assert type(is_terminal) == bool
    sample = (old_state, action, reward, new_state, is_terminal)
    if self.size < self._max_size:
      self._buffer.append(sample)
    else:
      self._buffer[random.randrange(self._max_size)] = sample

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
    buffer = np.array(buffer)
    old_state = np.stack(buffer[:,0])
    action = np.stack(buffer[:,1])
    reward = np.stack(buffer[:,2])
    new_state = np.stack(buffer[:,3])
    terminal = np.stack(buffer[:,4])
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
ACTION_SIZE = len(movement.ALL_ACTIONS)


class ActorCriticNetwork(object):
  def __init__(self, name, world_size_h_w, trainer=None):
    '''Creates an actor-critic network.

    This creates a TensorFlow model in the default graph. Variables
    are scoped with the specified name; the name should be unique.

    Args:
      name: The name of the network.
      world_size_h_w: A tuple of the height, width of the world in cells.
      trainer: If not None, builds additional ops for training.
    '''
    self._name = name
    h, w = world_size_h_w
    with tf.variable_scope(self._name):
      self.state_input = tf.placeholder(tf.int32, shape=[None, h, w],
                                        name='state')
      (self.actor_softmax, self.actor_action, self.critic) = (
          self._neural_network(self.state_input, world_size_h_w))
      if trainer:
        (self.train_actions, self.train_target_values, self.train_advantages,
         self.train_loss, self.train_update) = (
            self._training(trainer, self.actor_softmax, self.critic))

  def _training(self, trainer, actor, critic):
    '''Creates the part of the network for training.

    The target network is not trained and does not need this.

    Args:
      trainer: The optimizer to train with. See tf.train.*Optimizer.
      actor: The actor_softmax part of the network from _neural_network.
      critic: The critic part of the network from _neural_network.

    Returns:
      An n-tuple of:
        IN: actions [-1] tf.int32 of 0-3 for up, right, down, left.
        IN: target value [-1] tf.float32.
        IN: advantages [-1] tf.float32.
        OUT: loss [1] tf.float32.
        OUT: update training step which updates variables

      To use update, the feed for actor and critic must be supplied.
    '''
    with tf.variable_scope('training'):
      # Advantage estimate, A = R - V(s)
      action_input = tf.placeholder(shape=[None], dtype=tf.int32, name='action')
      action_onehot = tf.one_hot(action_input, ACTION_SIZE, dtype=tf.float32)
      target_value_input = tf.placeholder(shape=[None], dtype=tf.float32,
                                          name='target_value')
      advantages_input = tf.placeholder(shape=[None], dtype=tf.float32,
                                        name='advantage')
      responsible_outputs = tf.reduce_sum(actor * action_onehot, [1])

      value_loss = tf.reduce_sum(tf.square(target_value_input - critic))
      # Note: This is "just" entropy, a *positive* number of nats,
      # which is *subtracted* to *decrease* the loss. The network is
      # encouraged to be tentative--high entropy--about action choice
      # to encourage exploration.
      entropy_boost = -tf.reduce_sum(actor * tf.log(actor))
      # TODO: Need to think about negative rewards. "Asynchronous
      # Actor-Critic Agents" uses a log here but that will generate
      # smaller losses for overconfidence. Maybe that's just balanced
      # by the entropy term.
      policy_loss = tf.reduce_sum(tf.log(responsible_outputs) *
                                  advantages_input)
      loss = 0.3 * value_loss + 0.7 * policy_loss - 0.01 * entropy_boost

      local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     self._name)
      gradients = tf.gradients(loss, local_vars)
      # TODO: Examine gradients; does this clipping make sense?
      gradients, _ = tf.clip_by_global_norm(gradients, 40.0)
      update = trainer.apply_gradients(zip(gradients, local_vars))
      return action_input, target_value_input, advantages_input, loss, update

  def assign_op(self, source):
    '''Builds an operation which assigns the source network to this network.

    Args:
      source: The ActorCriticNetwork to take variable values from.

    Returns:
      A TensorFlow operation.
    '''
    sink_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  self._name)
    source_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                    source._name)
    assert len(sink_vars) == len(source_vars)
    assert len(sink_vars) > 0
    assignments = map(lambda u, v: tf.assign(u, v),
                      sink_vars[1:], source_vars[1:])
    with tf.control_dependencies(assignments):
      return tf.assign(sink_vars[0], source_vars[0])

  def _neural_network(self, state, world_size_h_w):
    '''Makes a neural network for examining worlds.

    This network features:
    - An embedding layer.
    - Two layers of convolutions with max pooling.
    - A fully connected layer. The embedding layer is resupplied.
    - Output softmax for actions.
    - Output value estimation for the state.

    Args:
      state: The map, a tensor of height * width.
      world_size_h_w: A tuple, the height and width of the world.

    Returns:
      A tuple of (actor_softmax, actor_action, critic). The actor is a
      softmax of actions and the index of the maximum for convenience;
      the critic is a value estimation.
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

    # Actor softmax.
    w3 = tf.Variable(
        tf.truncated_normal([fc_output_size, ACTION_SIZE], stddev=0.1))
    b3 = tf.Variable(tf.constant(0.1, shape=[ACTION_SIZE]))
    logits_pre = tf.matmul(fc, w3) + b3
    logits = tf.sigmoid(logits_pre)
    softmax = tf.nn.softmax(logits)
    action = tf.argmax(softmax, axis=1)

    # Critic
    w4 = tf.Variable(tf.truncated_normal([fc_output_size, 1], stddev=0.1))
    b4 = tf.Variable(tf.constant(0.1, shape=[1]))
    value = tf.matmul(fc, w4) + b4
    value = tf.squeeze(value, axis=1)

    return softmax, action, value


def sample(xs):
  y = random.random()
  for i, x in enumerate(xs):
    y -= x
    if y < 0.:
      return i
  return len(xs)-1


class ActorCriticPlayer(object):
  '''A game driver which learns using a neural network.'''
  def __init__(self, world_h_w):
    # TODO: Composition would be easier if this did not control the
    # graph and session.
    self._graph = tf.Graph()
    with self._graph.as_default():
      trainer = tf.train.AdamOptimizer()
      self._net = ActorCriticNetwork('net', world_h_w, trainer=trainer)
      self._target_net = ActorCriticNetwork('target', world_h_w)
      init = tf.global_variables_initializer()
      self._update_target = self._target_net.assign_op(self._net)
    self._session = tf.Session(graph=self._graph)
    self._session.run(init)
    self._experience = ReplayBuffer(100)
    self._step = 0
    self._annot_loss = 0.
    self._annot_actions = [0.] * ACTION_SIZE
    self._annotate_y = world_h_w[0] + 1

  @property
  def should_quit(self):
    return False

  def interact(self, sim, window):
    if sim.in_terminal_state:
      sim.reset()
      return
    # Move!
    state = simulation_to_array(sim)
    [[act], self._annot_actions] = self._session.run(
      [self._net.actor_action, self._net.actor_softmax],
      feed_dict={self._net.state_input: [state]})
    # TODO: Clean this duct tape up
    act = np.int64(sample(self._annot_actions[0]))
    reward = float(sim.act(movement.ALL_ACTIONS[act]))
    new_state = simulation_to_array(sim)
    is_terminal = sim.in_terminal_state
    self._experience.add(state, act, reward, new_state, is_terminal)

    # Learn
    old_states, actions, rewards, new_states, _ = self._experience.sample_batch(
        10)
    gamma = 0.9
    # TODO: This could be done in fewer runs with some batching and slicing.
    # Value
    new_states_v = self._session.run(self._target_net.critic, feed_dict={
        self._target_net.state_input: new_states
      })
    target_v = rewards + gamma * new_states_v

    # Policy
    # TODO: Should this use the target network or the learning network?
    old_states_v = self._session.run(self._target_net.critic, feed_dict={
        self._target_net.state_input: old_states
      })
    advantages = rewards + gamma * new_states_v - old_states_v

    self._annot_loss, _ = self._session.run(
      [self._net.train_loss, self._net.train_update],
      feed_dict={
          self._net.train_actions: actions,
          self._net.train_target_values: target_v,
          self._net.train_advantages: advantages,
          self._net.state_input: old_states
        })

    # Snapshot current network to target network
    self._step += 1
    # TODO: Output loss
    # Snapshot to target
    if self._step % 1000 == 0:
      self._session.run(self._update_target)

  def annotate(self, window):
    window.addstr(self._annotate_y, 0, 'Loss: %.4f' % self._annot_loss)
    window.addstr(self._annotate_y + 1, 0, 'Actions: %s' % self._annot_actions)


class TestNeuralNetwork(unittest.TestCase):
  def testTrain(self):
    w = world.World.parse('''
@
.
.
.
.
$''')
    g = tf.Graph()
    with g.as_default():
      net = ActorCriticNetwork('testTrain', (w.h, w.w),
                               trainer=tf.train.AdamOptimizer())
      init = tf.global_variables_initializer()
    state = simulation_to_array(simulation.Simulation(w))
    session = tf.Session(graph=g)
    session.run(init)
    losses = []
    for i in range(10):
      loss, _ = session.run([net.train_loss, net.train_update], feed_dict={
          net.train_actions: [3],
          net.train_target_values: [7],
          net.train_advantages: [3],
          net.state_input: [state]
        })
      losses.append(loss)
    self.assertTrue(losses[-1] < losses[0])

  def testAssign(self):
    w = world.Generator(10, 20).generate()
    g = tf.Graph()
    with g.as_default():
      net_a = ActorCriticNetwork('a', (w.h, w.w))
      net_b = ActorCriticNetwork('b', (w.h, w.w))
      assign = net_a.assign_op(net_b)
    state = simulation_to_array(simulation.Simulation(w))
    with tf.Session(graph=g) as session:
      session.run(tf.global_variables_initializer())
      act_a, act_b = session.run([net_a.actor_softmax, net_b.actor_softmax],
                                 feed_dict={net_a.state_input: [state],
                                            net_b.state_input: [state]})
      self.assertFalse(np.allclose(act_a, act_b))
      session.run(assign)
      act_a = session.run(net_a.actor_softmax,
                          feed_dict={net_a.state_input: [state]})
      self.assertTrue(np.allclose(act_a, act_b))

  def testSmallWorld(self):
    # This first because it is useful to have the world dimensions.
    w = world.World.parse('''
@.#$
^...
##^^
####
####
####''')
    self.do_test(w)

  def testLargeWorld(self):
    g = world.Generator(30, 20)
    w = g.generate()
    self.do_test(w)

  def do_test(self, w):
    # Make the graph.
    g = tf.Graph()
    with g.as_default():
      net = ActorCriticNetwork('test', (w.h, w.w))

    # Start a simulation.
    sim = simulation.Simulation(w)
    raw = simulation_to_array(sim)
    with tf.Session(graph=g) as session:
      session.run(tf.global_variables_initializer())
      act = session.run(net.actor_action, feed_dict={net.state_input: [raw]})
      self.assertTrue(0 <= act[0] and act[0] < ACTION_SIZE)

# TODO(dominicc):
# - implement a loop which drives learning
# - examine whether # gets a similar embedding to ' '

if __name__ == '__main__':
  unittest.main()
