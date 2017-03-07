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

import numpy as np
import tensorflow as tf
import unittest

from srl import dqn
from srl import movement
from srl import simulation
from srl import world


class TestReplayBuffer(unittest.TestCase):
  def test_fillWithPositive(self):
    rb = dqn.ReplayBuffer(2)
    self.assertEquals(0, rb.size)
    rb.add(1, 'foo')
    self.assertEquals(1, rb.size)
    rb.add(2, 'bar')
    self.assertEquals(2, rb.size)

    rb.add(3, 'foo')
    self.assertEquals(2, rb.size, 'should not grow beyond capacity')

  def test_sample(self):
    rb = dqn.ReplayBuffer(1)
    rb.add(42, 'foo')
    self.assertEquals('foo', rb.sample())


class TestDeepQNetwork(unittest.TestCase):
  def testPredict(self):
    g = tf.Graph()
    net = dqn.DeepQNetwork('testPredict', g, (7, 11))

    s = tf.Session(graph=g)
    with g.as_default():
      init = tf.global_variables_initializer()
      s.run(init)

    sim = simulation.Simulation(world.Generator(11, 7))
    [[act], _] = net.predict(s, [sim.to_array()])
    self.assertTrue(0 <= act)
    self.assertTrue(act < len(movement.ALL_ACTIONS))

  def testUpdateTarget(self):
    g = tf.Graph()
    target = dqn.DeepQNetwork('testTarget', g, (7, 11))
    net = dqn.DeepQNetwork('testTrain', g, (7, 11), target=target)

    s = tf.Session(graph=g)
    with g.as_default():
      init = tf.global_variables_initializer()
      s.run(init)

    # Run each network once.
    sim = simulation.Simulation(world.Generator(11, 7))
    [_, probabilities_net] = net.predict(s, [sim.to_array()])
    [_, probabilities_target] = target.predict(s, [sim.to_array()])
    self.assertFalse(
      np.all(np.isclose(probabilities_net, probabilities_target)),
      '"net" and "target" should (probably) be different')

    # Copy the network to target; run target.
    s.run(net.update_target)
    [_, probabilities_target] = target.predict(s, [sim.to_array()])

    self.assertTrue(
      np.all(np.isclose(probabilities_net, probabilities_target)),
      'the target network should now be a copy of "net"')

  def testTrain(self):
    g = tf.Graph()
    target = dqn.DeepQNetwork('testTarget', g, (4, 4))
    net = dqn.DeepQNetwork('testTrain', g, (4, 4), target=target)

    s = tf.Session(graph=g)
    with g.as_default():
      init = tf.global_variables_initializer()
      s.run(init)

    sim = simulation.Simulation(world.Generator(4, 4))
    state = sim.to_array()
    net.train(s, [[(state, 0, 3, state, False), (state, 1, 2, state, False)],
                  [(state, 0, 0, state, True)]])

  def testActionOut_untrainedPrediction(self):
    g = tf.Graph()
    net = dqn.DeepQNetwork('testActionOut_untrainedPrediction', g, (17, 13))
    s = tf.Session(graph=g)
    with g.as_default():
      init = tf.global_variables_initializer()
      s.run(init)
    act = s.run(net.action_out,
                feed_dict={
                  net.state: [np.zeros((17, 13))],
                })
    self.assertTrue(0 <= act)
    self.assertTrue(act < len(movement.ALL_ACTIONS))

  def testUpdate(self):
    g = tf.Graph()
    target = dqn.DeepQNetwork('testUpdate_target', g, (13, 23))
    net = dqn.DeepQNetwork('testUpdate', g, (13, 23), target=target)
    s = tf.Session(graph=g)
    with g.as_default():
      init = tf.global_variables_initializer()
      s.run(init)
    s.run(net.update, feed_dict={
        net.state: np.zeros((7, 13, 23)),
        net.action_in: np.zeros((7,)),
        net.reward: np.zeros((7,)),
        net.is_terminal: np.zeros((7,)),
        net.next_state: np.zeros((7, 13, 23)),
      })

  def testUpdate_lossDecreases(self):
    w = world.World.parse('@.....$')

    g = tf.Graph()
    target = dqn.DeepQNetwork('testUpdate_lossDecreases_target', g, (w.h, w.w))
    net = dqn.DeepQNetwork('testUpdate_lossDecreases', g, (w.h, w.w),
                           target=target)
    s = tf.Session(graph=g)
    with g.as_default():
      init = tf.global_variables_initializer()
      s.run(init)

    state = simulation.Simulation(world.Static(w)).to_array()
    losses = []
    for _ in range(10):
      loss, _ = s.run([net.loss, net.update], feed_dict={
            net.state: [state],
            net.next_state: [state],
            net.action_in: [3],
            net.reward: [1],
            net.is_terminal: [False],
          })
      losses.append(loss)
    self.assertTrue(losses[-1] < losses[0])
