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

from srl import movement
from srl import simulation
from srl import world
import srl.policy_gradient as pg


class TestReplayBuffer(unittest.TestCase):
  def test_fillWithPositive(self):
    rb = pg.ReplayBuffer(2)
    self.assertEquals(0, rb.size)
    rb.add(1, 'foo')
    self.assertEquals(1, rb.size)
    rb.add(2, 'bar')
    self.assertEquals(2, rb.size)

    rb.add(3, 'foo')
    self.assertEquals(2, rb.size, 'should not grow beyond capacity')

  def test_sample(self):
    rb = pg.ReplayBuffer(1)
    rb.add(42, 'foo')
    self.assertEquals('foo', rb.sample())


class TestPolicyGradientNetwork(unittest.TestCase):
  def testPredict(self):
    g = tf.Graph()
    net = pg.PolicyGradientNetwork('testPredict', g, (7, 11))

    s = tf.Session(graph=g)
    with g.as_default():
      init = tf.global_variables_initializer()
      s.run(init)

    sim = simulation.Simulation(world.Generator(11, 7))
    [[act], _] = net.predict(s, [sim.to_array()], [sim.score])
    self.assertTrue(0 <= act)
    self.assertTrue(act < len(movement.ALL_ACTIONS))

  def testTrain(self):
    g = tf.Graph()
    net = pg.PolicyGradientNetwork('testTrain', g, (4, 4))

    s = tf.Session(graph=g)
    with g.as_default():
      init = tf.global_variables_initializer()
      s.run(init)

    sim = simulation.Simulation(world.Generator(4, 4))
    state = sim.to_array()
    net.train(s, [[(state, 0, 3, 7), (state, -1, 3, -1)],
                  [(state, 0, 0, 1000)]])

  def testActionOut_untrainedPrediction(self):
    g = tf.Graph()
    net = pg.PolicyGradientNetwork('testActionOut_untrainedPrediction', g,
                                   (17, 13))
    s = tf.Session(graph=g)
    with g.as_default():
      init = tf.global_variables_initializer()
      s.run(init)
    act = s.run(net.action_out,
                feed_dict={
                  net.state: [np.zeros((17, 13))],
                  net.score: np.zeros((1,)),
                })
    self.assertTrue(0 <= act)
    self.assertTrue(act < len(movement.ALL_ACTIONS))

  def testUpdate(self):
    g = tf.Graph()
    net = pg.PolicyGradientNetwork('testUpdate', g, (13, 23))
    s = tf.Session(graph=g)
    with g.as_default():
      init = tf.global_variables_initializer()
      s.run(init)
    s.run(net.update, feed_dict={
        net.state: np.zeros((7, 13, 23)),
        net.score: np.zeros((7,)),
        net.action_in: np.zeros((7, 1)),
        net.advantage: np.zeros((7, 1)),
      })

  def testUpdate_lossDecreases(self):
    w = world.World.parse('@.....$')

    g = tf.Graph()
    net = pg.PolicyGradientNetwork('testUpdate_lossDecreases', g, (w.h, w.w))
    s = tf.Session(graph=g)
    with g.as_default():
      init = tf.global_variables_initializer()
      s.run(init)

    state = simulation.Simulation(world.Static(w)).to_array()
    losses = []
    for _ in range(10):
      loss, _ = s.run([net.loss, net.update], feed_dict={
            net.state: [state],
            net.score: [-50],
            net.action_in: [[1]],
            net.advantage: [[2]],
          })
      losses.append(loss)
    self.assertTrue(losses[-1] < losses[0])
