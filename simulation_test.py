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

import unittest

from simulation import *
import world


def load_tests(loader, tests, pattern):
  return loader.loadTestsFromTestCase(TestSimulation)


class TestSimulation(unittest.TestCase):
  def test_in_terminal_state(self):
    w = world.World.parse('@^')
    sim = Simulation(w)
    self.assertFalse(sim.in_terminal_state)
    sim.act(movement.ACTION_RIGHT)
    self.assertTrue(sim.in_terminal_state)

  def test_act_accumulates_score(self):
    w = world.World.parse('@.')
    sim = Simulation(w)
    sim.act(movement.ACTION_RIGHT)
    sim.act(movement.ACTION_LEFT)
    self.assertEqual(-2, sim.score)
