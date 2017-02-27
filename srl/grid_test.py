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

from unittest.mock import patch
import unittest

from srl import movement
from srl import simulation
from srl import world
from srl import grid


class TestMachinePlayer(unittest.TestCase):
  def test_interact(self):
    TEST_ACTION = movement.ACTION_RIGHT
    q = grid.QTable(-1)
    q.set((0, 0), TEST_ACTION, 1)

    player = grid.MachinePlayer(grid.GreedyQ(q), grid.StubLearner())
    w = world.World.parse('@.')
    with patch.object(simulation.Simulation, 'act') as mock_act:
      sim = simulation.Simulation(w)
      player.interact(sim, grid.StubWindow())
    mock_act.assert_called_once_with(TEST_ACTION)

  def test_does_not_quit(self):
    player = grid.MachinePlayer(None, None)
    self.assertFalse(player.should_quit)
