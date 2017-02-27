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

from world import *


def load_tests(loader, tests, pattern):
  suite = unittest.TestSuite()
  for test_class in (TestWorld, TestGenerator):
    tests = loader.loadTestsFromTestCase(test_class)
    suite.addTests(tests)
  return suite


class TestWorld(unittest.TestCase):
  def test_size(self):
    g = World.parse('@$')
    self.assertEqual((2, 1), g.size)

  def test_init_state(self):
    g = World.parse('####\n#.@#\n####')
    self.assertEqual((2, 1), g.init_state)

  def test_parse_no_init_state_fails(self):
    with self.assertRaises(WorldFailure):
      World.parse('#')


class TestGenerator(unittest.TestCase):
  def test_generate_tiny_world(self):
    g = Generator(2, 1)
    w = g.generate()
    # The world should have a start and goal
    if w.init_state == (0, 0):
      self.assertEqual('$', w.at((1, 0)))
    elif w.init_state == (1, 0):
      self.assertEqual('$', w.at((0, 0)))
    else:
      self.fail('the start position %s is invalid' % (w.init_state,))
