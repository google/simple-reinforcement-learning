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

import unittest

# Grid world maps are specified with characters a bit like NetHack:
# #, (blank) are impassable
# . is passable
# @ is the player start point
# ^ is a trap, with a large negative reward
# $ is the goal
VALID_CHARS = set(['#', '.', '@', '$', '^', ' '])


class WorldFailure(Exception):
  pass


class World(object):
  '''A grid world.'''
  def __init__(self, init_state, lines):
    '''Creates a grid world.
    init_state: the (x,y) player start position
    lines: list of strings of VALID_CHARS, the map'''
    self.init_state = init_state
    self._lines = [] + lines

  @classmethod
  def parse(cls, s):
    '''Parses a grid world in the string s.
s must be made up of equal-length lines of VALID_CHARS with one start position
denoted by @.'''
    init_state = None

    lines = s.split()
    if not lines:
      raise WorldFailure('no content')
    for (y, line) in enumerate(lines):
      if y > 0 and len(line) != len(lines[0]):
        raise WorldFailure('line %d is a different length (%d vs %d)' %
                                (y, len(line), len(lines[0])))
      for (x, ch) in enumerate(line):
        if not ch in VALID_CHARS:
          raise WorldFailure('invalid char "%c" at (%d, %d)' % (ch, x, y))
        if ch == '@':
          if init_state:
            raise WorldFailure('multiple initial states, at %o and '
                               '(%d, %d)' % (init_state, x, y))
          init_state = (x, y)
    if not init_state:
      raise WorldFailure('no initial state, use "@"')
    # The player start position is in fact ordinary ground.
    x, y = init_state
    line = lines[y]
    lines[y] = line[0:x] + '.' + line[x+1:]
    return World(init_state, lines)

  @property
  def size(self):
    '''The size of the grid world, width by height.'''
    return self.w, self.h

  @property
  def h(self):
    '''The height of the grid world.'''
    return len(self._lines)

  @property
  def w(self):
    '''The width of the grid world.'''
    return len(self._lines[0])

  def at(self, pos):
    '''Gets the character at an (x, y) coordinate.
Positions are indexed from the origin 0,0 at the top, left of the map.'''
    x, y = pos
    return self._lines[y][x]


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


if __name__ == '__main__':
  unittest.main()
