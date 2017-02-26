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

import curses
import math
import numpy as np

class ColorRamp(object):
  def __init__(self, color_offset, pair_offset, colors):
    '''Creates a ColorRamp starting at the specified index.

    1 <= color_offset and color_offset + len(colors) < curses.COLORS
    1 <= pair_offset and pair_offset + len(colors) < curses.COLOR_PAIRS

    Colors are r,g,b triples with each component between 0.0 and 1.0
    '''
    self._offset = pair_offset
    self._size = len(colors)
    def convert(x):
      assert 0.0 <= x <= 1.0
      return int(1000 * x)
    for i, (r, g, b) in enumerate(colors):
      curses.init_color(color_offset + i, convert(r), convert(g), convert(b))
      curses.init_pair(pair_offset + i, color_offset + i, curses.COLOR_BLACK)

  def __getitem__(self, i):
    if not (0 <= i and i < self._size):
      raise Exception('index out of range: %d' % i)
    return curses.color_pair(self._offset + i)


def scale(n, x, min_value, max_value):
  assert min_value <= x <= max_value
  scaled = (x - min_value) / (max_value - min_value)
  if math.isinf(scaled):
    return n // 2
  return int((n-1) * scaled)


def iterate_possible_states(state):
  '''Produces a stream of possible states by modifying state.

  Caution: This modifies the yielded arrays. Make a copy of them if
  you want to keep them.

  This uses the same encoding as actor_critic.simulation_to_array.

  Yields:
    A tuple of (y, x, state) where the player is at y, x.
  '''
  key = ' #$.@^'
  ground = key.index('.')
  player = key.index('@')
  # Erase the player.
  state = np.where(state == player, np.full(state.shape, ground), state)
  # Put the player in each possible non-terminal state:
  for v in range(state.shape[0]):
    for u in range(state.shape[1]):
      if state[v,u] == ground:
        state[v,u] = player
        yield (v, u, state)
        state[v,u] = ground
