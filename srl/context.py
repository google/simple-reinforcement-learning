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

import collections
import curses


class StubFailure(Exception):
  pass


class StubWindow(object):
  '''A no-op implementation of the game display.'''
  def addstr(self, y, x, s):
    pass

  def erase(self):
    pass

  def getch(self):
    raise StubFailure('"getch" not implemented; use a mock')

  def move(self, y, x):
    pass

  def refresh(self):
    pass


class StubContext(object):
  def __init__(self):
    self.window = StubWindow()

  def start(self):
    self.run_loop.start()


class Context(object):
  '''Provides the shared curses window and a run loop to other objects.

  Properties:
    run_loop: See RunLoop.
    window: A curses window to display text.
  '''

  def __init__(self):
    self.run_loop = RunLoop()
    self.window = None

  def start(self):
    '''Initializes the context and starts the run loop.'''
    curses.wrapper(self._capture_window)

  def _capture_window(self, window):
    self.window = window
    self.run_loop.start()


class RunLoop(object):
  '''A run loop invokes its tasks until there are none left.'''

  def __init__(self):
    self._tasks = collections.deque()
    self._quit = object()

  def start(self):
    while len(self._tasks):
      task = self._tasks.popleft()
      if task is self._quit:
        return
      task()

  def post_task(self, task, repeat=False):
    if repeat:
      def repeater():
        task()
        self.post_task(repeater)
      self.post_task(repeater)
    else:
      self._tasks.append(task)

  def post_quit(self):
    self._tasks.append(self._quit)
