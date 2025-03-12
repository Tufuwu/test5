#
# ----------------------------------------------------------------------------------------------------
#
# Copyright (c) 2024, 2024, Oracle and/or its affiliates. All rights reserved.
# DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
#
# This code is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License version 2 only, as
# published by the Free Software Foundation.
#
# This code is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# version 2 for more details (a copy is included in the LICENSE file that
# accompanied this code).
#
# You should have received a copy of the GNU General Public License version
# 2 along with this work; if not, write to the Free Software Foundation,
# Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
#
# Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
# or visit www.oracle.com if you need additional information or have any
# questions.
#
# ----------------------------------------------------------------------------------------------------
#

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from argparse import Namespace
from threading import Lock
from typing import Dict, Optional, MutableSequence

from ..daemon import Daemon
from ..suite import Dependency
from ... import mx
from ...support.logging import nyi, setLogTask
from ...support.processes import Process

__all__ = ["Task", "TaskAbortException"]

Args = Namespace

class TaskAbortException(Exception):
    pass

class Task(object, metaclass=ABCMeta):
    """A task executed during a build."""

    subject: Dependency
    deps: MutableSequence[Task]
    args: Args
    parallelism: int
    proc: Optional[Process]

    consoleLock = Lock()

    def __init__(self, subject: Dependency, args: Args, parallelism: int):
        """
        :param subject: the dependency for which this task is executed
        :param args: arguments of the build command
        :param parallelism: the number of CPUs used when executing this task
        """
        self.subject = subject
        self.args = args
        self.parallelism = parallelism
        self.deps = []
        self.proc = None
        self.subprocs = []
        self._log = mx.LinesOutputCapture()
        self._echoImportant = not hasattr(args, 'build_logs') or args.build_logs in ["important", "full"]
        self._echoAll = hasattr(args, 'build_logs') and args.build_logs == "full"
        self._exitcode = 0
        self.status = None
        self.statusInfo = ""

    def __str__(self) -> str:
        return nyi('__str__', self)

    def __repr__(self) -> str:
        return str(self)

    @property
    def name(self) -> str:
        return self.subject.name

    def enter(self):
        self.status = "running"
        setLogTask(self)

    def leave(self):
        if self.status == "running":
            self.status = "success"
        setLogTask(None)

    def abort(self, code):
        self._exitcode = code
        self.status = "failed"
        raise TaskAbortException(code)

    def log(self, msg, echo=False, log=True, important=True, replace=False):
        """
        Log output for this build task.

        Whether the output also goes to the console depends on the `--build-logs` option:
        * In `silent`, `oneline` and `interactive` mode, only messages with `echo=True` are printed.
        * In `full` mode, all messages are printed.
        * In `important` mode, only messages with `important=True` are printed.

        `log=False` can be used to only do output, without including it in the log. This is useful
        in combination with `echo=True` to print a shorter summary of information that's already in
        the log in a more detailed form.

        `replace=True` replaces the last logged line. This is useful for status output, e.g. download progress.
        """
        if log:
            if replace:
                del self._log.lines[-1]
            self._log(msg)
        if echo or self._echoAll or (important and self._echoImportant):
            with Task.consoleLock:
                print(msg.rstrip())

    def getLastLogLine(self):
        for line in reversed(self._log.lines):
            if line.strip():
                return line
        return None

    def addSubproc(self, p):
        self.subprocs += [p]

    def cancelSubprocs(self):
        from ...support.processes import _is_process_alive, _kill_process
        from signal import SIGTERM
        for p in self.subprocs:
            if not _is_process_alive(p):
                continue
            if mx.is_windows():
                p.terminate()
            else:
                _kill_process(p.pid, SIGTERM)

    @property
    def exitcode(self):
        if self._exitcode != 0:
            return self._exitcode
        else:
            return self.proc.exitcode

    @property
    def build_time(self):
        return getattr(self.subject, "build_time", 1)

    def initSharedMemoryState(self) -> None:
        pass

    def pushSharedMemoryState(self) -> None:
        pass

    def pullSharedMemoryState(self) -> None:
        pass

    def cleanSharedMemoryState(self) -> None:
        pass

    def prepare(self, daemons: Dict[str, Daemon]):
        """
        Perform any task initialization that must be done in the main process.
        This will be called just before the task is launched.
        The 'daemons' argument is a dictionary for storing any persistent state
        that might be shared between tasks.
        """

    @abstractmethod
    def execute(self) -> None:
        """Executes this task."""
