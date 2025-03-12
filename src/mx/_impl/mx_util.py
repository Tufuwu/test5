#
# ----------------------------------------------------------------------------------------------------
#
# Copyright (c) 2023, Oracle and/or its affiliates. All rights reserved.
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

#
# Utility functions for use by mx and mx extensions.
#
# This module must only import from the standard Python library
#

__all__ = [
    "min_required_python_version",
    "min_required_python_version_str",
    "get_file_extension",
    "ensure_dirname_exists",
    "ensure_dir_exists",
    "SafeFileCreation"
]

import os.path
import errno
import sys
import tempfile
from typing import Optional
from os.path import dirname, exists, join, isdir, basename

min_required_python_version = (3, 8)
min_required_python_version_str = f'{".".join((str(d) for d in min_required_python_version))}'

_tar_compressed_extensions = {'bz2', 'gz', 'lz', 'lzma', 'xz', 'Z'}
_known_zip_pre_extensions = {'src'}


def get_file_extension(path):
    root, ext = os.path.splitext(path)
    if len(ext) > 0:
        ext = ext[1:]  # remove leading .
    if ext in _tar_compressed_extensions and os.path.splitext(root)[1] == ".tar":
        return "tar." + ext
    if ext == 'zip':
        _, pre_ext = os.path.splitext(root)
        if len(pre_ext) > 0:
            pre_ext = pre_ext[1:]  # remove leading .
        if pre_ext in _known_zip_pre_extensions:
            return pre_ext + ".zip"
    if ext == 'map':
        _, pre_ext = os.path.splitext(root)
        if len(pre_ext) > 0:
            pre_ext = pre_ext[1:]  # remove leading .
            return pre_ext + ".map"
    return ext


def change_file_extension(path, new_extension):
    ext = get_file_extension(path)
    if not ext:
        return path + '.' + new_extension
    return path[:-len(ext)] + new_extension


def change_file_name(path, new_file_name):
    return join(dirname(path), new_file_name + '.' + get_file_extension(path))


def ensure_dirname_exists(path, mode=None):
    d = dirname(path)
    if d != '':
        ensure_dir_exists(d, mode)


def ensure_dir_exists(path, mode=None):
    """
    Ensures all directories on 'path' exists, creating them first if necessary with os.makedirs().
    """
    if not isdir(path):
        try:
            if mode:
                os.makedirs(path, mode=mode)
            else:
                os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and isdir(path):
                # be happy if another thread already created the path
                pass
            else:
                raise e
    return path


# Capture the current umask since there's no way to query it without mutating it.
# Initializing here is thread-safe: https://docs.python.org/3.3/whatsnew/3.3.html#a-finer-grained-import-lock
_current_umask = os.umask(0)
os.umask(_current_umask)

class SafeFileCreation(object):
    """
    Context manager for creating a file that tries hard to handle races between processes/threads
    creating the same file. It tries to ensure that the file is created with the content provided
    by exactly one process/thread but makes no guarantee about which process/thread wins.

    Note that truly atomic file copying is hard (http://stackoverflow.com/a/28090883/6691595)

    :Example:

    with SafeFileCreation(dst) as sfc:
        shutil.copy(src, sfc.tmpPath)

    """

    _tmp_fd: Optional[int]
    _tmp_path: Optional[str]
    path: str

    def __init__(self, path: str, companion_patterns=None):
        self.path = path
        self.companion_patterns = companion_patterns or []

    def __enter__(self):
        if self.path is not None:
            path_dir = dirname(self.path)
            ensure_dir_exists(path_dir)
            # Temporary file must be on the same file system as self.path for os.rename to be atomic.
            fd, tmp = tempfile.mkstemp(suffix=basename(self.path), dir=path_dir)
            self._tmp_fd = fd
            self._tmp_path = tmp
        else:
            self._tmp_fd = None
            self._tmp_path = None
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.path is None:
            return

        assert self._tmp_fd is not None

        # Windows will complain about tmp being in use by another process
        # when calling os.rename if we don't close the file descriptor.
        os.close(self._tmp_fd)
        self._tmp_fd = None

        def _handle_file(tmp_path, path):
            if exists(tmp_path):
                if exc_value:
                    # If an error occurred, delete the temp file
                    # instead of renaming it
                    os.remove(tmp_path)
                else:
                    # Correct the permissions on the temporary file which is created with restrictive permissions
                    os.chmod(tmp_path, 0o666 & ~_current_umask)
                    if sys.platform.startswith('win32'):
                        try:
                            if exists(path):
                                try:
                                    os.remove(path)
                                except (FileNotFoundError, PermissionError):
                                    # Another process removed or re-created it in the meantime
                                    pass
                            os.rename(tmp_path, path)
                        except FileExistsError:
                            # This is how atomic file rename is "supported" on Windows:
                            # the process loosing a file rename race gets this error.
                            os.remove(tmp_path)
                    else:
                        # Atomic if path does not already exist.
                        os.rename(tmp_path, path)
        _handle_file(self.tmpPath, self.path)
        for companion_pattern in self.companion_patterns:
            _handle_file(companion_pattern.format(path=self.tmpPath), companion_pattern.format(path=self.path))

        self._tmp_path = None

    @property
    def tmpPath(self) -> str:
        assert self._tmp_path is not None
        return self._tmp_path

    @property
    def tmpFd(self) -> int:
        assert self._tmp_fd is not None
        return self._tmp_fd

# Internal test support

def _create_tmp_files(tmp_dir, num):
    """
    Creates `num` files in `tmp_dir` using SafeFileCreation.
    """
    for i in range(num):
        filename = f'{tmp_dir}/tmp{i}'
        with SafeFileCreation(filename) as sfc:
            pid = os.getpid()
            with open(sfc.tmpPath, 'w') as out:
                print(f"file {i} created by process {pid}", file=out)
