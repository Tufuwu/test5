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

from argparse import ArgumentParser
import re

def parse_fd(fd, path="<fd>"):
    content = fd.read().decode('utf-8')
    return parse_string(content, path=path)

def parse_file(path):
    with open(path, "r") as f:
        return parse_fd(f, path)

def parse_string(content, path="<toml-string>"):
    parser = _StomlParser()
    return parser.parse(path, content)

class _Streamer:
    EOF = ""

    def __init__(self, path, content):
        self.path = path
        self.content = content
        self.lines = []
        self.line = ""
        self.row = 0
        self.column = 0
        self.pos = 0

    def terminate(self, message):
        row = self.row
        column = self.column
        self.slurp(len(self.content))
        raise RuntimeError(
            self.path + ":" + str(row + 1) + ":" + str(column) + ": " + message + "\n" +
            (self.lines[row] if row < len(self.lines) else ("<row " + str(row)) + ">") + "\n" +
            (" " * column) + "^" + "\n")

    def peek(self, ahead=0):
        if self.pos + ahead < len(self.content):
            return self.content[self.pos + ahead]
        return _Streamer.EOF

    def peek_to_whitespace(self):
        token = ""
        for i in range(len(self.content) - self.pos + 1):
            next_char = self.peek(i)
            if (next_char == _Streamer.EOF) or next_char.isspace():
                break
            token = token + next_char
        return token

    def pull(self, expected=None):
        if expected is None:
            self.slurp(1)
            return
        for i in range(0, len(expected)):
            if self.peek(i) != expected[i]:
                self.terminate("Unexpected string, expected '" + expected + "'")
        self.slurp(len(expected))

    def pullSpaces(self):
        inside_comment = False
        while True:
            next_char = self.peek()
            if next_char == _Streamer.EOF:
                break
            if inside_comment:
                if next_char in ["\r", "\n"]:
                    inside_comment = False
                self.pull()
                continue
            if next_char.isspace():
                self.pull()
                continue
            if next_char == "#":
                self.pull()
                inside_comment = True
                continue
            # Some other character and we are not inside a comment
            break

    def slurp(self, count):
        for _ in range(0, count):
            character = self.peek()
            if character in ("\n", _Streamer.EOF):
                self.lines.append(self.line)
                self.line = ""
                self.row = self.row + 1
                self.column = 0
            else:
                self.line = self.line + character
                self.column = self.column + 1
            self.pos = self.pos + 1

class _StomlParser:
    TABLE_MARKER = re.compile(r"^\[(?P<name>.*)]$")
    ARRAY_OF_TABLES_MARKER = re.compile(r"^\[\[(?P<name>.*)]]$")

    def parse(self, path, content):
        tree = {}
        streamer = _Streamer(path, content)
        self.root(streamer, tree)
        return tree

    def root(self, streamer, tree):
        while True:
            streamer.pullSpaces()
            if streamer.peek() == _Streamer.EOF:
                return
            next_token = streamer.peek_to_whitespace()

            is_table = _StomlParser.TABLE_MARKER.match(next_token)
            is_array_of_tables = _StomlParser.ARRAY_OF_TABLES_MARKER.match(next_token)

            if is_array_of_tables:
                streamer.pull(next_token)
                table_name = is_array_of_tables.group("name")
                if not table_name in tree:
                    tree[table_name] = []
                tree[table_name].append(self.parse_table(streamer))
            elif is_table:
                streamer.pull(next_token)
                tree[is_table.group("name")] = self.parse_table(streamer)
            else:
                streamer.terminate("Expected table or array of tables.")

    def parse_table(self, streamer):
        result = {}
        while True:
            streamer.pullSpaces()
            if self.valid_identifier_character(streamer.peek()):
                self.keyvalue(streamer, result)
            else:
                return result

    def keyvalue(self, streamer, result):
        key = self.identifier(streamer)
        streamer.pullSpaces()
        streamer.pull("=")
        streamer.pullSpaces()
        if streamer.peek() == "\"":
            # string
            value = self.string(streamer)
        elif streamer.peek() == "[":
            # list of strings
            value = self.list(streamer)
        elif streamer.peek_to_whitespace() in ["true", "false"]:
            value = self.boolean(streamer)
        else:
            value = None
            streamer.terminate("Expected either a string or a boolean or a list of strings.")
        result[key] = value

    def valid_identifier_character(self, c):
        return c.isalpha() or c == "_"

    def identifier(self, streamer):
        ident = ""
        while self.valid_identifier_character(streamer.peek()):
            ident = ident + streamer.peek()
            streamer.pull()
        return ident

    def boolean(self, streamer):
        val = self.identifier(streamer)
        if val == "true":
            return True
        elif val == "false":
            return False
        else:
            streamer.terminate("Expected either true or false.")

    def string(self, streamer):
        streamer.pull("\"")
        content = ""
        while streamer.peek() != "\"":
            content = content + streamer.peek()
            streamer.pull()
        streamer.pull()
        return content

    def list(self, streamer):
        streamer.pull("[")
        values = []
        streamer.pullSpaces()
        while streamer.peek() != "]":
            streamer.pullSpaces()
            value = self.string(streamer)
            values.append(value)
            streamer.pullSpaces()
            if streamer.peek() == ",":
                streamer.pull()
                streamer.pullSpaces()
        streamer.pull()
        return values


if __name__ == "__main__":
    parser = ArgumentParser(prog="SimpleTOML parser.", description="Parses a simplified version of TOML.")
    parser.add_argument("filename")
    args = parser.parse_args()

    rules = parse_file(args.filename)
    print(rules)
