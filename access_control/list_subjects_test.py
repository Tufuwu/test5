# Copyright 2021 Google LLC
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
#
"""Unit tests for the "list_subjects" module."""

import unittest
import argparse

from unittest import mock

from google.auth.transport import requests

from . import list_subjects


class ListSubjectsTest(unittest.TestCase):

  def test_initialize_command_line_args(self):
    actual = list_subjects.initialize_command_line_args([])
    self.assertEqual(actual,
                     argparse.Namespace(credentials_file=None, region="us"))

  @mock.patch.object(requests, "AuthorizedSession", autospec=True)
  @mock.patch.object(requests.requests, "Response", autospec=True)
  def test_list_subjects_error(self, mock_response, mock_session):
    mock_session.request.return_value = mock_response
    type(mock_response).status_code = mock.PropertyMock(return_value=400)
    mock_response.raise_for_status.side_effect = (
        requests.requests.exceptions.HTTPError())

    with self.assertRaises(requests.requests.exceptions.HTTPError):
      list_subjects.list_subjects(mock_session)

  @mock.patch.object(requests, "AuthorizedSession", autospec=True)
  @mock.patch.object(requests.requests, "Response", autospec=True)
  def test_list_subjects(self, mock_response, mock_session):
    mock_session.request.return_value = mock_response
    type(mock_response).status_code = mock.PropertyMock(return_value=200)
    expected = {
        "subjects": [{
            "name":
                "test@test.com",
            "type":
                "SUBJECT_TYPE_ANALYST",
            "roles": [{
                "name":
                    "Test",
                "title":
                    "Test role",
                "description":
                    "The Test role",
                "createTime":
                    "2020-11-05T00:00:00Z",
                "isDefault":
                    False,
                "permissions": [{
                    "name": "Test",
                    "title": "Test permission",
                    "description": "The Test permission",
                    "createTime": "2020-11-05T00:00:00Z",
                },]
            },]
        },]
    }
    mock_response.json.return_value = {"subjects": [expected]}
    actual = list_subjects.list_subjects(mock_session)
    self.assertEqual(actual, [expected])


if __name__ == "__main__":
  unittest.main()
