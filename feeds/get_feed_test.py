# Copyright 2022 Google LLC
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
"""Unit tests for the "get_feed" module."""

import unittest
from unittest import mock

from google.auth.transport import requests

from . import get_feed


class GetFeedTest(unittest.TestCase):

  @mock.patch.object(requests, "AuthorizedSession", autospec=True)
  @mock.patch.object(requests.requests, "Response", autospec=True)
  def test_http_error(self, mock_response, mock_session):
    mock_session.request.return_value = mock_response
    type(mock_response).status_code = mock.PropertyMock(return_value=400)
    mock_response.raise_for_status.side_effect = (
        requests.requests.exceptions.HTTPError())

    with self.assertRaises(requests.requests.exceptions.HTTPError):
      get_feed.get_feed(mock_session, "feed name")

  @mock.patch.object(requests, "AuthorizedSession", autospec=True)
  @mock.patch.object(requests.requests, "Response", autospec=True)
  def test_happy_path(self, mock_response, mock_session):
    mock_session.request.return_value = mock_response
    type(mock_response).status_code = mock.PropertyMock(return_value=200)
    expected_feed = {
        "feedDetails": {
            "cortexXdrSettings": {
                "hostname": "api-XXXX.xdr.XX.paloaltonetworks.com",
            },
            "feedType": "CORTEX_XDR",
        },
        "name": "feed name",
    }
    mock_response.json.return_value = expected_feed

    actual_feed = get_feed.get_feed(mock_session, "feed name")
    self.assertEqual(actual_feed, expected_feed)


if __name__ == "__main__":
  unittest.main()
