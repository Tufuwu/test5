#!/usr/bin/env python3

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
"""Executable and reusable sample for updating GCP log flow filter.

In Chronicle, customers can associate their GCP organizations with their
Chronicle instances to make Chronicle ingestion their logs. On top of that,
customers can control what kinds of logs will go into Chronicle by filters.
This example provides a programmatic way to update such filters.
"""

import argparse
import re
import sys
from typing import Optional, Sequence

from google.auth.transport import requests

from common import chronicle_auth

SERVICE_MANAGEMENT_API_BASE_URL = "https://chronicleservicemanager.googleapis.com"

AUTHORIZATION_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

PATTERN = re.compile(r"[a-z\d]{8}-[a-z\d]{4}-[a-z\d]{4}-[a-z\d]{4}-[a-z\d]{12}")


def initialize_command_line_args(
    args: Optional[Sequence[str]] = None) -> Optional[argparse.Namespace]:
  """Initializes and checks all the command-line arguments."""
  parser = argparse.ArgumentParser()
  chronicle_auth.add_argument_credentials_file(parser)
  parser.add_argument(
      "--organization_id",
      type=int,
      required=True,
      help="the GCP organization ID that the filter belongs to")
  parser.add_argument(
      "--filter_id", type=str, required=True, help="UUID of the filter")
  parser.add_argument(
      "--filter_expression",
      type=str,
      required=True,
      help="new filter expression (syntax: https://cloud.google.com/logging/docs/view/advanced-queries#advanced_logs_query_syntax)"
  )

  # Sanity checks for the command-line arguments.
  parsed_args = parser.parse_args(args)
  if parsed_args.organization_id >= 2**64 or parsed_args.organization_id < 0:
    print("Error: organization ID should not be bigger than 2^64")
    return None
  if PATTERN.fullmatch(parsed_args.filter_id) is None:
    print("Error: filter ID is invalid")
    return None

  return parsed_args


def update_gcp_log_flow_filter(http_session: requests.AuthorizedSession,
                               organization_id: int, filter_id: str,
                               filter_expression: str) -> None:
  """Updates GCP log flow filter for the given GCP organization.

  Args:
    http_session: Authorized session for HTTP requests.
    organization_id: GCP organization ID that the filter belongs to.
    filter_id: UUID for the filter.
    filter_expression: New filter expression (syntax:
      https://cloud.google.com/logging/docs/view/advanced-queries#advanced_logs_query_syntax).
        `log_id("dns.googleapis.com/dns_queries")` is an example.

  Raises:
    requests.exceptions.HTTPError: HTTP request resulted in an error
      (response.status_code >= 400).
  """
  name = f"organizations/{organization_id}/gcpAssociations/{organization_id}/gcpLogFlowFilters/{filter_id}"
  url = f"{SERVICE_MANAGEMENT_API_BASE_URL}/v1/{name}"

  body = {
      "filter": filter_expression,
      # Different states are allowed (e.g., FILTER_STATE_DISABLED).
      "state": "FILTER_STATE_ENABLED",
  }

  response = http_session.request("PATCH", url, json=body)

  if response.status_code >= 400:
    print(response.text)
  response.raise_for_status()


if __name__ == "__main__":
  cli = initialize_command_line_args()
  if not cli:
    sys.exit(1)  # A confidence check failed.

  session = chronicle_auth.initialize_http_session(
      cli.credentials_file, scopes=AUTHORIZATION_SCOPES)
  update_gcp_log_flow_filter(session, cli.organization_id, cli.filter_id,
                             cli.filter_expression)
