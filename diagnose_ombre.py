#!/usr/bin/env python3
"""Authenticated CLI for Ombre Brain diagnostics (not an MCP tool)."""

import argparse
import getpass
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar


def _request_json(opener, request):
    with opener.open(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only Ombre Brain diagnostics for maintainers",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("OMBRE_DIAGNOSTICS_URL", ""),
        help="service base URL (or OMBRE_DIAGNOSTICS_URL)",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument("--buckets", action="store_true", help="print the metadata page")
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    args = parser.parse_args()

    if not args.url:
        parser.error("--url or OMBRE_DIAGNOSTICS_URL is required")
    base_url = args.url.rstrip("/")
    password = os.environ.get("OMBRE_DASHBOARD_PASSWORD") or getpass.getpass(
        "Dashboard password: "
    )

    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(CookieJar())
    )
    login = urllib.request.Request(
        base_url + "/auth/login",
        data=json.dumps({"password": password}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    query = urllib.parse.urlencode({
        "offset": max(0, args.offset),
        "limit": max(1, min(100, args.limit)),
        "include_archive": "false" if args.no_archive else "true",
    })

    try:
        _request_json(opener, login)
        data = _request_json(
            opener,
            urllib.request.Request(base_url + "/api/admin/diagnostics?" + query),
        )
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: diagnostic request failed", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"diagnostic request failed: {type(exc).__name__}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    summary = data["summary"]
    print(
        f"total={summary['total_buckets']} active={summary['active_buckets']} "
        f"today={summary['today_added']} undigested={summary['undigested']} "
        f"feel={summary['feel_count']} tagging_failures={summary['tagging_failure_count']} "
        f"decay={summary['decay_engine']} embedding={summary['embedding_enabled']}"
    )
    anomalies = data.get("anomalies", [])
    if anomalies:
        for item in anomalies:
            samples = ",".join(item.get("sample_ids", []))
            print(f"[{item['code']}] {item['message']}" + (f" samples={samples}" if samples else ""))
    else:
        print("anomalies: none")

    if args.buckets:
        for bucket in data.get("buckets", []):
            print(
                f"{bucket['id']} {bucket['type']} score={bucket['score']:.2f} "
                f"created={bucket['created']} name={bucket['name']}"
            )
        page = data["pagination"]
        print(
            f"page offset={page['offset']} returned={page['returned']} total={page['total']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
