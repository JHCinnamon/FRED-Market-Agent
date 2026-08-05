"""Run prompt evaluations against the locally running FRED market agent."""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fred_agent import LocalFREDAgent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path(__file__).with_name("prompts.json"),
        help="Path to a JSON evaluation suite.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON evaluation report.",
    )
    return parser.parse_args()


def missing_requirements(requirements: list[str]) -> list[str]:
    environment_variables = {
        "fred": "FRED_API_KEY",
        "twelve_data": "TWELVE_DATA_API_KEY",
    }
    return [
        requirement
        for requirement in requirements
        if requirement in environment_variables and not os.getenv(environment_variables[requirement])
    ]


def evaluate_case(
    case: dict[str, Any], answer: str, chart_series: list[str], activities: list[str]
) -> list[str]:
    failures: list[str] = []
    normalized_answer = answer.casefold()
    minimum_length = int(case.get("min_response_characters", 1))
    if len(answer.strip()) < minimum_length:
        failures.append(f"response is shorter than {minimum_length} characters")

    missing_all = [
        phrase for phrase in case.get("contains_all", []) if phrase.casefold() not in normalized_answer
    ]
    if missing_all:
        failures.append(f"missing required phrases: {', '.join(missing_all)}")

    expected_any = case.get("contains_any", [])
    if expected_any and not any(phrase.casefold() in normalized_answer for phrase in expected_any):
        failures.append(f"missing one of: {', '.join(expected_any)}")

    missing_charts = [
        series for series in case.get("required_chart_series", []) if series not in chart_series
    ]
    if missing_charts:
        failures.append(f"missing chart data for: {', '.join(missing_charts)}")

    missing_activities = [
        prefix
        for prefix in case.get("required_activity_prefixes", [])
        if not any(activity.startswith(prefix) for activity in activities)
    ]
    if missing_activities:
        failures.append(f"missing expected tool activity: {', '.join(missing_activities)}")
    return failures


async def run_case(case: dict[str, Any]) -> dict[str, Any]:
    requirements = case.get("requires", [])
    unavailable = missing_requirements(requirements)
    if unavailable:
        return {
            "id": case["id"],
            "status": "skipped",
            "reason": f"missing configuration for: {', '.join(unavailable)}",
        }

    activities: list[str] = []
    charts: list[str] = []
    agent = LocalFREDAgent(
        activity_callback=activities.append,
        chart_callback=lambda series_id, _observations: charts.append(series_id),
    )
    conversation = case.get("conversation")
    if conversation is None:
        conversation = [{"role": "user", "content": case["prompt"]}]
    started_at = time.monotonic()
    try:
        answer = await agent.run(conversation)
    except Exception as error:
        return {
            "id": case["id"],
            "status": "failed",
            "reason": str(error),
            "duration_seconds": round(time.monotonic() - started_at, 2),
            "activities": activities,
        }

    failures = evaluate_case(case, answer, charts, activities)
    return {
        "id": case["id"],
        "status": "failed" if failures else "passed",
        "reason": "; ".join(failures) if failures else "",
        "duration_seconds": round(time.monotonic() - started_at, 2),
        "answer": answer,
        "chart_series": charts,
        "activities": activities,
    }


async def run_suite(suite_path: Path) -> dict[str, Any]:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = suite.get("cases", [])
    if not cases:
        raise ValueError("The evaluation suite has no cases.")
    results = [await run_case(case) for case in cases]
    counts = {
        status: sum(result["status"] == status for result in results)
        for status in ("passed", "failed", "skipped")
    }
    return {"suite": str(suite_path), "results": results, "summary": counts}


def main() -> int:
    arguments = parse_arguments()
    report = asyncio.run(run_suite(arguments.suite))
    rendered_report = json.dumps(report, indent=2)
    print(rendered_report)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(f"{rendered_report}\n", encoding="utf-8")
    return 1 if report["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())