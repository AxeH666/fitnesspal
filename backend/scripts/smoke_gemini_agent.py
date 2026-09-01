"""Run one local Barbarik reasoning request against Google Gemini."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import perf_counter_ns
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent_loop import (  # noqa: E402
    AgentLoopLimitError,
    AgentProtocolError,
    BarbarikReasoningLoop,
    ConversationMessage,
    ModelProvider,
    ModelToolCall,
    ModelTurn,
    ToolDefinition,
    ToolExecutionError,
)
from app.agent_tools import BarbarikAgentTools  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.gemini_provider import (  # noqa: E402
    GeminiConfigurationError,
    GeminiModelProvider,
    GeminiProviderError,
)
from app.models import Member  # noqa: E402


_DEFAULT_MEMBER_EMAIL = "rajesh.demo@barbarik.local"


@dataclass
class _TurnTimings:
    model_call_ns: list[int] = field(default_factory=list)
    tool_ns: int = 0
    tool_calls: int = 0
    total_ns: int | None = None


class _TimedModelProvider:
    def __init__(
        self,
        provider: ModelProvider,
        timings: _TurnTimings,
        clock: Callable[[], int],
    ) -> None:
        self._provider = provider
        self._timings = timings
        self._clock = clock

    def complete(
        self,
        *,
        messages: tuple[ConversationMessage, ...],
        tools: tuple[ToolDefinition, ...],
    ) -> ModelTurn:
        started = self._clock()
        try:
            return self._provider.complete(messages=messages, tools=tools)
        finally:
            self._timings.model_call_ns.append(self._clock() - started)


class _TimedReasoningLoop(BarbarikReasoningLoop):
    def __init__(
        self,
        provider: ModelProvider,
        backend_tools: BarbarikAgentTools,
        timings: _TurnTimings,
        clock: Callable[[], int],
    ) -> None:
        super().__init__(provider, backend_tools)
        self._timings = timings
        self._clock = clock

    def _execute_tool(self, call: ModelToolCall) -> ConversationMessage:
        started = self._clock()
        try:
            return super()._execute_tool(call)
        finally:
            self._timings.tool_calls += 1
            self._timings.tool_ns += self._clock() - started


def _respond_with_timings(
    provider: ModelProvider,
    backend_tools: BarbarikAgentTools,
    user_message: str,
    timings: _TurnTimings,
    *,
    clock: Callable[[], int] | None = None,
) -> str:
    resolved_clock = perf_counter_ns if clock is None else clock
    timed_provider = _TimedModelProvider(provider, timings, resolved_clock)
    loop = _TimedReasoningLoop(
        timed_provider,
        backend_tools,
        timings,
        resolved_clock,
    )
    started = resolved_clock()
    try:
        return loop.respond(user_message)
    finally:
        timings.total_ns = resolved_clock() - started


def _safe_timing_lines(timings: _TurnTimings, app_env: str) -> tuple[str, ...]:
    if app_env.strip().casefold() != "development":
        return ()

    lines: list[str] = []
    if timings.model_call_ns:
        lines.append(f"model_1_ms={timings.model_call_ns[0] / 1_000_000:.1f}")
    if timings.tool_calls:
        lines.append(f"tool_ms={timings.tool_ns / 1_000_000:.1f}")
    lines.extend(
        f"model_{index}_ms={duration / 1_000_000:.1f}"
        for index, duration in enumerate(timings.model_call_ns[1:], start=2)
    )
    if timings.total_ns is not None:
        lines.append(f"total_ms={timings.total_ns / 1_000_000:.1f}")
    return tuple(lines)


def _safe_gemini_failure_message(
    error: GeminiConfigurationError | GeminiProviderError,
    app_env: str,
) -> str:
    """Render fixed safe text, with a category only in local development."""

    message = (
        "Gemini configuration failed"
        if isinstance(error, GeminiConfigurationError)
        else "Gemini provider failed"
    )
    if app_env.strip().casefold() != "development":
        return message
    return f"{message} [category={error.category.value}]"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one request through the local Barbarik Gemini loop."
    )
    parser.add_argument("message", help="User message to send to Barbarik")
    parser.add_argument(
        "--member-email",
        default=_DEFAULT_MEMBER_EMAIL,
        help="Seeded member email used for authoritative personal state",
    )
    return parser.parse_args()


def _member_id(email: str) -> UUID:
    with SessionLocal() as session:
        member_id = session.scalar(select(Member.id).where(Member.email == email))
    if member_id is None:
        raise LookupError(f"No member found for {email}")
    return member_id


def main() -> int:
    args = _arguments()
    if not args.message.strip():
        print("Smoke test message must be non-empty", file=sys.stderr)
        return 2
    settings = get_settings()
    try:
        provider = GeminiModelProvider.from_settings(settings)
    except GeminiConfigurationError as error:
        failure = _safe_gemini_failure_message(error, settings.app_env)
        print(f"Smoke test could not start: {failure}", file=sys.stderr)
        return 2

    exit_code = 0
    timings: _TurnTimings | None = None
    try:
        try:
            member_id = _member_id(args.member_email)
        except LookupError:
            print("Smoke test could not start: member not found", file=sys.stderr)
            exit_code = 2
        except SQLAlchemyError:
            print(
                "Smoke test could not reach the local seeded database",
                file=sys.stderr,
            )
            exit_code = 2
        else:
            backend_tools = BarbarikAgentTools(
                SessionLocal,
                member_id,
                datetime.now(timezone.utc),
            )
            timings = _TurnTimings()
            response = _respond_with_timings(
                provider,
                backend_tools,
                args.message,
                timings,
            )
            print(response)
    except GeminiProviderError as error:
        print(
            _safe_gemini_failure_message(error, settings.app_env),
            file=sys.stderr,
        )
        exit_code = 1
    except (
        AgentLoopLimitError,
        AgentProtocolError,
        ToolExecutionError,
    ) as error:
        print(str(error), file=sys.stderr)
        exit_code = 1
    finally:
        try:
            provider.close()
        except GeminiProviderError as error:
            print(
                _safe_gemini_failure_message(error, settings.app_env),
                file=sys.stderr,
            )
            if exit_code == 0:
                exit_code = 1
    if timings is not None:
        for line in _safe_timing_lines(timings, settings.app_env):
            print(line, file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
