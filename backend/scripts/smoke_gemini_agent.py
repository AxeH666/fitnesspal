"""Run one local Barbarik reasoning request against Google Gemini."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
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
        print(f"Smoke test could not start: {error}", file=sys.stderr)
        return 2

    try:
        try:
            member_id = _member_id(args.member_email)
        except LookupError as error:
            print(f"Smoke test could not start: {error}", file=sys.stderr)
            return 2
        except SQLAlchemyError:
            print(
                "Smoke test could not reach the local seeded database",
                file=sys.stderr,
            )
            return 2

        backend_tools = BarbarikAgentTools(
            SessionLocal,
            member_id,
            datetime.now(timezone.utc),
        )
        response = BarbarikReasoningLoop(provider, backend_tools).respond(args.message)
        print(response)
    except (
        AgentLoopLimitError,
        AgentProtocolError,
        GeminiProviderError,
        ToolExecutionError,
    ) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        provider.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
