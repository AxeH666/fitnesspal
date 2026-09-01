"""Deterministic tests for the provider-neutral Barbarik reasoning loop."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timezone
from decimal import Decimal
import json
import unittest
from typing import cast
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.agent_loop import (
    SYSTEM_INSTRUCTIONS,
    TOOL_DEFINITIONS,
    AgentLoopLimitError,
    AgentProtocolError,
    BarbarikReasoningLoop,
    ConversationMessage,
    ModelToolCall,
    ModelTurn,
    ToolDefinition,
    ToolExecutionError,
)
from app.agent_tools import BarbarikAgentTools
from app.models import BodyweightLog, FoodLog, WorkoutLog
from app.services import member_state


ProviderRequest = tuple[
    tuple[ConversationMessage, ...],
    tuple[ToolDefinition, ...],
]


class ScriptedProvider:
    """Return fixed turns while retaining the exact provider transcript."""

    def __init__(self, turns: tuple[ModelTurn, ...]) -> None:
        self._turns = list(turns)
        self.requests: list[ProviderRequest] = []

    @property
    def remaining_turns(self) -> int:
        return len(self._turns)

    def complete(
        self,
        *,
        messages: tuple[ConversationMessage, ...],
        tools: tuple[ToolDefinition, ...],
    ) -> ModelTurn:
        self.requests.append((messages, tools))
        if not self._turns:
            raise AssertionError("scripted provider ran out of turns")
        return self._turns.pop(0)


class AgentReasoningLoopTests(unittest.TestCase):
    """Verify routing, continuation, and safety without a live provider."""

    member_id: UUID
    event_time: datetime
    session_mock: MagicMock
    session: Session
    factory_mock: MagicMock
    factory: Callable[[], Session]
    backend_tools: BarbarikAgentTools

    def setUp(self) -> None:
        self.member_id = uuid4()
        self.event_time = datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc)
        self.session_mock = MagicMock(spec=Session)
        self.session = cast(Session, self.session_mock)
        self.session_mock.__enter__.return_value = self.session
        self.factory_mock = MagicMock(return_value=self.session_mock)
        self.factory = cast(Callable[[], Session], self.factory_mock)
        self.backend_tools = BarbarikAgentTools(
            self.factory,
            self.member_id,
            self.event_time,
        )

    def _loop(
        self,
        *turns: ModelTurn,
    ) -> tuple[BarbarikReasoningLoop, ScriptedProvider]:
        provider = ScriptedProvider(turns)
        return BarbarikReasoningLoop(provider, self.backend_tools), provider

    def _bodyweight(self) -> BodyweightLog:
        return BodyweightLog(
            member_id=self.member_id,
            log_date=date(2026, 8, 31),
            weight_kg=Decimal("79.40"),
            recorded_at=self.event_time,
            note=None,
        )

    def test_weight_command_calls_tool_then_continues_with_ordered_json(self) -> None:
        call = ModelToolCall(
            id="weight-1",
            name="log_bodyweight",
            arguments={"weight_kg": "79.4"},
        )
        loop, provider = self._loop(
            ModelTurn(content="I'll record that.", tool_calls=(call,)),
            ModelTurn(content="Logged your weight as 79.4 kg."),
        )

        with patch(
            "app.agent_tools.member_state.log_bodyweight",
            return_value=self._bodyweight(),
        ) as operation:
            response = loop.respond("Weight 79.4")

        self.assertEqual(response, "Logged your weight as 79.4 kg.")
        operation.assert_called_once_with(
            self.session,
            self.member_id,
            Decimal("79.4"),
            self.event_time,
        )
        self.session_mock.commit.assert_called_once_with()
        self.session_mock.rollback.assert_not_called()
        self.assertEqual(provider.remaining_turns, 0)
        self.assertEqual(len(provider.requests), 2)

        initial_messages, definitions = provider.requests[0]
        self.assertEqual(definitions, TOOL_DEFINITIONS)
        self.assertEqual(
            initial_messages,
            (
                ConversationMessage(
                    role="system",
                    content=SYSTEM_INSTRUCTIONS,
                ),
                ConversationMessage(role="user", content="Weight 79.4"),
            ),
        )

        continued_messages, _ = provider.requests[1]
        self.assertEqual(
            [message.role for message in continued_messages],
            ["system", "user", "assistant", "tool"],
        )
        assistant_message = continued_messages[-2]
        tool_message = continued_messages[-1]
        self.assertEqual(assistant_message.tool_calls, (call,))
        self.assertEqual(tool_message.tool_call_id, "weight-1")
        self.assertEqual(tool_message.tool_name, "log_bodyweight")
        assert tool_message.content is not None
        payload = json.loads(tool_message.content)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["result"]["local_date"], "2026-08-31")
        self.assertEqual(
            Decimal(payload["result"]["weight_kg"]),
            Decimal("79.40"),
        )
        self.assertIn("recorded_at", payload["result"])

    def test_yesterday_workout_query_uses_one_postgres_backed_tool_round(self) -> None:
        yesterday_call = ModelToolCall(
            id="yesterday-1",
            name="get_yesterday_state",
        )
        loop, provider = self._loop(
            ModelTurn(tool_calls=(yesterday_call,)),
            ModelTurn(content="Yesterday you trained barbell bench press."),
        )
        workout = WorkoutLog(
            member_id=self.member_id,
            log_date=date(2026, 8, 30),
            exercises=[
                {
                    "exercise_id": str(uuid4()),
                    "name": "Barbell Bench Press",
                    "sets": [{"reps": 8, "weight_kg": 80}],
                }
            ],
            recorded_at=datetime(2026, 8, 30, 10, 0, tzinfo=timezone.utc),
            note=None,
        )
        yesterday_state = member_state.DayState(
            self.member_id,
            date(2026, 8, 30),
            None,
            (),
            member_state.NutritionTotals(0, 0, 0, 0),
            (workout,),
        )

        with patch(
            "app.agent_tools.member_state.get_yesterday_state",
            return_value=yesterday_state,
        ) as get_yesterday:
            response = loop.respond("What did I train yesterday?")

        self.assertEqual(
            response,
            "Yesterday you trained barbell bench press.",
        )
        get_yesterday.assert_called_once_with(
            self.session,
            self.member_id,
            now=self.event_time,
        )
        self.session_mock.commit.assert_not_called()
        self.assertEqual(len(provider.requests), 2)

        final_request = provider.requests[1][0]
        self.assertEqual(
            [message.role for message in final_request],
            ["system", "user", "assistant", "tool"],
        )
        self.assertEqual(final_request[-1].tool_call_id, "yesterday-1")
        assert final_request[-1].content is not None
        yesterday_payload = json.loads(final_request[-1].content)
        self.assertEqual(
            yesterday_payload["result"]["local_date"],
            "2026-08-30",
        )
        self.assertEqual(
            yesterday_payload["result"]["workout_logs"][0]["local_date"],
            "2026-08-30",
        )

    def test_tentative_calorie_discussion_does_not_mutate(self) -> None:
        loop, provider = self._loop(
            ModelTurn(
                content=(
                    "We can discuss the trade-offs before you choose a calorie "
                    "target."
                )
            )
        )

        response = loop.respond("I'm thinking about lowering calories")

        self.assertIn("discuss", response)
        self.assertEqual(len(provider.requests), 1)
        self.factory_mock.assert_not_called()

    def test_general_advice_returns_directly_without_database_retrieval(self) -> None:
        loop, provider = self._loop(
            ModelTurn(
                content="Progressive overload means gradually increasing the challenge."
            )
        )

        response = loop.respond("What is progressive overload?")

        self.assertIn("gradually increasing", response)
        self.assertEqual(len(provider.requests), 1)
        self.factory_mock.assert_not_called()
        prompt = provider.requests[0][0][0].content
        assert prompt is not None
        self.assertIn("Do not retrieve data merely because", prompt)
        self.assertIn("never invent stored personal facts", prompt)
        self.assertIn("discussion only", prompt)

    def test_user_provided_personal_context_does_not_force_retrieval(self) -> None:
        loop, provider = self._loop(
            ModelTurn(
                content=(
                    "Chicken and rice would make a straightforward balanced meal."
                )
            )
        )

        response = loop.respond(
            "I weigh 79.4 kg and have chicken and rice. What can I make?"
        )

        self.assertIn("Chicken and rice", response)
        self.assertEqual(len(provider.requests), 1)
        self.factory_mock.assert_not_called()
        prompt = provider.requests[0][0][0].content
        assert prompt is not None
        self.assertIn("Facts supplied by the user", prompt)
        self.assertIn("persisted state", prompt)

    def test_tool_registry_is_exact_and_hides_trusted_context(self) -> None:
        expected = {
            "accept_protein_recommendation",
            "calculate_and_store_protein_recommendation",
            "get_bodyweight_for_day",
            "get_day_state",
            "get_exercise_history",
            "get_food_for_day",
            "get_food_totals_for_day",
            "get_latest_bodyweight",
            "get_protein_target_state",
            "get_recent_workouts",
            "get_today_food_totals",
            "get_today_state",
            "get_yesterday_state",
            "get_workouts_for_day",
            "log_bodyweight",
            "log_food",
            "log_workout",
            "override_protein_target",
        }
        names = [definition.name for definition in TOOL_DEFINITIONS]
        self.assertEqual(set(names), expected)
        self.assertEqual(len(names), len(set(names)))

        forbidden = (
            "member_id",
            "event_time",
            "timezone",
            "local_date",
            "session",
            "sql",
            "query",
        )
        for definition in TOOL_DEFINITIONS:
            with self.subTest(tool=definition.name):
                self.assertTrue(definition.description.strip())
                self.assertEqual(definition.input_schema.get("type"), "object")
                self.assertFalse(
                    definition.input_schema.get("additionalProperties", True)
                )
                serialized_schema = json.dumps(definition.input_schema).lower()
                for field in forbidden:
                    self.assertNotIn(field, serialized_schema)

    def test_unknown_and_invalid_calls_return_ordered_failures_without_access(self) -> None:
        unknown_call = ModelToolCall(
            id="unknown-1",
            name="execute_sql",
            arguments={"query": "SELECT * FROM members"},
        )
        invalid_call = ModelToolCall(
            id="invalid-1",
            name="log_bodyweight",
            arguments={
                "weight_kg": "79.4",
                "member_id": str(uuid4()),
            },
        )
        loop, provider = self._loop(
            ModelTurn(tool_calls=(unknown_call, invalid_call)),
            ModelTurn(content="I could not safely perform those requests."),
        )

        response = loop.respond("Run these operations")

        self.assertIn("could not safely", response)
        self.factory_mock.assert_not_called()
        continued = provider.requests[1][0]
        tool_messages = [message for message in continued if message.role == "tool"]
        self.assertEqual(
            [message.tool_call_id for message in tool_messages],
            ["unknown-1", "invalid-1"],
        )
        payloads = [
            json.loads(cast(str, message.content)) for message in tool_messages
        ]
        self.assertEqual(
            [payload["error"]["code"] for payload in payloads],
            ["unknown_tool", "invalid_arguments"],
        )
        self.assertNotIn("Traceback", cast(str, tool_messages[-1].content))

    def test_domain_failure_is_a_tool_result_the_model_can_explain(self) -> None:
        call = ModelToolCall(
            id="weight-0",
            name="log_bodyweight",
            arguments={"weight_kg": "0"},
        )
        loop, provider = self._loop(
            ModelTurn(tool_calls=(call,)),
            ModelTurn(content="Weight must be greater than zero, so I did not log it."),
        )

        with patch(
            "app.agent_tools.member_state.log_bodyweight",
            side_effect=ValueError("weight_kg must be greater than zero"),
        ):
            response = loop.respond("Weight 0")

        self.assertIn("did not log", response)
        self.session_mock.commit.assert_not_called()
        self.session_mock.rollback.assert_called_once_with()
        tool_message = provider.requests[1][0][-1]
        assert tool_message.content is not None
        payload = json.loads(tool_message.content)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "rejected")
        self.assertEqual(
            payload["error"]["message"],
            "weight_kg must be greater than zero",
        )

    def test_unexpected_tool_failure_stops_without_a_fabricated_response(self) -> None:
        call = ModelToolCall(
            id="weight-db",
            name="log_bodyweight",
            arguments={"weight_kg": "79.4"},
        )
        loop, provider = self._loop(
            ModelTurn(tool_calls=(call,)),
            ModelTurn(content="This turn must never be consumed."),
        )

        with patch(
            "app.agent_tools.member_state.log_bodyweight",
            side_effect=RuntimeError("database unavailable"),
        ):
            with self.assertRaisesRegex(
                ToolExecutionError,
                "failed before a reliable tool result",
            ) as raised:
                loop.respond("Weight 79.4")

        self.assertIsInstance(raised.exception.__cause__, RuntimeError)
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(provider.remaining_turns, 1)
        self.session_mock.commit.assert_not_called()
        self.session_mock.rollback.assert_called_once_with()

    def test_malformed_backend_result_stops_as_a_tool_contract_failure(self) -> None:
        call = ModelToolCall(
            id="food-corrupt",
            name="get_food_for_day",
            arguments={"day": "2026-08-31"},
        )
        loop, provider = self._loop(
            ModelTurn(tool_calls=(call,)),
            ModelTurn(content="This turn must never be consumed."),
        )
        malformed_food = FoodLog(
            member_id=self.member_id,
            log_date=date(2026, 8, 31),
            meal_type="lunch",
            items=[{"name": "Rice", "qty": 1, "unit": "bowl"}],
            totals={
                "calories": "not-an-integer",
                "protein_g": 8,
                "carbs_g": 45,
                "fat_g": 2,
            },
            recorded_at=self.event_time,
            note=None,
        )

        with patch(
            "app.agent_tools.member_state.get_food_for_day",
            return_value=(malformed_food,),
        ):
            with self.assertRaisesRegex(
                ToolExecutionError,
                "failed the validated backend tool contract",
            ) as raised:
                loop.respond("What did I eat on August 31?")

        self.assertIsInstance(raised.exception.__cause__, ValidationError)
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(provider.remaining_turns, 1)

    def test_tool_call_ids_must_be_unique_within_the_entire_run(self) -> None:
        duplicate = ModelToolCall(id="duplicate", name="unknown")
        loop, _ = self._loop(ModelTurn(tool_calls=(duplicate, duplicate)))
        with self.assertRaisesRegex(AgentProtocolError, "must be unique"):
            loop.respond("first duplicate case")

        repeated_loop, repeated_provider = self._loop(
            ModelTurn(tool_calls=(duplicate,)),
            ModelTurn(tool_calls=(duplicate,)),
        )
        with self.assertRaisesRegex(AgentProtocolError, "must be unique"):
            repeated_loop.respond("second duplicate case")
        self.assertEqual(len(repeated_provider.requests), 2)
        self.factory_mock.assert_not_called()

    def test_total_tool_calls_are_bounded_across_model_turns(self) -> None:
        first_eight = tuple(
            ModelToolCall(id=f"call-{index}", name="unknown")
            for index in range(8)
        )
        ninth = ModelToolCall(id="call-8", name="unknown")
        loop, provider = self._loop(
            ModelTurn(tool_calls=first_eight),
            ModelTurn(tool_calls=(ninth,)),
        )

        with self.assertRaisesRegex(AgentLoopLimitError, "exceeded 8 tool calls"):
            loop.respond("keep calling tools")

        self.assertEqual(len(provider.requests), 2)
        first_continuation = provider.requests[1][0]
        self.assertEqual(
            len([message for message in first_continuation if message.role == "tool"]),
            8,
        )
        self.factory_mock.assert_not_called()

    def test_invalid_final_turn_and_empty_user_message_are_rejected(self) -> None:
        for content in (None, "", "   "):
            with self.subTest(content=content):
                loop, provider = self._loop(ModelTurn(content=content))
                with self.assertRaisesRegex(
                    AgentProtocolError,
                    "neither tool calls nor a final response",
                ):
                    loop.respond("hello")
                self.assertEqual(len(provider.requests), 1)

        loop, provider = self._loop(ModelTurn(content="unused"))
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            loop.respond("   ")
        self.assertEqual(len(provider.requests), 0)
        self.factory_mock.assert_not_called()

    def test_loop_is_one_shot_so_backend_bound_event_time_cannot_go_stale(self) -> None:
        loop, provider = self._loop(ModelTurn(content="First response."))

        self.assertEqual(loop.respond("first message"), "First response.")
        with self.assertRaisesRegex(RuntimeError, "request-scoped"):
            loop.respond("second message")

        self.assertEqual(len(provider.requests), 1)


if __name__ == "__main__":
    unittest.main()
