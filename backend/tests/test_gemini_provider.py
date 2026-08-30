"""Mocked tests for the concrete Google Gemini model adapter."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import json
import os
import traceback
import unittest
from typing import cast
from unittest.mock import MagicMock, patch
from uuid import uuid4

from google.genai import types
from pydantic import SecretStr
from sqlalchemy.orm import Session

from app.agent_loop import (
    TOOL_DEFINITIONS,
    AgentProtocolError,
    BarbarikReasoningLoop,
    ConversationMessage,
    ModelToolCall,
    ModelTurn,
    ToolDefinition,
)
from app.agent_tools import BarbarikAgentTools
from app.core.config import Settings
from app.gemini_provider import (
    GeminiConfigurationError,
    GeminiModelProvider,
    GeminiProviderError,
)


GeminiMessage = str | list[types.Part]


class FakeChat:
    """Return native Gemini responses while retaining sent chat messages."""

    def __init__(
        self,
        responses: list[types.GenerateContentResponse | Exception],
    ) -> None:
        self._responses = responses
        self.messages: list[GeminiMessage] = []

    def send_message(self, message: GeminiMessage) -> types.GenerateContentResponse:
        self.messages.append(message)
        if not self._responses:
            raise AssertionError("fake Gemini chat ran out of responses")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeChats:
    """Capture chat construction without making a network request."""

    def __init__(self, chat: FakeChat) -> None:
        self._chat = chat
        self.requests: list[tuple[str, types.GenerateContentConfig]] = []

    def create(
        self,
        *,
        model: str,
        config: types.GenerateContentConfig,
    ) -> FakeChat:
        self.requests.append((model, config))
        return self._chat


class FakeClient:
    def __init__(self, chat: FakeChat) -> None:
        self._chats = FakeChats(chat)
        self.close_calls = 0

    @property
    def chats(self) -> FakeChats:
        return self._chats

    def close(self) -> None:
        self.close_calls += 1


def _response(
    *parts: types.Part,
    finish_reason: types.FinishReason = types.FinishReason.STOP,
) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=list(parts)),
                finish_reason=finish_reason,
            )
        ]
    )


def _messages(user_message: str = "Hello") -> tuple[ConversationMessage, ...]:
    return (
        ConversationMessage(role="system", content="System policy"),
        ConversationMessage(role="user", content=user_message),
    )


NO_ARGUMENT_TOOL = ToolDefinition(
    name="get_today_state",
    description="Read today's authoritative member state.",
    input_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
)


class GeminiProviderTests(unittest.TestCase):
    """Verify serialization, mapping, continuation, and safe failures."""

    def _assert_secret_detached(
        self,
        error: BaseException,
        secret: str,
    ) -> None:
        self.assertIsNone(error.__cause__)
        self.assertIsNone(error.__context__)
        self.assertNotIn(secret, repr(error))

    def _provider(
        self,
        *responses: types.GenerateContentResponse | Exception,
        model: str = "gemini-test-model",
    ) -> tuple[GeminiModelProvider, FakeClient, FakeChat]:
        chat = FakeChat(list(responses))
        client = FakeClient(chat)
        provider = GeminiModelProvider(
            api_key=SecretStr("unit-test-key"),
            model=model,
            client=client,
        )
        return provider, client, chat

    def test_request_serializes_model_system_instruction_and_tool_schema(self) -> None:
        secret = "test-key-must-not-leak"
        chat = FakeChat([_response(types.Part(text="Direct answer."))])
        client = FakeClient(chat)
        provider = GeminiModelProvider(
            api_key=SecretStr(secret),
            model="gemini-configured-model",
            client=client,
        )

        turn = provider.complete(messages=_messages(), tools=TOOL_DEFINITIONS)

        self.assertEqual(turn, ModelTurn(content="Direct answer."))
        self.assertEqual(chat.messages, ["Hello"])
        self.assertEqual(len(client.chats.requests), 1)
        model, config = client.chats.requests[0]
        self.assertEqual(model, "gemini-configured-model")
        request = config.model_dump(exclude_none=True)
        self.assertEqual(request["system_instruction"], "System policy")
        self.assertTrue(request["automatic_function_calling"]["disable"])
        declarations = request["tools"][0]["function_declarations"]
        self.assertEqual(
            declarations,
            [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters_json_schema": tool.input_schema,
                }
                for tool in TOOL_DEFINITIONS
            ],
        )
        serialized_request = json.dumps(request)
        self.assertNotIn(secret, serialized_request)
        self.assertNotIn(secret, repr(provider))

    def test_text_mapping_omits_internal_thought_parts(self) -> None:
        provider, _, _ = self._provider(
            _response(
                types.Part(text="private reasoning", thought=True),
                types.Part(text="Useful final answer."),
            )
        )

        turn = provider.complete(messages=_messages(), tools=(NO_ARGUMENT_TOOL,))

        self.assertEqual(turn, ModelTurn(content="Useful final answer."))
        self.assertNotIn("private reasoning", cast(str, turn.content))

    def test_native_tool_call_maps_to_the_neutral_format(self) -> None:
        provider, _, _ = self._provider(
            _response(
                types.Part(
                    function_call=types.FunctionCall(
                        id="gemini-call-1",
                        name="get_today_state",
                        args={},
                    )
                )
            )
        )

        turn = provider.complete(messages=_messages(), tools=(NO_ARGUMENT_TOOL,))

        self.assertEqual(
            turn,
            ModelTurn(
                tool_calls=(
                    ModelToolCall(
                        id="gemini-call-1",
                        name="get_today_state",
                    ),
                )
            ),
        )

    def test_missing_native_call_id_gets_a_neutral_id_but_is_not_invented_for_gemini(self) -> None:
        provider, _, chat = self._provider(
            _response(
                types.Part(
                    function_call=types.FunctionCall(
                        name="get_today_state",
                        args={},
                    )
                )
            ),
            _response(types.Part(text="Continuation complete.")),
        )
        initial = _messages()
        first_turn = provider.complete(
            messages=initial,
            tools=(NO_ARGUMENT_TOOL,),
        )
        generated_id = first_turn.tool_calls[0].id
        continuation = initial + (
            ConversationMessage(
                role="assistant",
                tool_calls=first_turn.tool_calls,
            ),
            ConversationMessage(
                role="tool",
                content='{"ok":true,"result":null}',
                tool_call_id=generated_id,
                tool_name="get_today_state",
            ),
        )

        final_turn = provider.complete(
            messages=continuation,
            tools=(NO_ARGUMENT_TOOL,),
        )

        self.assertTrue(generated_id.startswith("gemini-generated-call-"))
        self.assertEqual(final_turn.content, "Continuation complete.")
        response_parts = cast(list[types.Part], chat.messages[1])
        function_response = response_parts[0].function_response
        assert function_response is not None
        self.assertIsNone(function_response.id)
        self.assertEqual(function_response.name, "get_today_state")

    def test_parallel_tool_results_continue_together_in_original_order(self) -> None:
        provider, _, chat = self._provider(
            _response(
                types.Part(
                    function_call=types.FunctionCall(
                        id="call-a",
                        name="get_today_state",
                        args={},
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        id="call-b",
                        name="get_today_state",
                        args={},
                    )
                ),
            ),
            _response(types.Part(text="Both results received.")),
        )
        initial = _messages()
        first_turn = provider.complete(
            messages=initial,
            tools=(NO_ARGUMENT_TOOL,),
        )
        continuation = initial + (
            ConversationMessage(
                role="assistant",
                tool_calls=first_turn.tool_calls,
            ),
            ConversationMessage(
                role="tool",
                content='{"ok":true,"result":{"order":1}}',
                tool_call_id="call-a",
                tool_name="get_today_state",
            ),
            ConversationMessage(
                role="tool",
                content='{"ok":true,"result":{"order":2}}',
                tool_call_id="call-b",
                tool_name="get_today_state",
            ),
        )

        turn = provider.complete(
            messages=continuation,
            tools=(NO_ARGUMENT_TOOL,),
        )

        self.assertEqual(turn.content, "Both results received.")
        self.assertEqual(len(chat.messages), 2)
        response_parts = cast(list[types.Part], chat.messages[1])
        self.assertEqual(len(response_parts), 2)
        function_responses = [part.function_response for part in response_parts]
        self.assertEqual(
            [response.id if response is not None else None for response in function_responses],
            ["call-a", "call-b"],
        )
        self.assertEqual(
            [
                response.response["result"]["order"]
                if response is not None and response.response is not None
                else None
                for response in function_responses
            ],
            [1, 2],
        )

    def test_two_sequential_tool_turns_use_one_native_chat(self) -> None:
        provider, client, chat = self._provider(
            _response(
                types.Part(
                    function_call=types.FunctionCall(
                        id="sequential-1",
                        name="get_today_state",
                        args={},
                    )
                )
            ),
            _response(
                types.Part(
                    function_call=types.FunctionCall(
                        id="sequential-2",
                        name="get_today_state",
                        args={},
                    )
                )
            ),
            _response(types.Part(text="All reasoning turns complete.")),
        )
        initial = _messages()
        first_turn = provider.complete(
            messages=initial,
            tools=(NO_ARGUMENT_TOOL,),
        )
        first_continuation = initial + (
            ConversationMessage(
                role="assistant",
                tool_calls=first_turn.tool_calls,
            ),
            ConversationMessage(
                role="tool",
                content='{"ok":true,"result":{"turn":1}}',
                tool_call_id="sequential-1",
                tool_name="get_today_state",
            ),
        )
        second_turn = provider.complete(
            messages=first_continuation,
            tools=(NO_ARGUMENT_TOOL,),
        )
        second_continuation = first_continuation + (
            ConversationMessage(
                role="assistant",
                tool_calls=second_turn.tool_calls,
            ),
            ConversationMessage(
                role="tool",
                content='{"ok":true,"result":{"turn":2}}',
                tool_call_id="sequential-2",
                tool_name="get_today_state",
            ),
        )

        final_turn = provider.complete(
            messages=second_continuation,
            tools=(NO_ARGUMENT_TOOL,),
        )

        self.assertEqual(final_turn.content, "All reasoning turns complete.")
        self.assertEqual(len(client.chats.requests), 1)
        self.assertEqual(len(chat.messages), 3)
        continued_ids = []
        for message in chat.messages[1:]:
            response_parts = cast(list[types.Part], message)
            function_response = response_parts[0].function_response
            assert function_response is not None
            continued_ids.append(function_response.id)
        self.assertEqual(continued_ids, ["sequential-1", "sequential-2"])

    def test_provider_instance_cannot_leak_state_into_a_second_request(self) -> None:
        provider, client, _ = self._provider(
            _response(types.Part(text="First request complete."))
        )

        first_turn = provider.complete(
            messages=_messages("First request"),
            tools=(NO_ARGUMENT_TOOL,),
        )

        self.assertEqual(first_turn.content, "First request complete.")
        with self.assertRaisesRegex(GeminiProviderError, "request-scoped"):
            provider.complete(
                messages=_messages("Second request"),
                tools=(NO_ARGUMENT_TOOL,),
            )
        self.assertEqual(len(client.chats.requests), 1)

    def test_real_agent_loop_executes_tool_then_continues_on_the_same_chat(self) -> None:
        provider, client, chat = self._provider(
            _response(
                types.Part(
                    function_call=types.FunctionCall(
                        id="latest-1",
                        name="get_latest_bodyweight",
                        args={},
                    )
                )
            ),
            _response(types.Part(text="No bodyweight has been logged yet.")),
        )
        session_context = MagicMock(spec=Session)
        session = cast(Session, MagicMock(spec=Session))
        session_context.__enter__.return_value = session
        session_factory_mock = MagicMock(return_value=session_context)
        session_factory = cast(Callable[[], Session], session_factory_mock)
        tools = BarbarikAgentTools(
            session_factory,
            uuid4(),
            datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc),
        )

        with patch(
            "app.agent_tools.member_state.get_latest_bodyweight",
            return_value=None,
        ) as get_latest:
            answer = BarbarikReasoningLoop(provider, tools).respond(
                "What is my latest weight?"
            )

        self.assertEqual(answer, "No bodyweight has been logged yet.")
        get_latest.assert_called_once()
        self.assertEqual(len(client.chats.requests), 1)
        self.assertEqual(len(chat.messages), 2)
        response_parts = cast(list[types.Part], chat.messages[1])
        function_response = response_parts[0].function_response
        assert function_response is not None
        self.assertEqual(function_response.id, "latest-1")
        self.assertEqual(function_response.response, {"ok": True, "result": None})

    def test_malformed_gemini_responses_are_rejected(self) -> None:
        malformed = (
            types.GenerateContentResponse(candidates=[]),
            types.GenerateContentResponse(
                candidates=[
                    types.Candidate(
                        content=types.Content(parts=[]),
                        finish_reason=types.FinishReason.STOP,
                    )
                ]
            ),
            _response(types.Part(text="hidden only", thought=True)),
            _response(
                types.Part(
                    function_call=types.FunctionCall(
                        id="call-without-name",
                        args={},
                    )
                )
            ),
        )

        for response in malformed:
            with self.subTest(response=response.model_dump(exclude_none=True)):
                provider, _, _ = self._provider(response)
                with self.assertRaises(AgentProtocolError):
                    provider.complete(
                        messages=_messages(),
                        tools=(NO_ARGUMENT_TOOL,),
                    )

    def test_unsuccessful_finish_reasons_are_rejected(self) -> None:
        failure_reasons = (
            types.FinishReason.MAX_TOKENS,
            types.FinishReason.SAFETY,
            types.FinishReason.MALFORMED_FUNCTION_CALL,
            types.FinishReason.UNEXPECTED_TOOL_CALL,
            types.FinishReason.TOO_MANY_TOOL_CALLS,
        )

        for finish_reason in failure_reasons:
            with self.subTest(finish_reason=finish_reason):
                provider, _, _ = self._provider(
                    _response(
                        types.Part(text="partial or blocked output"),
                        finish_reason=finish_reason,
                    )
                )
                with self.assertRaisesRegex(
                    AgentProtocolError,
                    "did not complete successfully",
                ):
                    provider.complete(
                        messages=_messages(),
                        tools=(NO_ARGUMENT_TOOL,),
                    )

    def test_continuation_rejects_transcript_mismatch_and_invalid_json(self) -> None:
        for tool_content in (None, "not-json", "[]"):
            with self.subTest(tool_content=tool_content):
                provider, _, _ = self._provider(
                    _response(
                        types.Part(
                            function_call=types.FunctionCall(
                                id="pending-1",
                                name="get_today_state",
                                args={},
                            )
                        )
                    )
                )
                initial = _messages()
                first_turn = provider.complete(
                    messages=initial,
                    tools=(NO_ARGUMENT_TOOL,),
                )
                bad_continuation = initial + (
                    ConversationMessage(
                        role="assistant",
                        tool_calls=first_turn.tool_calls,
                    ),
                    ConversationMessage(
                        role="tool",
                        content=tool_content,
                        tool_call_id="pending-1",
                        tool_name="get_today_state",
                    ),
                )
                with self.assertRaises(AgentProtocolError):
                    provider.complete(
                        messages=bad_continuation,
                        tools=(NO_ARGUMENT_TOOL,),
                    )

    def test_sdk_failure_is_sanitized_and_does_not_leak_the_key(self) -> None:
        secret = "key-visible-only-inside-the-fake-sdk"
        chat = FakeChat([RuntimeError(f"upstream failure with {secret}")])
        provider = GeminiModelProvider(
            api_key=SecretStr(secret),
            model="gemini-test-model",
            client=FakeClient(chat),
        )

        with self.assertRaises(GeminiProviderError) as raised:
            provider.complete(
                messages=_messages(),
                tools=(NO_ARGUMENT_TOOL,),
            )

        rendered = "".join(
            traceback.format_exception(
                type(raised.exception),
                raised.exception,
                raised.exception.__traceback__,
            )
        )
        self.assertEqual(str(raised.exception), "Gemini request failed")
        self.assertNotIn(secret, rendered)
        self._assert_secret_detached(raised.exception, secret)
        self.assertNotIn(secret, repr(provider))

    def test_all_sdk_boundary_errors_detach_the_upstream_exception(self) -> None:
        secret = "upstream-context-must-not-survive"

        with patch(
            "app.gemini_provider.genai.Client",
            side_effect=RuntimeError(secret),
        ):
            with self.assertRaises(GeminiConfigurationError) as initialization:
                GeminiModelProvider(
                    api_key=SecretStr(secret),
                    model="gemini-test-model",
                )
        self._assert_secret_detached(initialization.exception, secret)

        provider, client, _ = self._provider()
        with patch.object(
            client.chats,
            "create",
            side_effect=RuntimeError(secret),
        ):
            with self.assertRaises(GeminiProviderError) as creation:
                provider.complete(
                    messages=_messages(),
                    tools=(NO_ARGUMENT_TOOL,),
                )
        self._assert_secret_detached(creation.exception, secret)

        sdk_client = MagicMock()
        sdk_client.close.side_effect = RuntimeError(secret)
        with patch(
            "app.gemini_provider.genai.Client",
            return_value=sdk_client,
        ):
            provider = GeminiModelProvider(
                api_key=SecretStr(secret),
                model="gemini-test-model",
            )
        with self.assertRaises(GeminiProviderError) as closing:
            provider.close()
        self._assert_secret_detached(closing.exception, secret)

    def test_missing_or_blank_api_key_is_rejected_before_client_use(self) -> None:
        for api_key in (None, SecretStr(""), SecretStr("   ")):
            with self.subTest(api_key=api_key):
                with self.assertRaisesRegex(
                    GeminiConfigurationError,
                    "GEMINI_API_KEY is required",
                ):
                    GeminiModelProvider(
                        api_key=api_key,
                        model="gemini-test-model",
                        client=FakeClient(FakeChat([])),
                    )

    def test_environment_settings_are_explicit_and_secret_is_redacted(self) -> None:
        secret = "environment-only-test-key"
        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": secret,
                "GEMINI_MODEL": "gemini-env-model",
            },
        ):
            settings = Settings()

        self.assertEqual(settings.gemini_model, "gemini-env-model")
        assert settings.gemini_api_key is not None
        self.assertEqual(settings.gemini_api_key.get_secret_value(), secret)
        self.assertNotIn(secret, repr(settings))

        sdk_client = MagicMock()
        with patch(
            "app.gemini_provider.genai.Client",
            return_value=sdk_client,
        ) as constructor:
            provider = GeminiModelProvider.from_settings(settings)
            constructor.assert_called_once_with(api_key=secret)
            self.assertNotIn(secret, repr(provider))
            provider.close()
        sdk_client.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
