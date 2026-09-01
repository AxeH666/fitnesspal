"""Google Gemini adapter for the provider-neutral Barbarik reasoning loop."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
import json
from typing import Any, Protocol, Self, cast

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
import httpx
from pydantic import SecretStr, ValidationError

from app.agent_loop import (
    AgentProtocolError,
    ConversationMessage,
    ModelToolCall,
    ModelTurn,
    ToolDefinition,
)
from app.core.config import Settings, get_settings


class GeminiFailureCategory(StrEnum):
    """Safe, non-sensitive classifications for local diagnostics."""

    RATE_LIMIT = "rate_limit"
    TIMEOUT_NETWORK = "timeout_network"
    AUTHENTICATION = "authentication"
    PROVIDER_ERROR = "provider_error"


class GeminiConfigurationError(ValueError):
    """Raised when the Gemini provider cannot be configured safely."""

    def __init__(
        self,
        message: str,
        *,
        category: GeminiFailureCategory = GeminiFailureCategory.PROVIDER_ERROR,
    ) -> None:
        super().__init__(message)
        self.category = category


class GeminiProviderError(RuntimeError):
    """Raised when Gemini fails before a usable model turn is available."""

    def __init__(
        self,
        message: str,
        *,
        category: GeminiFailureCategory = GeminiFailureCategory.PROVIDER_ERROR,
    ) -> None:
        super().__init__(message)
        self.category = category


_RATE_LIMIT_STATUSES = frozenset(
    {"RATE_LIMIT_EXCEEDED", "RESOURCE_EXHAUSTED"}
)
_RATE_LIMIT_REASONS = frozenset({"RATE_LIMIT_EXCEEDED", "QUOTA_EXCEEDED"})
_TIMEOUT_STATUSES = frozenset(
    {"DEADLINE_EXCEEDED", "GATEWAY_TIMEOUT", "REQUEST_TIMEOUT"}
)
_AUTHENTICATION_STATUSES = frozenset(
    {"AUTHENTICATION", "PERMISSION_DENIED", "UNAUTHENTICATED"}
)
_THINKING_LEVELS = {
    "minimal": types.ThinkingLevel.MINIMAL,
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
}


def _validated_thinking_level(value: str) -> types.ThinkingLevel:
    if not isinstance(value, str):
        raise GeminiConfigurationError(
            "GEMINI_THINKING_LEVEL must be minimal, low, medium, or high"
        )
    resolved = _THINKING_LEVELS.get(value.strip().casefold())
    if resolved is None:
        raise GeminiConfigurationError(
            "GEMINI_THINKING_LEVEL must be minimal, low, medium, or high"
        )
    return resolved


def _validated_request_timeout_ms(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GeminiConfigurationError(
            "GEMINI_REQUEST_TIMEOUT_MS must be a positive integer"
        )
    return value


def _normalized_api_error_reasons(error: genai_errors.APIError) -> tuple[str, ...]:
    """Read only whitelisted structured reason fields from an SDK error."""

    payload = error.details
    if not isinstance(payload, dict):
        return ()
    nested_error = payload.get("error")
    if isinstance(nested_error, dict):
        payload = nested_error
    raw_details = payload.get("details")
    if not isinstance(raw_details, list):
        return ()

    reasons: list[str] = []
    for detail in raw_details:
        if not isinstance(detail, dict):
            continue
        reason = detail.get("reason")
        if isinstance(reason, str):
            reasons.append(reason.strip().upper())
    return tuple(reasons)


def _classify_known_gemini_failure(error: Exception) -> GeminiFailureCategory:
    if isinstance(error, genai_errors.APIError):
        status = error.status
        normalized_status = (
            status.strip().upper() if isinstance(status, str) else ""
        )
        reasons = _normalized_api_error_reasons(error)
        if (
            error.code == 429
            or normalized_status in _RATE_LIMIT_STATUSES
            or any(reason in _RATE_LIMIT_REASONS for reason in reasons)
        ):
            return GeminiFailureCategory.RATE_LIMIT
        if (
            error.code in (408, 504)
            or normalized_status in _TIMEOUT_STATUSES
        ):
            return GeminiFailureCategory.TIMEOUT_NETWORK
        if (
            error.code in (401, 403)
            or normalized_status in _AUTHENTICATION_STATUSES
            or any(reason.startswith("API_KEY_") for reason in reasons)
        ):
            return GeminiFailureCategory.AUTHENTICATION
        return GeminiFailureCategory.PROVIDER_ERROR

    if isinstance(
        error,
        (httpx.TransportError, TimeoutError, ConnectionError),
    ):
        return GeminiFailureCategory.TIMEOUT_NETWORK
    return GeminiFailureCategory.PROVIDER_ERROR


def _classify_gemini_failure(error: Exception) -> GeminiFailureCategory:
    """Reduce an upstream failure to a safe category and retain nothing else."""

    try:
        return _classify_known_gemini_failure(error)
    except Exception:
        return GeminiFailureCategory.PROVIDER_ERROR


class _GeminiChat(Protocol):
    def send_message(
        self,
        message: str | list[types.Part],
    ) -> types.GenerateContentResponse:
        """Send one user or tool-result turn."""


class _GeminiChats(Protocol):
    def create(
        self,
        *,
        model: str,
        config: types.GenerateContentConfig,
    ) -> _GeminiChat:
        """Create one request-scoped Gemini chat."""


class _GeminiClient(Protocol):
    @property
    def chats(self) -> _GeminiChats:
        """Expose the SDK chat factory."""

    def close(self) -> None:
        """Release SDK network resources."""


@dataclass(frozen=True)
class _PendingToolCall:
    neutral_id: str
    native_id: str | None
    name: str


@dataclass(frozen=True)
class _ActiveRun:
    chat: _GeminiChat
    tools: tuple[ToolDefinition, ...]
    expected_prefix: tuple[ConversationMessage, ...]
    pending_calls: tuple[_PendingToolCall, ...]


class GeminiModelProvider:
    """Translate between Gemini and Barbarik's existing model boundary.

    Each instance serves one reasoning run. Its SDK chat is retained across
    that run's tool turns so Gemini can replay native thought signatures
    without adding provider-specific state to the neutral agent loop.
    """

    def __init__(
        self,
        *,
        api_key: SecretStr | None,
        model: str,
        thinking_level: str = "minimal",
        request_timeout_ms: int = 15_000,
        client: _GeminiClient | None = None,
    ) -> None:
        raw_api_key = "" if api_key is None else api_key.get_secret_value().strip()
        if not raw_api_key:
            raise GeminiConfigurationError(
                "GEMINI_API_KEY is required",
                category=GeminiFailureCategory.AUTHENTICATION,
            )
        if not isinstance(model, str) or not model.strip():
            raise GeminiConfigurationError("GEMINI_MODEL must be non-empty")

        self._model = model.strip()
        self._thinking_level = _validated_thinking_level(thinking_level)
        if (
            self._model.casefold() == "gemini-3.7-flash"
            and self._thinking_level == types.ThinkingLevel.MINIMAL
        ):
            raise GeminiConfigurationError(
                "gemini-3.7-flash does not support minimal thinking"
            )
        self._request_timeout_ms = _validated_request_timeout_ms(
            request_timeout_ms
        )
        self._owns_client = client is None
        resolved_client = client
        if resolved_client is None:
            failure_category = GeminiFailureCategory.PROVIDER_ERROR
            try:
                resolved_client = cast(
                    _GeminiClient,
                    genai.Client(api_key=raw_api_key),
                )
            except Exception as error:
                failure_category = _classify_gemini_failure(error)
                resolved_client = None
            if resolved_client is None:
                raise GeminiConfigurationError(
                    "Gemini client initialization failed",
                    category=failure_category,
                )
        self._client = resolved_client
        self._active_run: _ActiveRun | None = None
        self._generated_call_number = 0
        self._has_started = False
        self._closed = False

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> Self:
        """Build the provider from environment-backed application settings."""

        resolved = get_settings() if settings is None else settings
        return cls(
            api_key=resolved.gemini_api_key,
            model=resolved.gemini_model,
            thinking_level=resolved.gemini_thinking_level,
            request_timeout_ms=resolved.gemini_request_timeout_ms,
        )

    def close(self) -> None:
        """Release an owned SDK client and discard request-local chat state."""

        if self._closed:
            return
        self._active_run = None
        self._closed = True
        if self._owns_client:
            failure_category: GeminiFailureCategory | None = None
            try:
                self._client.close()
            except Exception as error:
                failure_category = _classify_gemini_failure(error)
            if failure_category is not None:
                raise GeminiProviderError(
                    "Gemini client close failed",
                    category=failure_category,
                )

    def _abort_run(self) -> None:
        self._active_run = None

    def _request_config(
        self,
        system_instruction: str,
        tools: tuple[ToolDefinition, ...],
    ) -> types.GenerateContentConfig:
        declarations = [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=deepcopy(tool.input_schema),
            )
            for tool in tools
        ]
        gemini_tools = (
            [types.Tool(function_declarations=declarations)]
            if declarations
            else None
        )
        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            thinking_config=types.ThinkingConfig(
                thinking_level=self._thinking_level,
            ),
            http_options=types.HttpOptions(
                timeout=self._request_timeout_ms,
            ),
            # The SDK annotation uses an invariant union list even though a
            # plain list of Tool is the documented input.
            tools=gemini_tools,  # type: ignore[arg-type]
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

    @staticmethod
    def _initial_messages(
        messages: tuple[ConversationMessage, ...],
    ) -> tuple[str, str]:
        if len(messages) != 2:
            raise AgentProtocolError(
                "Gemini provider requires one system message and one user message"
            )
        system_message, user_message = messages
        if (
            system_message.role != "system"
            or system_message.content is None
            or not system_message.content.strip()
            or user_message.role != "user"
            or user_message.content is None
            or not user_message.content.strip()
        ):
            raise AgentProtocolError(
                "Gemini provider requires one system message and one user message"
            )
        return system_message.content, user_message.content

    def _next_generated_call_id(self) -> str:
        self._generated_call_number += 1
        return f"gemini-generated-call-{self._generated_call_number}"

    def _map_response(
        self,
        response: types.GenerateContentResponse,
    ) -> tuple[ModelTurn, tuple[_PendingToolCall, ...]]:
        candidates = response.candidates
        if not candidates:
            raise AgentProtocolError("Gemini returned no usable response candidate")
        candidate = candidates[0]
        if candidate.finish_reason != types.FinishReason.STOP:
            raise AgentProtocolError("Gemini did not complete successfully")
        if candidate.content is None:
            raise AgentProtocolError("Gemini returned no usable response candidate")
        parts = candidate.content.parts
        if not parts:
            raise AgentProtocolError("Gemini returned an empty response candidate")

        text_parts: list[str] = []
        tool_calls: list[ModelToolCall] = []
        pending_calls: list[_PendingToolCall] = []
        call_ids: set[str] = set()

        for part in parts:
            if part.text is not None and part.thought is not True:
                text_parts.append(part.text)

            function_call = part.function_call
            if function_call is None:
                continue

            name = function_call.name
            if not isinstance(name, str) or not name.strip():
                raise AgentProtocolError(
                    "Gemini returned a function call without a name"
                )
            arguments = function_call.args
            if arguments is not None and not isinstance(arguments, dict):
                raise AgentProtocolError(
                    "Gemini returned non-object function arguments"
                )

            native_id = function_call.id
            if native_id is not None:
                if not isinstance(native_id, str) or not native_id.strip():
                    raise AgentProtocolError(
                        "Gemini returned an invalid function call ID"
                    )
                neutral_id = native_id
            else:
                neutral_id = self._next_generated_call_id()

            if neutral_id in call_ids:
                raise AgentProtocolError(
                    "Gemini returned duplicate function call IDs"
                )
            call_ids.add(neutral_id)

            model_call: ModelToolCall | None = None
            try:
                model_call = ModelToolCall(
                    id=neutral_id,
                    name=name,
                    arguments={} if arguments is None else dict(arguments),
                )
            except ValidationError:
                pass
            if model_call is None:
                raise AgentProtocolError(
                    "Gemini returned an invalid function call"
                )
            tool_calls.append(model_call)
            pending_calls.append(
                _PendingToolCall(
                    neutral_id=neutral_id,
                    native_id=native_id,
                    name=name,
                )
            )

        content = "".join(text_parts)
        normalized_content = content if content.strip() else None
        if not tool_calls and normalized_content is None:
            raise AgentProtocolError(
                "Gemini returned neither tool calls nor a final response"
            )
        return (
            ModelTurn(
                content=normalized_content,
                tool_calls=tuple(tool_calls),
            ),
            tuple(pending_calls),
        )

    @staticmethod
    def _tool_result_parts(
        messages: tuple[ConversationMessage, ...],
        active_run: _ActiveRun,
    ) -> list[types.Part]:
        prefix = active_run.expected_prefix
        if messages[: len(prefix)] != prefix:
            raise AgentProtocolError(
                "Gemini continuation does not match the active reasoning run"
            )
        results = messages[len(prefix) :]
        if len(results) != len(active_run.pending_calls):
            raise AgentProtocolError(
                "Gemini continuation must provide every pending tool result once"
            )

        parts: list[types.Part] = []
        for message, pending in zip(results, active_run.pending_calls, strict=True):
            if (
                message.role != "tool"
                or message.tool_call_id != pending.neutral_id
                or message.tool_name != pending.name
                or message.content is None
            ):
                raise AgentProtocolError(
                    "Gemini continuation contains a mismatched tool result"
                )
            payload: object = None
            payload_parse_failed = False
            try:
                payload = json.loads(message.content)
            except (TypeError, json.JSONDecodeError):
                payload_parse_failed = True
            if payload_parse_failed:
                raise AgentProtocolError(
                    "Gemini continuation contains an invalid tool result"
                )
            if not isinstance(payload, dict):
                raise AgentProtocolError(
                    "Gemini continuation tool result must be a JSON object"
                )
            response_payload = cast(dict[str, Any], payload)
            parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=pending.native_id,
                        name=pending.name,
                        response=response_payload,
                    )
                )
            )
        return parts

    def _send(
        self,
        chat: _GeminiChat,
        message: str | list[types.Part],
    ) -> types.GenerateContentResponse:
        failure_category = GeminiFailureCategory.PROVIDER_ERROR
        try:
            return chat.send_message(message)
        except Exception as error:
            failure_category = _classify_gemini_failure(error)
        self._abort_run()
        raise GeminiProviderError(
            "Gemini request failed",
            category=failure_category,
        )

    def _retain_if_needed(
        self,
        *,
        chat: _GeminiChat,
        tools: tuple[ToolDefinition, ...],
        messages: tuple[ConversationMessage, ...],
        turn: ModelTurn,
        pending_calls: tuple[_PendingToolCall, ...],
    ) -> None:
        if not turn.tool_calls:
            self._active_run = None
            return
        assistant_message = ConversationMessage(
            role="assistant",
            content=turn.content,
            tool_calls=turn.tool_calls,
        )
        self._active_run = _ActiveRun(
            chat=chat,
            tools=tools,
            expected_prefix=messages + (assistant_message,),
            pending_calls=pending_calls,
        )

    def complete(
        self,
        *,
        messages: tuple[ConversationMessage, ...],
        tools: tuple[ToolDefinition, ...],
    ) -> ModelTurn:
        """Return one Gemini text response or neutralized tool-call turn."""

        if self._closed:
            raise GeminiProviderError("Gemini provider is closed")

        active_run = self._active_run
        try:
            if active_run is None:
                if self._has_started:
                    raise GeminiProviderError(
                        "Gemini provider is request-scoped and cannot start another run"
                    )
                system_instruction, user_message = self._initial_messages(messages)
                self._has_started = True
                config = self._request_config(system_instruction, tools)
                chat: _GeminiChat | None = None
                failure_category = GeminiFailureCategory.PROVIDER_ERROR
                try:
                    chat = self._client.chats.create(
                        model=self._model,
                        config=config,
                    )
                except Exception as error:
                    failure_category = _classify_gemini_failure(error)
                if chat is None:
                    raise GeminiProviderError(
                        "Gemini request failed",
                        category=failure_category,
                    )
                response = self._send(chat, user_message)
            else:
                if tools != active_run.tools:
                    raise AgentProtocolError(
                        "Gemini tools changed during an active reasoning run"
                    )
                chat = active_run.chat
                response = self._send(
                    chat,
                    self._tool_result_parts(messages, active_run),
                )

            turn, pending_calls = self._map_response(response)
            self._retain_if_needed(
                chat=chat,
                tools=tools,
                messages=messages,
                turn=turn,
                pending_calls=pending_calls,
            )
            return turn
        except (AgentProtocolError, GeminiProviderError):
            self._abort_run()
            raise
