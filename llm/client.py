from __future__ import annotations

import copy
import json
import os
import threading
import time

import urllib3
from openai import OpenAI

from config import default_config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


MOCK_INITIALIZATION_RESPONSE = """<ROUTE>[{"Molecule set":["CCO"],"Rational":"Mock step.","Product":["CCO"],"Reaction":"C>>C","Reactants":["CCO"],"Updated molecule set":["CCO"]}]</ROUTE><EXPLANATION>Mock response.</EXPLANATION>"""
MOCK_MUTATION_RESPONSE = MOCK_INITIALIZATION_RESPONSE
FAILED_RESPONSE = "<ROUTE></ROUTE><EXPLANATION>LLM API Call Failed</EXPLANATION>"


class LLMAccountError(BaseException):
    """Fatal account/API-key problem; stop the current experiment worker."""


class LLMServiceError(BaseException):
    """Transient provider/network problem; stop the worker without recording a route failure."""


class MockLLMClient:
    def __init__(self, api_key: str, model_name: str):
        self.model_name = model_name
        print(f"[LLM][WARN] Using mock client for {model_name}.")

    def query(self, prompt: str, temperature: float, system_prompt: str | None = None) -> str:
        time.sleep(0.01)
        if "MUTATION" in prompt:
            return MOCK_MUTATION_RESPONSE
        return MOCK_INITIALIZATION_RESPONSE

    def get_usage_snapshot(self):
        return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def print_global_stats(self, save_to_file=None):
        print("[LLM][INFO] Mock client: no token usage.")


class OpenAIProxyClient:
    def __init__(self, api_key: str, model_name: str, base_url: str):
        print(f"[LLM][INFO] Initializing client: model={model_name}, base_url={base_url}")
        if not api_key or "YOUR_API_KEY" in api_key:
            raise ValueError("API key is not configured.")
        if not base_url or not base_url.startswith("https://"):
            raise ValueError("Base URL is not configured correctly.")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=getattr(default_config, "LLM_TIMEOUT_SECONDS", 90),
        )
        self.model_name = model_name
        self.base_url = base_url
        self.global_stats = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.consecutive_failures = 0
        self.cooldown_until = 0.0
        self._state_lock = threading.Lock()
        print(f"[LLM][INFO] Client ready: model={model_name}")

    def get_usage_snapshot(self):
        with self._state_lock:
            return copy.deepcopy(self.global_stats)

    def print_global_stats(self, save_to_file="total_token_usage.json"):
        stats = self.get_usage_snapshot()
        print(
            "[LLM][INFO] Token usage: "
            f"model={self.model_name}, calls={stats['calls']}, "
            f"prompt_tokens={stats['prompt_tokens']}, "
            f"completion_tokens={stats['completion_tokens']}, "
            f"total_tokens={stats['total_tokens']}"
        )

        if save_to_file:
            try:
                save_path = (
                    os.path.join(default_config.OUTPUT_DIR, save_to_file)
                    if not os.path.isabs(save_to_file)
                    else save_to_file
                )
                with open(save_path, "w", encoding="utf-8") as handle:
                    json.dump(stats, handle, indent=4)
                print(f"[LLM][INFO] Token usage saved: {save_path}")
            except Exception as exc:
                print(f"[LLM][WARN] Failed to save token usage: {exc}")

    def query(self, prompt: str, temperature: float, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        max_retries = getattr(default_config, "LLM_MAX_RETRIES", 2)
        service_recovery_attempts = max(
            0,
            int(getattr(default_config, "LLM_SERVICE_RECOVERY_ATTEMPTS", 0)),
        )
        service_recoveries = 0
        response_data_str = None

        while True:
            cooldown_wait = self._cooldown_remaining()
            if cooldown_wait > 0:
                if service_recoveries < service_recovery_attempts:
                    service_recoveries += 1
                    print(
                        "[LLM][WARN] Cooldown active; waiting "
                        f"{cooldown_wait}s before recovery attempt "
                        f"{service_recoveries}/{service_recovery_attempts}."
                    )
                    time.sleep(cooldown_wait)
                    continue
                print(f"[LLM][WARN] Cooldown active; skipping call for {cooldown_wait}s.")
                raise LLMServiceError(f"API cooldown active for {cooldown_wait}s")

            for attempt in range(max_retries + 1):
                try:
                    completion_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                    )

                    if hasattr(completion_response, "usage") and completion_response.usage:
                        usage = completion_response.usage
                        with self._state_lock:
                            self.global_stats["calls"] += 1
                            self.global_stats["prompt_tokens"] += usage.prompt_tokens
                            self.global_stats["completion_tokens"] += usage.completion_tokens
                            self.global_stats["total_tokens"] += usage.total_tokens

                    if isinstance(completion_response, str):
                        response_data_str = completion_response
                        data_to_parse = json.loads(completion_response)
                    elif hasattr(completion_response, "model_dump"):
                        data_to_parse = completion_response.model_dump()
                    else:
                        raise TypeError(f"Unexpected API response type: {type(completion_response)}")

                    if not data_to_parse:
                        raise ValueError("Empty response struct")
                    response_text = data_to_parse["choices"][0]["message"]["content"]
                    if not response_text:
                        raise ValueError("Empty content")
                    with self._state_lock:
                        self.consecutive_failures = 0
                    return response_text

                except json.JSONDecodeError:
                    print("[LLM][ERROR] API response JSON parsing failed.")
                    if response_data_str:
                        print(f"[LLM][DEBUG] Raw response: {response_data_str}")
                    return "<ROUTE></ROUTE><EXPLANATION>LLM API Call Failed (JSONDecodeError)</EXPLANATION>"
                except Exception as exc:
                    brief = str(exc).replace("\n", " ")[:240]
                    if self._is_account_error(exc):
                        print(f"[LLM][FATAL] Account/API-key error: {brief}")
                        raise LLMAccountError(brief) from exc
                    retryable = self._is_retryable_error(exc)
                    if retryable and attempt < max_retries:
                        wait_seconds = self._retry_wait_seconds(exc, attempt)
                        print(
                            f"[LLM][WARN] Retryable API error; retry "
                            f"{attempt + 1}/{max_retries} after {wait_seconds}s: {brief}"
                        )
                        time.sleep(wait_seconds)
                        continue

                    self._record_failure()
                    print(f"[LLM][ERROR] API call failed after {attempt + 1} attempt(s): {brief}")
                    if retryable and getattr(default_config, "STOP_ROUTE_BATCH_ON_LLM_FAILURE", True):
                        if service_recoveries < service_recovery_attempts:
                            service_recoveries += 1
                            wait_seconds = max(
                                self._cooldown_remaining(),
                                self._retry_wait_seconds(exc, attempt),
                            )
                            print(
                                "[LLM][WARN] Transient API failure; waiting "
                                f"{wait_seconds}s before recovery attempt "
                                f"{service_recoveries}/{service_recovery_attempts}."
                            )
                            time.sleep(wait_seconds)
                            break
                        raise LLMServiceError(brief) from exc
                    return FAILED_RESPONSE

        return FAILED_RESPONSE

    def _cooldown_remaining(self) -> int:
        now = time.time()
        with self._state_lock:
            cooldown_until = self.cooldown_until
        return max(0, int(cooldown_until - now))

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        if OpenAIProxyClient._is_account_error(exc):
            return False
        status_code = getattr(exc, "status_code", None)
        text = str(exc).lower()
        return (
            status_code in {429, 500, 502, 503, 504}
            or "overloaded" in text
            or "timeout" in text
            or "connection error" in text
            or "api connection" in text
            or "upstream error" in text
            or "do request failed" in text
            or "bad_response_status_code" in text
            or "openai_error" in text
            or "temporarily unavailable" in text
        )

    @staticmethod
    def _is_account_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        text = str(exc).lower()
        account_phrases = (
            "token quota is not enough",
            "pre_consume",
            "insufficient balance",
            "balance",
            "quota",
            "invalid token",
        )
        return (
            status_code == 401
            or (
                status_code in {403, 429}
                and any(phrase in text for phrase in account_phrases)
            )
        )

    @staticmethod
    def _retry_wait_seconds(exc: Exception, attempt: int) -> int:
        text = str(exc).lower()
        if "overloaded" in text or getattr(exc, "status_code", None) == 503:
            base = getattr(default_config, "LLM_OVERLOAD_BACKOFF_SECONDS", 20)
        else:
            base = getattr(default_config, "LLM_RETRY_BACKOFF_SECONDS", 8)
        return int(base * (attempt + 1))

    def _record_failure(self) -> None:
        with self._state_lock:
            self.consecutive_failures += 1
            threshold = getattr(default_config, "LLM_CIRCUIT_BREAKER_FAILURES", 2)
            if self.consecutive_failures < threshold:
                return
            cooldown = getattr(default_config, "LLM_CIRCUIT_BREAKER_COOLDOWN_SECONDS", 120)
            self.cooldown_until = time.time() + cooldown
            self.consecutive_failures = 0
        print(f"[LLM][WARN] Circuit breaker opened for {cooldown}s.")


_GLOBAL_LLM_CLIENT_INSTANCE = None
_GLOBAL_DEEPSEEK_CLIENT_INSTANCE = None


def get_llm_client():
    global _GLOBAL_LLM_CLIENT_INSTANCE

    requested_model = getattr(
        default_config,
        "resolve_model_name",
        lambda model=None: getattr(default_config, "LLM_MODEL_NAME", None),
    )()
    api_key, base_url = _provider_credentials()

    if (
        _GLOBAL_LLM_CLIENT_INSTANCE is not None
        and getattr(_GLOBAL_LLM_CLIENT_INSTANCE, "model_name", None) == requested_model
        and getattr(_GLOBAL_LLM_CLIENT_INSTANCE, "base_url", None) == base_url
    ):
        return _GLOBAL_LLM_CLIENT_INSTANCE

    if default_config.USE_MOCK:
        client = MockLLMClient(api_key="mock", model_name="mock")
    else:
        client = OpenAIProxyClient(
            api_key=api_key,
            model_name=requested_model or default_config.GPT4O_MODEL_NAME,
            base_url=base_url,
        )

    _GLOBAL_LLM_CLIENT_INSTANCE = client
    return client


def get_deepseek_client():
    global _GLOBAL_DEEPSEEK_CLIENT_INSTANCE

    requested_model = getattr(
        default_config,
        "resolve_model_name",
        lambda model=None: getattr(default_config, "CRITIC_LLM_MODEL_NAME", None),
    )(getattr(default_config, "CRITIC_LLM_MODEL_NAME", None))
    api_key, base_url = _provider_credentials(
        getattr(default_config, "CRITIC_LLM_PROVIDER", getattr(default_config, "LLM_PROVIDER", "qian_duo_duo"))
    )

    if (
        _GLOBAL_DEEPSEEK_CLIENT_INSTANCE is not None
        and getattr(_GLOBAL_DEEPSEEK_CLIENT_INSTANCE, "model_name", None) == requested_model
        and getattr(_GLOBAL_DEEPSEEK_CLIENT_INSTANCE, "base_url", None) == base_url
    ):
        return _GLOBAL_DEEPSEEK_CLIENT_INSTANCE

    if default_config.USE_MOCK:
        client = MockLLMClient(api_key="mock", model_name="mock_deepseek")
    else:
        client = OpenAIProxyClient(
            api_key=api_key,
            model_name=requested_model or getattr(default_config, "DEEPSEEK_V4_FLASH_MODEL_NAME", "deepseek-v4-flash"),
            base_url=base_url,
        )

    _GLOBAL_DEEPSEEK_CLIENT_INSTANCE = client
    return client


def reset_llm_clients() -> None:
    """Clear cached clients after changing model/API configuration in notebooks."""
    global _GLOBAL_LLM_CLIENT_INSTANCE, _GLOBAL_DEEPSEEK_CLIENT_INSTANCE
    _GLOBAL_LLM_CLIENT_INSTANCE = None
    _GLOBAL_DEEPSEEK_CLIENT_INSTANCE = None


def _provider_credentials(provider: str | None = None) -> tuple[str, str]:
    provider = (provider or getattr(default_config, "LLM_PROVIDER", "qian_duo_duo")).lower()
    if provider in {"qian_duo_duo", "qianduoduo", "qdd", "openai_proxy"}:
        return default_config.QIAN_DUO_DUO_API_KEY, default_config.QIAN_DUO_DUO_BASE_URL
    raise ValueError(f"Unknown LLM provider: {provider}")
