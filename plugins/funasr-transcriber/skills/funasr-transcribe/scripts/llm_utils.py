"""Shared LLM call infrastructure for transcription scripts.

Supports three providers with auto-detection:
  - AWS Bedrock (ARN or cross-region ID like us.anthropic.claude-*)
  - Anthropic Messages API (bare claude-* model IDs)
  - OpenAI-compatible API (gpt-*, deepseek-*, vLLM, Ollama, etc.)
"""

import re
import time
from typing import Optional


def detect_llm_provider(model_id: str) -> str:
    """Detect LLM provider from model ID string.

    Returns one of: 'bedrock', 'anthropic', 'openai'.
    """
    if model_id.startswith("arn:aws:bedrock:") or re.match(r"^[a-z]{2}\.", model_id):
        return "bedrock"
    if "claude" in model_id and not model_id.startswith("arn:"):
        return "anthropic"
    return "openai"


def _call_bedrock(system_prompt: str, user_message: str,
                  model_id: str, region: str, max_tokens: int = 8192) -> str:
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise RuntimeError(
            "The 'boto3' package is required for Bedrock API calls. "
            "Install it with: pip install boto3")

    client = boto3.client(
        "bedrock-runtime", region_name=region,
        config=Config(read_timeout=300, connect_timeout=10, retries={"max_attempts": 3}),
    )
    response = client.converse(
        modelId=model_id,
        system=[{"text": system_prompt}],
        messages=[{"role": "user", "content": [{"text": user_message}]}],
        inferenceConfig={"maxTokens": max_tokens},
    )
    try:
        return response["output"]["message"]["content"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(
            f"Unexpected Bedrock response structure: {e}. "
            f"Response keys: {list(response.keys())}"
        ) from e


def _call_anthropic(system_prompt: str, user_message: str,
                    model_id: str, max_tokens: int = 8192) -> str:
    try:
        from anthropic import Anthropic
    except ImportError:
        raise RuntimeError(
            "The 'anthropic' package is required for Anthropic API calls. "
            "Install it with: pip install anthropic")

    client = Anthropic()
    response = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    if not response.content:
        raise RuntimeError(
            f"Anthropic API returned empty content (stop_reason={response.stop_reason})")
    return response.content[0].text


def _call_openai(system_prompt: str, user_message: str,
                 model_id: str, max_tokens: int = 8192) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "The 'openai' package is required for OpenAI-compatible API calls. "
            "Install it with: pip install openai")

    client = OpenAI()
    response = client.chat.completions.create(
        model=model_id,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    if not response.choices or response.choices[0].message.content is None:
        raise RuntimeError(f"OpenAI API returned empty content for model {model_id}")
    return response.choices[0].message.content


def is_retryable(e: Exception) -> bool:
    """Check if an LLM exception is retryable (rate limit / throttling)."""
    msg = str(e).lower()
    return any(token in msg for token in ("throttl", "rate_limit", "ratelimit", "429", "529"))


def call_llm(system_prompt: str, user_message: str,
             model_id: str, region: Optional[str] = None,
             max_retries: int = 3) -> str:
    """Call LLM with auto-detected provider and retry logic."""
    provider = detect_llm_provider(model_id)
    for attempt in range(max_retries):
        try:
            if provider == "bedrock":
                return _call_bedrock(system_prompt, user_message,
                                     model_id, region or "us-west-2")
            elif provider == "anthropic":
                return _call_anthropic(system_prompt, user_message, model_id)
            else:
                return _call_openai(system_prompt, user_message, model_id)
        except Exception as e:
            if attempt < max_retries - 1 and is_retryable(e):
                wait = 5 * (attempt + 1)
                print(f"  LLM call attempt {attempt+1} failed ({type(e).__name__}), "
                      f"retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("call_llm: all retries exhausted")
