"""Single definition of the OpenAI-vs-self-hosted backend rule.

Every generation stage talks to a chat LLM through the OpenAI Python client,
and the backend is picked by the *model name*: proprietary OpenAI models read
``OPENAI_API_KEY`` and hit ``api.openai.com``; anything else (``Qwen/Qwen3-32B``
and friends) is posted to the OpenAI-compatible ``--base_url`` you serve
yourself.

The rule used to be spelled three different ways -- a hardcoded set in
``src/speechify_run.py`` and ``src/synthesis/run.py``, a name-prefix test in
``src/synthesis/run_add_bc.py``, and nothing at all in
``src/inference_turntaking_llm.py``, which sent ``gpt-4.1`` to localhost. Import
from here instead of re-deriving it.
"""

import os
from typing import Optional

from openai import OpenAI

# Known-good proprietary model names, kept for documentation value. The prefix
# test below is what actually decides, so a newer OpenAI model absent from this
# set still routes correctly rather than silently going to localhost.
GPT_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5.1",
    "gpt-5.2",
}

_OPENAI_PREFIXES = ("gpt-", "chatgpt-", "o1", "o3", "o4")


def is_openai_model(model_name: str) -> bool:
    """True when `model_name` should be served by api.openai.com."""
    if model_name in GPT_MODELS:
        return True
    return model_name.startswith(_OPENAI_PREFIXES)


def make_client(
    model_name: str,
    api_key: str = "EMPTY",
    base_url: Optional[str] = "http://localhost:8000/v1",
) -> OpenAI:
    """Build the client for `model_name`.

    OpenAI models ignore `api_key` / `base_url` and read `OPENAI_API_KEY`;
    everything else posts to `base_url` with `api_key`.
    """
    if is_openai_model(model_name):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                f"{model_name!r} is an OpenAI model but OPENAI_API_KEY is unset. "
                "Export it, or pass a self-hosted model name with --base_url."
            )
        return OpenAI(api_key=key)
    return OpenAI(api_key=api_key, base_url=base_url)


def no_thinking_extra_body(model_name: str) -> Optional[dict]:
    """`extra_body` that turns a reasoning model's thinking block off, or None.

    `chat_template_kwargs` is a vLLM extension: api.openai.com rejects the
    unknown field outright with

        400 Unrecognized request argument supplied: chat_template_kwargs

    so it must only ever be sent to a self-hosted server. Passing the result
    through as `extra_body=` is safe either way -- the OpenAI client omits the
    field when it is None.
    """
    if is_openai_model(model_name):
        return None
    return {"chat_template_kwargs": {"enable_thinking": False}}
