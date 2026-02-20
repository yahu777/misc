#!/usr/bin/env python3
"""Minimal local AI Dungeon Master agent using LM Studio's OpenAI-compatible API.

This script intentionally avoids agent frameworks and keeps all orchestration explicit
so it is easy to understand, debug, and extend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
from urllib import error, request

# LM Studio local OpenAI-compatible endpoint.
API_URL = "http://localhost:1234/v1/chat/completions"
# Change MODEL_NAME to any installed Qwen model name in LM Studio.
MODEL_NAME = "qwen2.5-7b-instruct"

# File locations (stored next to this script).
BASE_DIR = Path(__file__).resolve().parent
PROMPT_FILE = BASE_DIR / "agent_prompt.txt"
CANON_FILE = BASE_DIR / "canon.txt"


def read_text_file(path: Path, fallback: str = "") -> str:
    """Read UTF-8 text content from a file.

    Returns `fallback` if the file does not exist, so first-run experience
    remains stable.
    """
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return fallback


def ensure_files_exist() -> None:
    """Create canon/prompt placeholders if missing.

    This keeps the script production-stable for first launch instead of failing
    with file-not-found errors.
    """
    if not PROMPT_FILE.exists():
        PROMPT_FILE.write_text(
            "You are the Dungeon Master. Describe the world and NPCs only.",
            encoding="utf-8",
        )
    if not CANON_FILE.exists():
        CANON_FILE.write_text(
            "# Campaign Canon\n\n- Add persistent facts here.\n", encoding="utf-8"
        )


def load_runtime_context() -> Dict[str, str]:
    """Load system prompt and persistent canon from disk."""
    system_prompt = read_text_file(PROMPT_FILE)
    canon_text = read_text_file(CANON_FILE)
    return {"system_prompt": system_prompt, "canon": canon_text}


def build_messages(system_prompt: str, canon_text: str, player_input: str) -> List[Dict[str, str]]:
    """Build message list for the chat completion API.

    Canon is injected each turn as a dedicated system message, ensuring the model
    can consistently reference persistent campaign facts.
    """
    canon_message = (
        "Campaign canon (persistent world facts; obey this unless explicitly updated):\n"
        f"{canon_text or '[No canon recorded yet.]'}"
    )

    # Explicitly reinforce that the DM must never decide player actions.
    behavior_guardrail = (
        "Important DM rule: Never control, decide, or narrate the player's character actions. "
        "Stop after describing scene/world/NPC outcomes and wait for player input."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": canon_message},
        {"role": "system", "content": behavior_guardrail},
        {"role": "user", "content": player_input},
    ]


def call_lm_studio(messages: List[Dict[str, str]], timeout_seconds: int = 90) -> str:
    """Send a chat completion request to LM Studio and return assistant text."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.8,
        "max_tokens": 600,
    }

    data = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            result = json.loads(raw)
            return result["choices"][0]["message"]["content"].strip()
    except error.URLError as exc:
        raise RuntimeError(
            "Failed to reach LM Studio API. Ensure LM Studio is running, local server is enabled, "
            f"and model '{MODEL_NAME}' is loaded. Details: {exc}"
        ) from exc
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Received an unexpected response format from LM Studio API."
        ) from exc


def append_canon_update(new_fact: str) -> None:
    """Append a new canonical fact to canon.txt.

    Simple persistence hook for manual updates during play.
    """
    if not new_fact.strip():
        return
    with CANON_FILE.open("a", encoding="utf-8") as canon:
        canon.write(f"- {new_fact.strip()}\n")


def retrieve_relevant_lore_stub(query: str) -> str:
    """Future RAG hook.

    In a future version, this function can retrieve relevant lore snippets from
    embeddings/vector storage. For now, it returns an empty string to keep the
    runtime simple and dependency-free.
    """
    _ = query
    return ""


def discord_message_hook_stub(channel_id: str, message: str) -> None:
    """Future Discord integration hook.

    Keep this stub so connecting a Discord bot later is straightforward without
    restructuring the core game loop.
    """
    _ = (channel_id, message)


def game_loop() -> None:
    """Run an interactive scene-by-scene dungeon master loop.

    The loop stops after every model response and waits for player input,
    satisfying turn-based play requirements.
    """
    ensure_files_exist()

    print("Local AI Dungeon Master is ready.")
    print("Commands: /quit, /reload, /canon <fact>")

    runtime = load_runtime_context()

    while True:
        player_input = input("\nPlayer > ").strip()

        if not player_input:
            continue

        if player_input.lower() == "/quit":
            print("Session ended.")
            break

        if player_input.lower() == "/reload":
            runtime = load_runtime_context()
            print("Reloaded agent_prompt.txt and canon.txt")
            continue

        if player_input.lower().startswith("/canon "):
            append_canon_update(player_input[7:])
            runtime = load_runtime_context()
            print("Canon updated.")
            continue

        # Optional stub call for future retrieval augmentation.
        _lore_snippet = retrieve_relevant_lore_stub(player_input)

        messages = build_messages(
            system_prompt=runtime["system_prompt"],
            canon_text=runtime["canon"],
            player_input=player_input,
        )

        try:
            dm_reply = call_lm_studio(messages)
        except RuntimeError as exc:
            print(f"\n[Error] {exc}")
            continue

        print("\nDM >")
        print(dm_reply)

        # Future Discord output hook.
        discord_message_hook_stub(channel_id="", message=dm_reply)


if __name__ == "__main__":
    game_loop()
