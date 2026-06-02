import os
from google import genai
from google.genai import types

MAX_TURNS = 5

_client = None
_conversation_history: list[dict] = []


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in environment")
        _client = genai.Client(api_key=api_key)
    return _client


SYSTEM_INSTRUCTION = (
    "You are a helpful voice assistant. "
    "Keep responses concise and conversational, suitable for text-to-speech. "
    "Avoid markdown, bullet points, code blocks, or special characters."
)


def generate_response(user_text: str) -> str:
    client = _get_client()

    history = [
        types.Content(
            role=turn["role"],
            parts=[types.Part(text=turn["text"])]
        )
        for turn in _conversation_history
    ]

    history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=300,
        ),
    )

    bot_text = response.text.strip()

    _conversation_history.append({"role": "user", "text": user_text})
    _conversation_history.append({"role": "model", "text": bot_text})

    if len(_conversation_history) > MAX_TURNS * 2:
        del _conversation_history[: len(_conversation_history) - MAX_TURNS * 2]

    return bot_text


def get_history() -> list[dict]:
    return list(_conversation_history)


def clear_history() -> None:
    _conversation_history.clear()
