import os

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
MODEL = os.getenv("LLM_MODEL", "")


def ask_llm(system: str, user: str) -> str:
    """Send a prompt to the configured provider and return the text reply."""
    if not MODEL:
        raise ValueError("Set LLM_MODEL in your .env file.")

    if PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        resp = client.messages.create(
            model=MODEL,
            max_tokens=400,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text

    if PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI()  # reads OPENAI_API_KEY
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content

    raise ValueError(f"Unknown provider: {PROVIDER}")