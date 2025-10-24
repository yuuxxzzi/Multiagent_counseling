import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from openai import OpenAI


def main() -> int:
    model_id = os.getenv(
        "ROLEPLAY_MODEL",
        "ft:gpt-4o-mini-2024-07-18:personal:aihub-chat-v1:CTq3k8vi",
    )

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY not set", file=sys.stderr)
        return 2

    org = os.getenv("OPENAI_ORG_ID")
    project = os.getenv("OPENAI_PROJECT")

    kwargs = {}
    if org:
        kwargs["organization"] = org
    if project:
        kwargs["project"] = project

    client = OpenAI(api_key=key, **kwargs)

    print(f"Testing retrieve for model: {model_id}")
    try:
        m = client.models.retrieve(model_id)
        print("OK id=", m.id)
        return 0
    except Exception as e:
        print("ERROR:", repr(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())


