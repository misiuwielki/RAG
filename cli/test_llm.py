import os
from dotenv import load_dotenv
from openai import OpenAI

def main():
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

    client  = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    chat_completion = client.chat.completions.create(
        model="openrouter/free",
        messages= [
            {
            "role": "user",
            "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
            }
        ]
    )
    print(f"{chat_completion.choices[0].message.content.strip() if chat_completion.choices[0].message.content else "No reply"}")
    print(f"Prompt tokens: {chat_completion.usage.prompt_tokens if chat_completion.usage else 0}")
    print(f"Response tokens: {chat_completion.usage.completion_tokens if chat_completion.usage else 0}")

main()
