import requests
import json
import os

def ask_ai(api_key, prompt):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ks-suraj/projects",
        "X-Title": "OpenRouter AI Agent on GitHub",
    }

    data = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(data)
    )

    try:
        result = response.json()
    except Exception as e:
        raise ValueError(f"Invalid JSON Response: {e}\nResponse: {response.text}")

    if 'choices' not in result:
        raise ValueError("API Error: " + result.get('error', {}).get('message', 'Unknown error'))

    return result['choices'][0]['message']['content']

def main():
    api_key = os.getenv("API_KEY")
    prompt = os.getenv("PROMPT")

    if not api_key or not prompt:
        raise EnvironmentError("Missing API_KEY or PROMPT environment variables.")

    print("Running AI Agent...")
    answer = ask_ai(api_key, prompt)
    print("\n=== Agent Response ===")
    print(answer)

if __name__ == "__main__":
    main()
