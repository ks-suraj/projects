import requests
import json
import time

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
        raise ValueError(f"Invalid JSON: {e}\n{response.text}")

    if 'choices' not in result:
        raise ValueError("API Error: " + result.get('error', {}).get('message', 'Unknown error'))

    return result['choices'][0]['message']['content']

def agentic_query(api_key, prompt, max_retries=2):
    for attempt in range(max_retries + 1):
        print(f"\n--- Attempt {attempt + 1} ---")
        try:
            answer = ask_ai(api_key, prompt)
        except Exception as e:
            print(f"API Call Failed: {e}")
            break

        print("AI Answer:", answer)

        if "I don't know" in answer or "uncertain" in answer or "unsure" in answer:
            print("Agent detected uncertainty. Retrying...")
            prompt += " Please clarify and explain thoroughly."
            time.sleep(2)
        else:
            print("Agent is satisfied with the answer.")
            break

if __name__ == "__main__":
    api_key = input("Enter your OpenRouter API Key: ").strip()
    prompt = input("Enter your prompt: ").strip()
    agentic_query(api_key, prompt)
