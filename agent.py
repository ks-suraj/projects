import requests
import json
import time

API_KEY = "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"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/ks-suraj/projects/new/openrouter-ai-agent",
    "X-Title": "OpenRouter AI Agent on GitHub",
}

def ask_ai(prompt):
    data = {
        "model": "deepseek/deepseek-chat-v3-0324:free",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
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
        raise ValueError(f"Failed to decode JSON response: {e}\nRaw Response:\n{response.text}")

    # Print entire response for debugging
    print("API Raw Response:", json.dumps(result, indent=2))

    # Check for 'choices' key safely
    if 'choices' not in result:
        raise ValueError("API Error: " + result.get('error', {}).get('message', 'Unknown error'))

    return result['choices'][0]['message']['content']

def agentic_query(prompt, max_retries=2):
    for attempt in range(max_retries + 1):
        print(f"\n--- Attempt {attempt + 1} ---")
        try:
            answer = ask_ai(prompt)
        except Exception as e:
            print(f"API Call Failed: {e}")
            break

        print("AI Answer:", answer)

        if "I don't know" in answer or "uncertain" in answer or "unsure" in answer:
            print("Agent detected uncertainty. Retrying...")
            prompt = f"{prompt} Please clarify and explain thoroughly."
            time.sleep(2)
        else:
            print("Agent is satisfied with the answer.")
            break

# Example task
agentic_query("What is the meaning of life?")
