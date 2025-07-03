import gradio as gr
import requests
import json
import os

API_KEY = os.getenv("OPENROUTER_API_KEY")  # Hugging Face Space secret

def ask_ai(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ks-suraj/projects",
        "X-Title": "OpenRouter AI Agent Gradio App",
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
    result = response.json()
    if 'choices' not in result:
        return f"API Error: {result.get('error', {}).get('message', 'Unknown error')}"
    return result['choices'][0]['message']['content']

iface = gr.Interface(
    fn=ask_ai,
    inputs="text",
    outputs="text",
    title="OpenRouter AI Agent",
    description="Enter your prompt to chat with the AI.",
)

if __name__ == "__main__":
    iface.launch()
