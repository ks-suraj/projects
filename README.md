## 🚀 Run This Project In Your Own GitHub Account

Anyone can easily run this project by **forking the repository** and using their own OpenRouter API key.

### ✅ Steps:
1. **Fork This Repository**  
Click the "Fork" button (top-right) to copy this repository to your GitHub account.

2. **Add Your OpenRouter API Key**  
In your forked repo:
- Go to **Settings → Secrets and variables → Actions → New repository secret**.
- Name: `OPENROUTER_API_KEY`
- Value: *(your actual API key from [openrouter.ai](https://openrouter.ai))*

3. **Run the Workflow**  
- Go to the **Actions** tab in your fork.
- Select the workflow called **"Run OpenRouter AI Agent"**.
- Click **"Run workflow"** → The AI Agent will run using your API key.

### ✅ Notes:
- Your API key stays private inside your GitHub Secrets.
- You can modify the prompts inside `.github/workflows/openrouter-ai-agent.yml` or `agent.py` if you wish to customize queries.
- This project uses OpenRouter models: [https://openrouter.ai](https://openrouter.ai)

---

### 🎯 Why Fork & Run?
GitHub Actions cannot be run by visitors directly on someone else's repo (for security reasons).  
Forking lets you run this project fully under your own GitHub account, safely.

---
