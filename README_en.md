# 🎵 Symphony Agent Conductor

<div align="center">
  <img src="assets/icons/Symphony Agent Conductor.png" width="200px">
  <p><strong>AI Agent Orchestra for You</strong></p>
</div>

[日本語](README.md)

Welcome to Symphony Agent Conductor!
This is the command center that orchestrates various capable AI agents (Browser automation, IoT, Schedule management, etc.) to support your life and tasks.

Just talk to it in chat, and the agents will work together! 🤖✨

---

## ✨ What can it do?

*   🗣️ **Chat Requests**: Give instructions naturally like "Check tomorrow's weather" or "Turn on the lights".
*   🌐 **Browser Automation**: Browses websites to gather information or perform actions on your behalf.
*   🏠 **Smart Home (IoT)**: Controls home appliances and checks room environments (temperature, etc.).
*   📅 **Schedule Management**: Leave schedule adjustments and confirmations to us.
*   🧠 **Memory**: Remembers conversation contents and your preferences, getting smarter over time.

## 🚀 Get Started (Docker)

If you have Docker, the concert (system) starts with a single command! 🎼

### 1. Preparation 🔑

First, write the API key that serves as the AI's brain into the configuration file.
Copy `secrets.env.example` in the project folder to create a file named `secrets.env`, and fill in your actual API keys.

```bash
cp secrets.env.example secrets.env
```

**secrets.env**
```env
OPENAI_API_KEY=sk-proj-xxxxxxxx... (Your OpenAI API Key)
# Check secrets.env.example for other configurations
```

> 💡 **Note**: `secrets.env` is a secret key, so please do not show it to others or upload it to Git.

### 2. Launch 🐳

Run the following command in your terminal (command prompt).

```bash
docker compose up --build web
```

Various text will flow, like tuning instruments. Please wait a while.

### 3. Showtime! 🎭

When ready, access the following URL in your browser.

👉 **[http://localhost:5050](http://localhost:5050)**

If the screen appears, it's a success! Type "Hello!" in the chat box and enjoy interacting with the agents.

---

## 🛠️ For Developers (Local Execution)

If you want to run it directly on your computer without using Docker, look here.

1.  **Python Preparation**: Python 3.11 or higher is required.
2.  **Installation**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # .venv\Scripts\activate for Windows
    pip install -r requirements.txt
    ```
3.  **Launch**:
    ```bash
    uvicorn app:app --host 0.0.0.0 --port 5050 --reload
    ```

## 📚 Learn More

For detailed agent settings and development behind-the-scenes, please take a look at [AGENTS.md](AGENTS.md). Technical details and customization methods are written there.

---

<div align="center">
  Enjoy your Symphony! 🎶
</div>
