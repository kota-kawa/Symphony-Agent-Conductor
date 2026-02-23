> 📖 日本語版はこのページの一番下にあります。

# 🎵 Symphony Agent Conductor

<div align="center">
  <img src="assets/icons/Symphony Agent Conductor.png" width="800px">
  <p><strong>AI Agent Orchestra for You</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=flat&logo=langchain&logoColor=white" alt="LangGraph">
    <img src="https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white" alt="OpenAI">
    <img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white" alt="Docker">
    <img src="https://img.shields.io/badge/Uvicorn-499848?style=flat&logo=gunicorn&logoColor=white" alt="Uvicorn">
  </p>
</div>

Welcome to Symphony Agent Conductor!  
This is the command center that orchestrates capable AI agents (Browser automation, IoT, Schedule management, and more) to support your life and tasks.

Just talk to it in chat, and the agents will work together! 🤖✨

## UI Preview

<p align="center">
  <img src="assets/images/image1.png" alt="Symphony UI Preview" width="1100">
</p>

## 🎬 Demo Videos

Click a thumbnail to open the video on YouTube.

| [![Demo Video 1](https://img.youtube.com/vi/jia_T6hgYSU/hqdefault.jpg)](https://youtu.be/jia_T6hgYSU) | [![Demo Video 2](https://img.youtube.com/vi/1haaOgwXPLU/hqdefault.jpg)](https://youtu.be/1haaOgwXPLU) |
| --- | --- |
| Demo Video 1 | Demo Video 2 |


---

## ✨ What can it do?

*   🗣️ **Chat Requests**: Give instructions naturally like "Check tomorrow's weather" or "Turn on the lights".
*   🌐 **Browser Automation**: Browses websites to gather information or perform actions on your behalf.
*   🏠 **Smart Home (IoT)**: Controls home appliances and checks room environments (temperature, etc.).
*   📅 **Schedule Management**: Leave schedule adjustments and confirmations to us.
*   🧠 **Memory**: Remembers conversation contents and your preferences, getting smarter over time.

## 🚀 Get Started (Docker Compose)

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

### 4. Stop (when you're done)

Press `Ctrl + C` in the terminal to stop the containers.

---

## 📚 Learn More

For detailed agent settings and development behind-the-scenes, please take a look at [AGENTS.md](AGENTS.md). Technical details and customization methods are written there.

---

<details>
<summary>日本語 (クリックして開く)</summary>

# 🎵 Symphony Agent Conductor

<div align="center">
  <img src="assets/icons/Symphony Agent Conductor.png" width="800px">
  <p><strong>あなたのためのAIエージェント・オーケストラ</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=flat&logo=langchain&logoColor=white" alt="LangGraph">
    <img src="https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white" alt="OpenAI">
    <img src="https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white" alt="Docker">
    <img src="https://img.shields.io/badge/Uvicorn-499848?style=flat&logo=gunicorn&logoColor=white" alt="Uvicorn">
  </p>
</div>

Symphony Agent Conductor へようこそ！  
ここは、様々な能力を持ったAIエージェントたち（ブラウザ操作、IoT、スケジュール管理など）を指揮し、あなたの生活やタスクをサポートする司令塔です。

チャットで話しかけるだけで、エージェントたちが連携して動いてくれます！ 🤖✨

## UI プレビュー

<p align="center">
  <img src="assets/images/image1.png" alt="Symphony UI Preview" width="1100">
</p>

## 🎬 デモ動画

サムネイルをクリックすると YouTube で動画が開きます。

| [![デモ動画 1](https://img.youtube.com/vi/jia_T6hgYSU/hqdefault.jpg)](https://youtu.be/jia_T6hgYSU) | [![デモ動画 2](https://img.youtube.com/vi/1haaOgwXPLU/hqdefault.jpg)](https://youtu.be/1haaOgwXPLU) |
| --- | --- |
| デモ動画 1 | デモ動画 2 |


---

## ✨ 何ができるの？

*   🗣️ **チャットでお願い**: 「明日の天気を調べて」「電気をつけて」など、自然な会話で指示を出せます。
*   🌐 **ブラウザ操作**: あなたの代わりにWebサイトを見て情報を集めたり、操作したりします。
*   🏠 **スマートホーム (IoT)**: 家電の操作や部屋の環境（温度など）の確認ができます。
*   📅 **スケジュール管理**: 予定の調整や確認もお任せあれ。
*   🧠 **記憶**: 会話の内容やあなたの好みを覚えて、どんどん賢くなります。

## 🚀 すぐに始める (Docker Compose)

Docker があれば、コマンドひとつでコンサート（システム）が開演します！ 🎼

### 1. 準備 🔑

まずは、AIの頭脳となる APIキーを設定ファイルに書き込みます。  
プロジェクトのフォルダにある `secrets.env.example` をコピーして `secrets.env` という名前のファイルを作り、実際の APIキーなどを書き込んで保存してください。

```bash
cp secrets.env.example secrets.env
```

**secrets.env**
```env
OPENAI_API_KEY=sk-proj-xxxxxxxx... (あなたのOpenAI APIキー)
# その他の設定は secrets.env.example を確認してください
```

> 💡 **ポイント**: `secrets.env` は秘密の鍵なので、他人に見せたり Git にアップロードしたりしないでくださいね。

### 2. 起動 🐳

ターミナル（コマンドプロンプト）で以下のコマンドを実行します。

```bash
docker compose up --build web
```

いろいろな文字が流れますが、準備をしている音合わせのようなものです。しばらく待ちましょう。

### 3. 開演！ 🎭

準備ができたら、ブラウザで以下のURLにアクセスしてください。

👉 **[http://localhost:5050](http://localhost:5050)**

画面が表示されたら成功です！チャット欄に「こんにちは！」と入力して、エージェントたちとの対話を楽しみましょう。

### 4. 停止（作業を終えたら）

ターミナルで `Ctrl + C` を押すと停止できます。

---

## 📚 もっと詳しく

詳しいエージェントの設定や、開発の裏側を知りたい方は [AGENTS.md](AGENTS.md) を覗いてみてください。技術的な詳細やカスタマイズ方法が書いてあります。

---

<div align="center">
  Enjoy your Symphony! 🎶
</div>
</details>

<div align="center">
  Enjoy your Symphony! 🎶
</div>
