# 🤖 AI Automation Telegram Bot v5.0

Advanced AI-powered Telegram bot with multi-agent system, 5-layer memory, and auto-restart.

## ✨ Features

### 🧠 AI Engine
- **Primary**: `openai/gpt-oss-120b` — 128K context, 64K max output, 500 tps
- **Fallback**: `moonshotai/kimi-k2-instruct-0905` — 256K context, 16K output
- Automatic fallback on rate limits/errors
- Exponential backoff with smart retry

### 🤖 Sub-Agent System
- Main Agent + unlimited Sub-Agents
- Sub-agents run in parallel via `asyncio.gather()`
- Each sub-agent gets focused system prompt
- Results auto-fed back to Main Agent

### 💾 5-Layer Memory System
1. **Conversation Summary** — Auto-summarize when context > 80K tokens
2. **Database Memory** — SQLite for structured facts/preferences
3. **File Memory** — Persistent .md files for knowledge
4. **Sliding Window + Pinned** — Last 30 messages + pinned important ones
5. **RAG** — TF-IDF semantic search over past conversations

### 🎭 Custom Persona
- Edit `persona.txt` to customize bot personality
- Leave empty for default behavior

### ⚡ Code Execution
- Python 3.12 & Bash
- Auto-install dependencies
- Background script support
- Script tracking & management

### 🌐 Web Tools
- Multi-engine search (DuckDuckGo, Brave, Yahoo, etc.)
- 6-layer web scraping fallback
- Deep scan (extract all media, links, downloads)
- Cookie/auth file support

### 🔄 Auto-Restart
- Runs for 5 hours per GitHub Actions run
- Cron schedule auto-starts new run every 5 hours
- Graceful shutdown 35 seconds before expiry
- State preserved via GitHub Actions cache
- **No GH_PAT needed** — fully automated via cron

## 🚀 Setup

### 1. Create Telegram Bot
- Message [@BotFather](https://t.me/BotFather) on Telegram
- Create new bot and get the token

### 2. Get Your Telegram User ID
- Message [@userinfobot](https://t.me/userinfobot) to get your ID

### 3. Get Groq API Key
- Sign up at [console.groq.com](https://console.groq.com)
- Create an API key

### 4. GitHub Setup

**Create a new repository** and push this code.

**Add these secrets** in your repo → Settings → Secrets and variables → Actions:

| Secret | Required | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | ✅ | Bot token from BotFather |
| `GROQ_API_KEY` | ✅ | Groq API key |
| `OWNER_ID` | ✅ | Your Telegram user ID |
| `CHANNEL_ID` | ❌ | Channel ID for membership check (e.g. `-1001234567890`) |
| `CHANNEL_LINK` | ❌ | Channel invite link (e.g. `https://t.me/mychannel`) |

3 secrets wajib, 2 optional untuk channel gating.

### 5. Custom Persona (Optional)
Edit `persona.txt` to give your bot a custom personality. Leave empty for default.

### 6. Deploy
```bash
git add .
git commit -m "Deploy AI Bot v5.0"
git push origin main
```

The bot will start automatically via GitHub Actions and restart every 5 hours via cron!

## 📌 Commands

| Command | Description |
|---|---|
| `/start` | Show status, countdown, and capabilities |
| `/stop` | Stop AI processing |
| `/new` | Start new session (clears conversation, keeps memory) |
| `/scripts` | List all tracked scripts |
| `/memory` | Show memory status and entries |

## 💡 Tips

- **Pin messages**: Reply to any message with "pin" to save it permanently
- **Sub-agents**: The AI automatically delegates parallel tasks when needed
- **Memory**: AI automatically saves important facts and decisions
- **Background scripts**: Long-running scripts continue in background
- **Cookie support**: Upload Netscape cookie files for authenticated scraping
- **Stop AI**: Only `/stop` command and New Session button can stop AI

## 📝 License

MIT — Do whatever you want.
