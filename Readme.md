# Telegram Token Bot

Bot Telegram untuk generate OAuth token file (`{email}.json`).
Host guna GitHub Actions — auto-restart via cron setiap 5 jam.
Takde data yang disimpan. Link tunnel berubah setiap restart.

---

## Setup

### 1. GitHub Secrets

Tambah secrets dalam repo **Settings → Secrets and variables → Actions**:

| Secret | Keterangan |
|---|---|
| `BOT_TOKEN` | Telegram Bot token dari @BotFather |
| `OWNER_ID` | Telegram user ID owner (nombor) |
| `CREDENTIALS_JSON` | Isi penuh file `credentials.json` (Google OAuth client) |

### 2. Google OAuth Setup

1. Pergi ke [Google Cloud Console](https://console.cloud.google.com/)
2. Create / pilih project
3. Enable **Gmail API**
4. Buat OAuth 2.0 credentials — pilih **Desktop App**
5. Download `credentials.json`
6. Copy **isi penuh** file tu ke secret `CREDENTIALS_JSON`

### 3. Run

Push ke `main` branch — bot auto start.
Lepas tu cron akan restart setiap 5 jam secara automatik.

Boleh juga manual: **Actions → Run Bot → Run workflow**

---

## Cara Guna

1. **`/start`** di Telegram — keluar link Tunnel & Login
2. Click **Login Google** → authorize account → copy authorization code
3. Buka **Token Page** (link tunnel) → paste code → download `{email}.json`

---

## Nota

- **Push = auto start** — setiap push ke `main`, bot terus jalan
- **Cron setiap 5 jam** — kalau run lama mati, cron akan start balik
- **`concurrency: cancel-in-progress`** — kalau cron trigger semasa run lama masih jalan, run lama auto cancel
- Cloudflare tunnel URL berubah setiap kali restart — ini normal
- Hanya owner (`OWNER_ID`) boleh guna bot
- Takde data disimpan — semua token di-generate on-the-fly dan terus download
