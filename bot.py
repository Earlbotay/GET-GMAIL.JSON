#!/usr/bin/env python3
"""
Telegram Token Bot v1.1
========================
python-telegram-bot==13.7 | Python 3.10+

Commands (owner only):
  /start — Show info, tunnel link, login link

Hosted via GitHub Actions (cron every 5h + push trigger).
No data stored. Tunnel URL changes each restart.
"""

import json
import os
import re
import io
import subprocess
import threading
import logging
import time
from pathlib import Path

import requests as http_req
from flask import (
    Flask,
    request as flask_request,
    jsonify,
    send_file,
    render_template_string,
)
from telegram import ParseMode, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
OWNER_ID = int(os.environ.get("OWNER_ID", "0"))

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
]

BASE_DIR = Path(__file__).parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"

WEB_PORT = 8080

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---- runtime state (no persistence) ----
tunnel_url = None

# ============================================================
# HELPERS
# ============================================================

def bq(title, body):
    """Single full blockquote — open at top, close at bottom."""
    return f"<blockquote>{title}\n\n{body}</blockquote>"


def check_owner(update):
    if update.effective_user.id == OWNER_ID:
        return True
    name = (
        update.effective_user.full_name
        or update.effective_user.first_name
        or "Unknown"
    )
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(
            "\U0001f4ac Chat Owner",
            url=f"tg://user?id={OWNER_ID}",
        )]
    ])
    update.message.reply_text(
        bq(
            "\u26d4 Access Denied",
            f"You are not authorized.\n\n"
            f"Order report \u27a1 Chat Owner\n\n"
            f"\U0001f464 {name}",
        ),
        parse_mode=ParseMode.HTML,
        reply_markup=kb,
    )
    return False


def generate_auth_url():
    """Generate OAuth URL — OOB flow, no localhost."""
    flow = Flow.from_client_secrets_file(
        str(CREDENTIALS_FILE),
        scopes=GMAIL_SCOPES,
        redirect_uri="urn:ietf:wg:oauth:2.0:oob",
    )
    url, _ = flow.authorization_url(access_type="offline", prompt="consent")
    return url


# ============================================================
# FLASK WEB SERVER
# ============================================================

webapp = Flask(__name__)

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Token Generator</title>
    <style>
        *{margin:0;padding:0;box-sizing:border-box}
        body{
            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
            background:#0a0a0f;color:#e0e0e0;
            min-height:100vh;display:flex;align-items:center;justify-content:center;
        }
        .c{
            background:#12121a;border:1px solid #1e1e2e;border-radius:16px;
            padding:40px;max-width:460px;width:90%;
            box-shadow:0 16px 48px rgba(0,0,0,.5);
        }
        .ic{font-size:40px;margin-bottom:16px}
        h1{color:#00d4ff;font-size:22px;margin-bottom:6px}
        .sb{color:#666;font-size:13px;margin-bottom:28px}
        label{display:block;margin-bottom:8px;font-weight:600;font-size:13px;
               color:#aaa;text-transform:uppercase;letter-spacing:.5px}
        input[type=text]{
            width:100%;padding:14px 16px;border:2px solid #1e1e2e;border-radius:10px;
            background:#0a0a0f;color:#e0e0e0;font-size:15px;outline:none;
            transition:border-color .3s;font-family:'Courier New',monospace;
        }
        input[type=text]:focus{border-color:#00d4ff}
        input::placeholder{color:#333}
        button{
            width:100%;padding:14px;border:none;border-radius:10px;
            background:linear-gradient(135deg,#00d4ff,#0099cc);
            color:#000;font-size:15px;font-weight:700;cursor:pointer;
            margin-top:18px;transition:opacity .3s;
        }
        button:hover{opacity:.9}
        button:disabled{opacity:.4;cursor:not-allowed}
        .st{margin-top:16px;padding:12px 16px;border-radius:10px;font-size:13px;display:none}
        .er{background:#1a0f0f;color:#ff6b6b;border:1px solid #331a1a}
        .ok{background:#0f1a0f;color:#6bff6b;border:1px solid #1a331a}
        .sp{
            display:none;margin:18px auto 0;width:28px;height:28px;
            border:3px solid #1e1e2e;border-top:3px solid #00d4ff;
            border-radius:50%;animation:spin .8s linear infinite;
        }
        @keyframes spin{to{transform:rotate(360deg)}}
    </style>
</head>
<body>
<div class="c">
    <div class="ic">&#x1f510;</div>
    <h1>Token Generator</h1>
    <p class="sb">Paste your Google OAuth authorization code to download token file.</p>
    <form id="f">
        <label>Authorization Code</label>
        <input type="text" id="code" placeholder="4/0Axxxxxx..." required autocomplete="off">
        <button type="submit" id="btn">&#x2B07; Download Token</button>
    </form>
    <div class="sp" id="sp"></div>
    <div class="st" id="st"></div>
</div>
<script>
document.getElementById('f').onsubmit=async e=>{
    e.preventDefault();
    const code=document.getElementById('code').value.trim();
    if(!code)return;
    const btn=document.getElementById('btn'),
          sp=document.getElementById('sp'),
          st=document.getElementById('st');
    btn.disabled=true;sp.style.display='block';st.style.display='none';
    try{
        const r=await fetch('/process',{
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({code})
        });
        if(r.ok){
            const d=r.headers.get('Content-Disposition'),
                  blob=await r.blob(),
                  fn=d?d.split('filename=')[1].replace(/"/g,''):'token.json',
                  a=document.createElement('a');
            a.href=URL.createObjectURL(blob);a.download=fn;a.click();
            URL.revokeObjectURL(a.href);
            st.className='st ok';st.textContent='\u2705 Downloaded: '+fn;
        }else{
            const err=await r.json();
            st.className='st er';st.textContent='\u274c '+(err.error||'Failed');
        }
    }catch(err){
        st.className='st er';st.textContent='\u274c '+err.message;
    }
    st.style.display='block';sp.style.display='none';btn.disabled=false;
};
</script>
</body>
</html>
"""


@webapp.route("/")
def index():
    return render_template_string(HTML_PAGE)


@webapp.route("/process", methods=["POST"])
def process_token():
    data = flask_request.get_json()
    code = (data or {}).get("code", "").strip()

    if not code:
        return jsonify({"error": "No authorization code provided"}), 400

    try:
        flow = Flow.from_client_secrets_file(
            str(CREDENTIALS_FILE),
            scopes=GMAIL_SCOPES,
            redirect_uri="urn:ietf:wg:oauth:2.0:oob",
        )
        flow.fetch_token(code=code)
        creds = flow.credentials

        # Detect email
        email = None
        try:
            r = http_req.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {creds.token}"},
                timeout=10,
            )
            if r.ok:
                email = r.json().get("email")
        except Exception:
            pass

        if not email:
            email = "unknown"

        # Build JSON in memory — nothing saved to disk
        token_json = creds.to_json()
        buf = io.BytesIO(token_json.encode("utf-8"))
        buf.seek(0)

        return send_file(
            buf,
            as_attachment=True,
            download_name=f"{email}.json",
            mimetype="application/json",
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ============================================================
# CLOUDFLARE TUNNEL
# ============================================================

def start_tunnel():
    global tunnel_url
    try:
        proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://localhost:{WEB_PORT}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        for line in iter(proc.stdout.readline, b""):
            text = line.decode("utf-8", errors="replace")
            log.info("cloudflared: %s", text.strip())
            m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", text)
            if m:
                tunnel_url = m.group(0)
                log.info("Tunnel ready: %s", tunnel_url)
                break

        # Keep draining stdout so buffer doesn't fill
        for _ in iter(proc.stdout.readline, b""):
            pass

    except FileNotFoundError:
        log.error("cloudflared not found — tunnel disabled")
    except Exception as e:
        log.error("Tunnel error: %s", e)


# ============================================================
# COMMAND HANDLERS
# ============================================================

def cmd_start(update, ctx):
    """/start — Show info, tunnel link, login link."""
    if not check_owner(update):
        return

    # OAuth login link
    try:
        auth_url = generate_auth_url()
        login_line = f'\U0001f510 <a href="{auth_url}">Login Google</a>'
    except Exception:
        login_line = "\u274c credentials.json missing"

    # Tunnel link
    if tunnel_url:
        tunnel_line = f'\U0001f310 <a href="{tunnel_url}">Open Token Page</a>'
    else:
        tunnel_line = "\U0001f310 \u23f3 Tunnel starting..."

    update.message.reply_text(
        bq(
            "\U0001f916 Token Bot",
            f"{tunnel_line}\n"
            f"{login_line}\n\n"
            f"\U0001f4cb Steps:\n"
            f"1. Click Login Google\n"
            f"2. Authorize account\n"
            f"3. Copy authorization code\n"
            f"4. Open Token Page\n"
            f"5. Paste code \u2192 Download token",
        ),
        parse_mode=ParseMode.HTML,
        disable_web_page_preview=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    # 1. Start Flask web server
    threading.Thread(
        target=lambda: webapp.run(
            host="0.0.0.0", port=WEB_PORT, debug=False, use_reloader=False
        ),
        daemon=True,
    ).start()
    log.info("Web server started on port %d", WEB_PORT)

    # 2. Start Cloudflare tunnel
    threading.Thread(target=start_tunnel, daemon=True).start()

    # 3. Wait for tunnel to establish
    for i in range(30):
        if tunnel_url:
            break
        time.sleep(1)
    if tunnel_url:
        log.info("Tunnel URL: %s", tunnel_url)
    else:
        log.warning("Tunnel not ready after 30s — bot starts anyway")

    # 4. Start Telegram bot
    updater = Updater(token=BOT_TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", cmd_start))

    log.info("Bot started!")
    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()
