import os
import requests
from groq import Groq
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes

# ========== CONFIG ==========
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

PROVIDERS = {
    "groq": {
        "api_key": os.environ.get("GROQ_API_KEY"),
        "model": "llama3-70b-8192",
        "enabled": True
    },
    "openrouter": {
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "model": "mistralai/mistral-7b-instruct",
        "enabled": True
    },
    "huggingface": {
        "api_key": os.environ.get("HF_API_KEY"),
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "enabled": True
    }
}

SYSTEM_PROMPT = "You are Openclaw, a helpful AI assistant. Answer clearly and concisely."

# ========== PROVIDER FUNCTIONS ==========

def call_groq(message, api_key, model):
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message.content

def call_openrouter(message, api_key, model):
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message.content

def call_huggingface(message, api_key, model):
    url = f"https://api-inference.huggingface.co/models/{model}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ],
        "max_tokens": 1024
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

CALLER_MAP = {
    "groq": call_groq,
    "openrouter": call_openrouter,
    "huggingface": call_huggingface
}

def get_ai_response(message):
    errors = []
    for provider_name, config in PROVIDERS.items():
        if not config["enabled"] or not config["api_key"]:
            continue
        try:
            print(f"[INFO] Trying: {provider_name}")
            reply = CALLER_MAP[provider_name](message, config["api_key"], config["model"])
            print(f"[INFO] Success: {provider_name}")
            return reply, provider_name
        except Exception as e:
            print(f"[WARN] {provider_name} failed: {str(e)}")
            errors.append(f"{provider_name}: {str(e)}")
    return "❌ Semua provider gagal:\n" + "\n".join(errors), "none"

# ========== TELEGRAM HANDLERS ==========

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Halo! Saya *Openclaw AI Bot*\n\n"
        "Didukung oleh:\n"
        "• Groq (Llama 3)\n"
        "• OpenRouter (Mistral)\n"
        "• HuggingFace (Mistral)\n\n"
        "Langsung ketik pertanyaan kamu!",
        parse_mode="Markdown"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return

    user_message = update.message.text

    thinking_msg = await update.message.reply_text("⏳ Thinking...")

    reply, provider_used = get_ai_response(user_message)

    await thinking_msg.delete()
    await update.message.reply_text(
        f"{reply}\n\n_via {provider_used}_",
        parse_mode="Markdown"
    )

# ========== MAIN ==========

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("[INFO] Bot started polling...")
    app.run_polling()

if __name__ == "__main__":
    main()
