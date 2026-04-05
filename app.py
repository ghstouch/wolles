import os
import requests
from groq import Groq
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
import google.generativeai as genai

# ========== CONFIG ==========
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

SYSTEM_PROMPT = "You are Openclaw, a helpful AI assistant. Answer clearly and concisely."

# ========== PROVIDER FUNCTIONS ==========

def call_openrouter(message):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    response = client.chat.completions.create(
        model="qwen/qwen3.6-plus:free",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message.content

def call_groq(message):
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message.content

def call_google(message):
    api_key = os.environ.get("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=SYSTEM_PROMPT
    )
    response = model.generate_content(message)
    return response.text

def call_huggingface(message):
    api_key = os.environ.get("HF_API_KEY")
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.3",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ],
        "max_tokens": 1024
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# Urutan provider sesuai request
PROVIDERS = [
    ("openrouter", call_openrouter),
    ("groq", call_groq),
    ("google", call_google),
    ("huggingface", call_huggingface),
]

def get_ai_response(message):
    errors = []
    for provider_name, caller in PROVIDERS:
        try:
            print(f"[INFO] Trying: {provider_name}")
            reply = caller(message)
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
        "• OpenRouter (Qwen 3.6)\n"
        "• Groq (Llama 3)\n"
        "• Google (Gemini 2.0 Flash)\n"
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
