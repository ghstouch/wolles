import os
import time
import requests
import threading
from groq import Groq
from openai import OpenAI
from telegram import Update, BotCommand
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
import google.generativeai as genai

# ========== SYSTEM PROMPT (SOUL + IDENTITY) ==========
SYSTEM_PROMPT = """
# 🧠 IDENTITY: THE BESTIE
Lu adalah Research Analyst kelas kakap sekaligus Trader berpengalaman yang punya insting tajam di Crypto, Saham, dan Forex. Jangan cuma ngeringkas, tapi bedah sampe ke akar-akarnya.

## Role & Mission:
- **Deep Dive Analyst:** Kalo nemu data, jangan cuma 'copy-paste'. Analisa polanya, cari anomali, terus kasih tahu Bos apa artinya buat bisnis/projek dia.
- **On-Chain Degen:** Kalo nyari meme token di Pump.fun atau Dexscreener, lu WAJIB cek: Liquiditasnya dikunci gak? Dev-nya 'jeet' (tukang kabur) gak? Ada holder besar yang mencurigakan gak? Jangan asal rekomendasiin token 'rug-pull'.
- **CEX & Spot Hunter:** Kalo Bos nanya buat duit dingin, cari token yang fundamentalnya oke atau lagi ada narasi besar (AI, RWA, DePIN).
- **Future Specialist:** Buat trading futures, analisa teknikal tipis-tipis. Liat Volume, RSI, sama Funding Rate. Kalo market lagi 'bloody', bilang: "Sabar Bos, market lagi gak asik, mending wait and see dulu."
- **Fact-Check First:** Jangan kemakan hoax. Kalo datanya meragukan, lapor: "Waduh Bos, ini data dari sumber X kok agak sus ya, mending kita cross-check lagi."
- **Risk Management:** Lu harus cerewet soal stop-loss. Ingetin Bos: "Pake duit dingin ya Cuy, jangan pake duit beras!"
- **JAWAB SINGKAT, CEKAT, dan to the point.** No fluff, no formalitas (seperlunya saja).

# 🛑 STRICT LANGUAGE RULES
- **Primary Language:** INDONESIAN (Jakarta/Betawi Ala tongkrongan JakSel).
- **NO FORMAL INDONESIAN:** Avoid words like "Saya, Anda".
- **Language Style:** 100% Betawi Gaul tongkrongan JakSel.
## Output Format:
- Gunakan Markdown rapi (Bolding, Lists, Tables).
- **JAWAB SINGKAT, CEKAT, to the point.**

# Aturan Mutlak:
1. Border Ganda: `=======================================` buat header.
2. Border Tunggal: `---------------------------------------` buat pemisah.
3. Bars Visual: `█` dan `░` buat indikator (skala 1-10).

# ✨ SOUL: THE COOL BESTIE
- **Bahasa:** Betawi Gaul (Bos, Cuy, Bang, W, Lu, Sikat, Mantul, Moon, Rug).
- **Style:** Santai tapi solutif.
- Sapa Bos random: "Yoo Bos!", "Gimana kabar Cuy?", "Gas lah Bang!"
- Vibe: Energik, cepat, antusias.
- Emoji Signature: 🚀 atau 🦅
"""

# ========== MODEL REGISTRY ==========
# Cache model list supaya tidak fetch tiap request
_model_cache = {
    "openrouter": {"models": [], "last_update": 0},
    "groq": {"models": [], "last_update": 0},
    "google": {"models": [], "last_update": 0},
    "huggingface": {"models": [], "last_update": 0},
}
CACHE_TTL = 3600  # refresh tiap 1 jam

# Model fallback hardcode kalau fetch gagal
FALLBACK_MODELS = {
    "openrouter": [
        "qwen/qwen3.6-plus:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "google/gemma-3-4b-it:free",
        "deepseek/deepseek-r1:free",
    ],
    "groq": [
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "gemma2-9b-it",
        "mixtral-8x7b-32768",
    ],
    "google": [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ],
    "huggingface": [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "google/gemma-2-2b-it",
    ],
}

# ========== MODEL DISCOVERY ==========

def fetch_openrouter_free_models():
    """Fetch & rank free models dari OpenRouter"""
    try:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("data", [])

        free_models = []
        for m in data:
            pricing = m.get("pricing", {})
            prompt_price = float(pricing.get("prompt", 1))
            completion_price = float(pricing.get("completion", 1))
            if prompt_price == 0 and completion_price == 0:
                # Scoring: context length + recency
                context = m.get("context_length", 0)
                score = context / 1_000_000  # normalize
                free_models.append({
                    "id": m["id"],
                    "score": score,
                    "context": context,
                })

        # Sort by score descending
        free_models.sort(key=lambda x: x["score"], reverse=True)
        models = [m["id"] for m in free_models[:10]]  # top 10
        print(f"[OpenRouter] Found {len(models)} free models")
        return models
    except Exception as e:
        print(f"[OpenRouter] Fetch failed: {e}")
        return FALLBACK_MODELS["openrouter"]


def fetch_groq_models():
    """Fetch free models dari Groq"""
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return FALLBACK_MODELS["groq"]
        client = Groq(api_key=api_key)
        models = client.models.list()
        # Filter yang bukan embed/whisper
        chat_models = [
            m.id for m in models.data
            if "whisper" not in m.id and "embed" not in m.id
        ]
        # Prioritasin instant models (lebih cepet)
        chat_models.sort(key=lambda x: (0 if "instant" in x else 1, x))
        print(f"[Groq] Found {len(chat_models)} models")
        return chat_models[:6]
    except Exception as e:
        print(f"[Groq] Fetch failed: {e}")
        return FALLBACK_MODELS["groq"]


def fetch_google_models():
    """Fetch free models dari Google Gemini"""
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return FALLBACK_MODELS["google"]
        genai.configure(api_key=api_key)
        models = genai.list_models()
        # Filter yang support generateContent dan gratis (flash/lite)
        free_models = []
        for m in models:
            if "generateContent" in m.supported_generation_methods:
                name = m.name.replace("models/", "")
                if any(kw in name for kw in ["flash", "lite", "8b"]):
                    free_models.append(name)
        # Sort: flash-2.0 dulu
        free_models.sort(key=lambda x: (0 if "2.0" in x else 1, x))
        print(f"[Google] Found {len(free_models)} free models")
        return free_models if free_models else FALLBACK_MODELS["google"]
    except Exception as e:
        print(f"[Google] Fetch failed: {e}")
        return FALLBACK_MODELS["google"]


def fetch_huggingface_models():
    """HuggingFace tidak ada free model API, pakai hardcode tapi bisa expand"""
    # HuggingFace inference API tidak ada endpoint list free models
    # jadi tetap pakai curated list yang udah ditest
    return FALLBACK_MODELS["huggingface"]


def get_models(provider):
    """Get cached models atau fetch baru kalau expired"""
    cache = _model_cache[provider]
    now = time.time()
    if not cache["models"] or (now - cache["last_update"]) > CACHE_TTL:
        print(f"[Cache] Refreshing {provider} models...")
        fetchers = {
            "openrouter": fetch_openrouter_free_models,
            "groq": fetch_groq_models,
            "google": fetch_google_models,
            "huggingface": fetch_huggingface_models,
        }
        cache["models"] = fetchers[provider]()
        cache["last_update"] = now
    return cache["models"]


# ========== PROVIDER CALLERS ==========

def call_openrouter(message, model):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise Exception("No API key")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ],
        timeout=30
    )
    return resp.choices[0].message.content


def call_groq(message, model):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise Exception("No API key")
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ],
    )
    return resp.choices[0].message.content


def call_google(message, model):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise Exception("No API key")
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(
        model_name=model,
        system_instruction=SYSTEM_PROMPT
    )
    resp = m.generate_content(message)
    return resp.text


def call_huggingface(message, model):
    api_key = os.environ.get("HF_API_KEY")
    if not api_key:
        raise Exception("No API key")
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
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# Provider config: name, caller, enabled check
PROVIDER_CONFIG = [
    ("groq",        call_groq,         lambda: os.environ.get("GROQ_API_KEY")),
    ("openrouter",  call_openrouter,   lambda: os.environ.get("OPENROUTER_API_KEY")),
    ("google",      call_google,       lambda: os.environ.get("GOOGLE_API_KEY")),
    ("huggingface", call_huggingface,  lambda: os.environ.get("HF_API_KEY")),
]


# ========== SMART FALLBACK ENGINE ==========

def get_ai_response(message):
    """
    Auto-discovery + smart fallback:
    - Groq dulu (paling cepet)
    - OpenRouter (banyak model gratis)
    - Google
    - HuggingFace
    Tiap provider coba semua modelnya sebelum pindah ke provider berikutnya
    """
    all_errors = []

    for provider_name, caller, key_check in PROVIDER_CONFIG:
        if not key_check():
            continue

        models = get_models(provider_name)
        print(f"[{provider_name}] Trying {len(models)} models...")

        for model in models:
            try:
                start = time.time()
                reply = caller(message, model)
                elapsed = round(time.time() - start, 2)
                print(f"[{provider_name}] ✅ {model} ({elapsed}s)")
                return reply, provider_name, model
            except Exception as e:
                err_msg = str(e)[:100]
                print(f"[{provider_name}] ❌ {model}: {err_msg}")
                all_errors.append(f"{provider_name}/{model}: {err_msg}")
                continue

    return (
        "❌ Waduh Bos, semua provider lagi KO nih!\n" +
        "\n".join(all_errors[:5]),
        "none",
        "none"
    )


# ========== BACKGROUND MODEL REFRESH ==========

def background_refresh():
    """Refresh model list di background tiap 1 jam"""
    while True:
        time.sleep(CACHE_TTL)
        print("[Cache] Background refresh started...")
        for provider in _model_cache.keys():
            _model_cache[provider]["last_update"] = 0  # force refresh
        # Trigger fetch
        for provider_name, _, key_check in PROVIDER_CONFIG:
            if key_check():
                get_models(provider_name)


# ========== TELEGRAM HANDLERS ==========

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Yoo Bos! W udah siap! 🚀\n\n"
        "Auto-pilih model terbaik dari:\n"
        "⚡ Groq • 🔀 OpenRouter\n"
        "🔵 Google • 🤗 HuggingFace\n\n"
        "Kirim ticker `$BTC` atau tanya apapun!\n"
        "/help buat command lengkap",
        parse_mode="Markdown"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *Command:*\n\n"
        "/start - Mulai bot\n"
        "/help - Bantuan\n"
        "/models - Lihat model aktif\n"
        "/status - Cek semua provider\n"
        "/refresh - Refresh model list\n\n"
        "Kirim `$BTC`, `$SOL`, dll buat analisa! 🦅",
        parse_mode="Markdown"
    )


async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = ["🤖 *Model Aktif Bos:*\n"]
    emojis = {"groq": "⚡", "openrouter": "🔀", "google": "🔵", "huggingface": "🤗"}
    for provider_name, _, key_check in PROVIDER_CONFIG:
        if not key_check():
            lines.append(f"{emojis[provider_name]} *{provider_name}*: ❌ No key")
            continue
        models = get_models(provider_name)
        lines.append(f"{emojis[provider_name]} *{provider_name}* ({len(models)} models):")
        for m in models[:3]:
            lines.append(f"  • `{m}`")
        if len(models) > 3:
            lines.append(f"  • _...+{len(models)-3} more_")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    checking = await update.message.reply_text("🔍 Ngecek semua provider...")
    lines = ["📊 *Status Provider:*\n"]
    emojis = {"groq": "⚡", "openrouter": "🔀", "google": "🔵", "huggingface": "🤗"}

    for provider_name, caller, key_check in PROVIDER_CONFIG:
        if not key_check():
            lines.append(f"{emojis[provider_name]} *{provider_name}*: ❌ No key")
            continue
        models = get_models(provider_name)
        if not models:
            lines.append(f"{emojis[provider_name]} *{provider_name}*: ⚠️ No models")
            continue
        # Test dengan model pertama
        try:
            start = time.time()
            caller("hi", models[0])
            elapsed = round(time.time() - start, 2)
            lines.append(f"{emojis[provider_name]} *{provider_name}*: ✅ {elapsed}s")
        except Exception as e:
            lines.append(f"{emojis[provider_name]} *{provider_name}*: ❌ Error")

    await checking.edit_text("\n".join(lines), parse_mode="Markdown")


async def refresh_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔄 Refreshing model list dari semua provider...")
    for provider in _model_cache.keys():
        _model_cache[provider]["last_update"] = 0
    counts = []
    for provider_name, _, key_check in PROVIDER_CONFIG:
        if key_check():
            models = get_models(provider_name)
            counts.append(f"• {provider_name}: {len(models)} models")
    await msg.edit_text(
        "✅ *Refresh selesai Bos!*\n\n" + "\n".join(counts),
        parse_mode="Markdown"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    user_message = update.message.text
    thinking_msg = await update.message.reply_text("⏳ Gaspol...")

    reply, provider_used, model_used = get_ai_response(user_message)

    await thinking_msg.delete()

    # Truncate model name biar ga kepanjangan
    model_short = model_used.split("/")[-1] if "/" in model_used else model_used

    await update.message.reply_text(
        f"{reply}\n\n_⚡ {provider_used} • {model_short}_",
        parse_mode="Markdown"
    )


# ========== MAIN ==========

async def post_init(application: Application):
    await application.bot.set_my_commands([
        BotCommand("start", "Mulai bot"),
        BotCommand("help", "Tampilkan bantuan"),
        BotCommand("models", "Lihat model aktif"),
        BotCommand("status", "Cek status provider"),
        BotCommand("refresh", "Refresh model list"),
    ])
    # Pre-load model list di background
    print("[Init] Pre-loading model lists...")
    for provider_name, _, key_check in PROVIDER_CONFIG:
        if key_check():
            get_models(provider_name)
    print("[Init] Model lists loaded!")


def main():
    # Start background refresh thread
    refresh_thread = threading.Thread(target=background_refresh, daemon=True)
    refresh_thread.start()

    token = os.environ.get("TELEGRAM_TOKEN")
    app = Application.builder().token(token).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("models", models_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("refresh", refresh_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("[INFO] Bot started polling...")
    app.run_polling()


if __name__ == "__main__":
    main()
