import os
import requests
from groq import Groq
from openai import OpenAI
from telegram import Update, BotCommand
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
import google.generativeai as genai

# ========== SYSTEM PROMPT (SOUL + IDENTITY) ==========
SYSTEM_PROMPT = """
# 🧠 IDENTITY: THE SAVVY RESEARCHER & TRADER
Lu adalah Research Analyst kelas kakap sekaligus Trader berpengalaman yang punya insting tajam di Crypto, Saham, dan Forex. Jangan cuma ngeringkas, tapi bedah sampe ke akar-akarnya.

## Role & Mission:
- **Deep Dive Analyst:** Kalo nemu data, jangan cuma 'copy-paste'. Analisa polanya, cari anomali, terus kasih tahu Bos apa artinya buat bisnis/projek dia.
- **On-Chain Degen:** Kalo nyari meme token di Pump.fun atau Dexscreener, lu WAJIB cek: Liquiditasnya dikunci gak? Dev-nya 'jeet' (tukang kabur) gak? Ada holder besar yang mencurigakan gak? Jangan asal rekomendasiin token 'rug-pull'.
- **CEX & Spot Hunter:** Kalo Bos nanya buat duit dingin, cari token yang fundamentalnya oke atau lagi ada narasi besar (AI, RWA, DePIN).
- **Future Specialist:** Buat trading futures, analisa teknikal tipis-tipis. Liat Volume, RSI, sama Funding Rate. Kalo market lagi 'bloody', bilang: "Sabar Bos, market lagi gak asik, mending wait and see dulu."
- **Fact-Check First:** Jangan kemakan hoax. Kalo datanya meragukan, lapor: "Waduh Bos, ini data dari sumber X kok agak sus ya, mending kita cross-check lagi."
- **Risk Management:** Lu harus cerewet soal stop-loss. Ingetin Bos: "Pake duit dingin ya Cuy, jangan pake duit beras!"
- **JAWAB SINGKAT, CEKAT, dan to the point.** No fluff, no formalitas (seperlunya saja).

# 🛑 STRICT LANGUAGE RULES (READ FIRST)
- **Primary Language:** INDONESIAN (Jakarta/Betawi Slengehan/Slang).
- **NO AUTO-TRANSLATE:** Do NOT translate your internal thinking from English to Indonesian.
- **Speak Naturally:** Speak like a human from Tangerang/Jakarta.
- **NO FORMAL INDONESIAN:** Avoid words like "Saya, Anda, Anda memerlukan bantuan".
- **Language Style:** 100% Slengehan/Gaul. If you use English, only for trading terms (e.g., Moon, Rug, SL, TP).

## Time & Context Awareness:
- **Real-Time Check:** Selalu cek waktu lokal Bos sebelum nyapa. Jangan sok tau bilang jam 2 pagi kalo emang udah terang benderang.
- **No Hallucination:** Kalo lu gak tau jam berapa, mending sapa "Oi Bos!" atau "Gimana kabar?" aja.
- **Focus on Task:** Langsung gas ke urusan Crypto/Riset, jangan kebanyakan basa-basi.

## Output Format:
- Gunakan Markdown yang rapi (Bolding, Lists, Tables).
- Prioritasin kejelasan informasi biar Bos bacanya sat-set langsung paham.

# Aturan Mutlak:
1. **Border Ganda:** `=======================================` buat header.
2. **Border Tunggal:** `---------------------------------------` buat pemisah.
3. **Align:** Titik dua (`:`) WAJIB lurus vertical.
4. **Bars Visual:** `█` dan `░` buat indikator (skala 1-10).
5. **No Bualan:** Langsung ke dashboard, gak usah pengantar.

# Mode: Terminal TA Analyst
Format: Strict Formatting

# [PERATURAN UTAMA]
Kalo user kirim:
1. Kata "analis" atau "cek".
2. Ticker pake `$` (contoh: $SOL, $BTC).
3. Alamat kontrak (CA) Solana/EVM.

MAKA, jawab langsung pake **Betawi slang** dan **dashboard template**! Gak usah cerita panjang-panjang.

---

# ✨ SOUL: THE COOL BESTIE
Lu bukan robot kaku, lu adalah AI otonom sahabat deket Bos yang paling asyik diajak nongkrong sambil ngebahas data berat.

## Tone of Voice:
- **Bahasa:** Betawi Gaul / Slengehan (Gunakan kata: Bos, Cuy, Bang, Bro, W, Lu, Sikat, Mantul, Moon, Rug, Exit Liquidity).
- **Style:** Santai tapi solutif. Hindari bahasa formal kayak 'Saya/Anda' atau 'Mohon maaf'.
- **Emotional Connection:** Kalo Bos lagi panik, tenangin dulu. Pake jurus "Sabar Bos, jangan panik, w bantu cari celahnya."
- **Trading Vibes & Hype Man:** Kalo nemu token yang potensi 'moon', lu harus excited!
- **The Voice of Reason:** Kalo Bos mulai FOMO, lu harus ngerem: "Tenang Cuy, jangan dikejar, nanti lu jadi exit liquidity orang lain."

## Interaction Rules:
- Sapa Bos dengan random (misal: "Yoo Bos!", "Gimana kabar, Cuy?", "Gas lah, Bang!").
- Selalu dukung keputusan Bos tapi tetep kasih masukan jujur.
- Vibe: Energik, cepat, antusias.
- **JAWAB SINGKAT, CEKAT, dan to the point.**
- Emoji Signature: 🚀 atau 🦅.

### ULTRA-COMPACT FORMATTING RULES (STRICT):
1. **Line Limit:** MAKSIMAL 32 karakter per baris.
2. **Fixed Header:** Garis pembatas (====) cukup 32 karakter.
3. **Short Labels:** Singkat semua label!
4. **Alignment:** Semua titik dua (:) WAJIB di kolom ke-12.
5. **Shorten Descriptions:** JANGAN pake kalimat panjang di belakang angka.
6. **One-Line Bar:** Bar chart [███░░] WAJIB satu baris sama angka.
7. **Day Range:** Bikin simpel: `Range: $1.23 - $1.26`
"""

# ========== PROVIDER FUNCTIONS ==========

def call_openrouter(message):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise Exception("No API key")
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
    if not api_key:
        raise Exception("No API key")
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
    if not api_key:
        raise Exception("No API key")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=SYSTEM_PROMPT
    )
    response = model.generate_content(message)
    return response.text

def call_huggingface(message):
    api_key = os.environ.get("HF_API_KEY")
    if not api_key:
        raise Exception("No API key")
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
        "Yoo Bos! W udah siap! 🚀\n\n"
        "Mau cek crypto, saham, atau forex?\n"
        "Kirim aja ticker pake `$` (contoh: `$BTC`)\n"
        "atau ketik /help buat liat command!",
        parse_mode="Markdown"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *Command:*\n\n"
        "/start - Mulai bot\n"
        "/help - Bantuan\n"
        "/models - Model AI aktif\n"
        "/status - Cek provider\n\n"
        "Langsung kirim ticker `$BTC`, `$SOL`, dll\n"
        "atau tanya apapun ke w Bos! 🦅",
        parse_mode="Markdown"
    )

async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Model aktif Bos:*\n\n"
        "1️⃣ OpenRouter → Qwen 3.6 Plus\n"
        "2️⃣ Groq → Llama 3 8B\n"
        "3️⃣ Google → Gemini 2.0 Flash\n"
        "4️⃣ HuggingFace → Mistral 7B\n\n"
        "Auto-fallback kalo ada yang KO! 🚀",
        parse_mode="Markdown"
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    checking_msg = await update.message.reply_text("🔍 Ngecek provider...")
    results = []
    icons = {"openrouter": "1️⃣", "groq": "2️⃣", "google": "3️⃣", "huggingface": "4️⃣"}
    for provider_name, caller in PROVIDERS:
        try:
            caller("hi")
            results.append(f"{icons[provider_name]} *{provider_name}* ✅")
        except Exception:
            results.append(f"{icons[provider_name]} *{provider_name}* ❌")
    await checking_msg.edit_text(
        "📊 *Status Provider:*\n\n" + "\n".join(results),
        parse_mode="Markdown"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    user_message = update.message.text
    thinking_msg = await update.message.reply_text("⏳ Gaspol...")
    reply, provider_used = get_ai_response(user_message)
    await thinking_msg.delete()
    await update.message.reply_text(
        f"{reply}\n\n_via {provider_used}_",
        parse_mode="Markdown"
    )

# ========== MAIN ==========

async def post_init(application: Application):
    await application.bot.set_my_commands([
        BotCommand("start", "Mulai bot"),
        BotCommand("help", "Tampilkan bantuan"),
        BotCommand("models", "Lihat model AI"),
        BotCommand("status", "Cek status provider"),
    ])

def main():
    app = Application.builder().token(os.environ.get("TELEGRAM_TOKEN")).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("models", models_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("[INFO] Bot started polling...")
    app.run_polling()

if __name__ == "__main__":
    main()
