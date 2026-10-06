import os

import httpx
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = "http://127.0.0.1:8000/predict"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Send me an image and I'll tell you whether it is AI-generated or real."
    )


async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message

    # Get the highest-resolution version of the Telegram photo
    photo = message.photo[-1]

    telegram_file = await context.bot.get_file(photo.file_id)

    # Download image into memory
    image_bytes = await telegram_file.download_as_bytearray()

    await message.reply_text("🔍 Analyzing image...")

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                API_URL,
                files={
                    "file": (
                        "image.jpg",
                        bytes(image_bytes),
                        "image/jpeg",
                    )
                },
            )

        response.raise_for_status()
        result = response.json()

        prediction = result["prediction"]
        confidence = result["confidence"] * 100

        await message.reply_text(
            f"🤖 Prediction: {prediction}\n"
            f"🎯 Confidence: {confidence:.2f}%"
        )

    except Exception as e:
        print(f"Error: {e}")

        await message.reply_text(
            "❌ Something went wrong while analyzing the image."
        )


def main():
    if not BOT_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN environment variable is not set."
        )

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(
        MessageHandler(filters.PHOTO, handle_image)
    )

    print("Telegram bot is running...")

    app.run_polling()


if __name__ == "__main__":
    main()