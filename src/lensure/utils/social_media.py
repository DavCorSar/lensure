"""
Benign social media attacks: upload an image to a platform and download
the version the platform serves back, which includes its own compression
and resizing pipeline.
"""

import io
import os
import time

import requests
from dotenv import load_dotenv
from PIL import Image
from atproto import Client as BlueskyClient
from atproto import models as atproto_models

load_dotenv()


def bluesky_attack(image: Image.Image) -> Image.Image:
    """
    Uploads the image to Bluesky and returns the version served by their CDN,
    which applies JPEG compression and may resize the image.
    The post is deleted after downloading the result.
    """
    handle = os.environ.get("BLUESKY_HANDLE")
    password = os.environ.get("BLUESKY_APP_PASSWORD")

    if not handle or not password:
        raise EnvironmentError(
            "BLUESKY_HANDLE and BLUESKY_APP_PASSWORD must be set in .env"
        )

    client = BlueskyClient()
    client.login(handle, password)

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=95)
    buf.seek(0)

    upload = client.upload_blob(buf.read())

    embed = atproto_models.AppBskyEmbedImages.Main(
        images=[atproto_models.AppBskyEmbedImages.Image(alt="", image=upload.blob)]
    )
    post = client.send_post(text="", embed=embed)

    time.sleep(3)

    did = client.me.did
    cid = upload.blob.ref.link
    cdn_url = f"https://cdn.bsky.app/img/feed_fullsize/plain/{did}/{cid}@jpeg"

    response = requests.get(cdn_url, timeout=30)
    response.raise_for_status()

    client.delete_post(post.uri)

    return Image.open(io.BytesIO(response.content)).convert("RGB")


def telegram_attack(image: Image.Image) -> Image.Image:
    """
    Sends the image to Telegram as a photo (not a document) and returns
    the version Telegram serves back, which applies JPEG compression and
    caps resolution at 1280px on the longest side.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        raise EnvironmentError(
            "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env"
        )

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    buf.seek(0)

    send_response = requests.post(
        f"https://api.telegram.org/bot{token}/sendPhoto",
        data={"chat_id": chat_id},
        files={"photo": ("image.png", buf, "image/png")},
        timeout=30,
    )
    send_response.raise_for_status()

    photo_sizes = send_response.json()["result"]["photo"]
    file_id = photo_sizes[-1]["file_id"]

    file_response = requests.get(
        f"https://api.telegram.org/bot{token}/getFile",
        params={"file_id": file_id},
        timeout=30,
    )
    file_response.raise_for_status()
    file_path = file_response.json()["result"]["file_path"]

    download_response = requests.get(
        f"https://api.telegram.org/file/bot{token}/{file_path}",
        timeout=30,
    )
    download_response.raise_for_status()

    return Image.open(io.BytesIO(download_response.content)).convert("RGB")
