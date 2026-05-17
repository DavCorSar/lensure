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


class BlueskyClientPool:
    def __init__(self, clients: list[BlueskyClient]):
        if not clients:
            raise ValueError("At least one Bluesky client is required")
        self._clients = clients
        self._index = 0

    @property
    def current(self) -> BlueskyClient:
        return self._clients[self._index]

    def rotate(self) -> None:
        next_index = (self._index + 1) % len(self._clients)
        if next_index == self._index:
            raise RuntimeError("All Bluesky accounts have hit the rate limit")
        print(f"[bluesky] Rate limit hit on account {self._index + 1}/{len(self._clients)}, rotating...")
        self._index = next_index

    def __len__(self) -> int:
        return len(self._clients)


def create_bluesky_client_pool() -> BlueskyClientPool:
    handles_raw = os.environ.get("BLUESKY_HANDLES", "")
    passwords_raw = os.environ.get("BLUESKY_APP_PASSWORDS", "")

    handles = [h.strip() for h in handles_raw.split(",") if h.strip()]
    passwords = [p.strip() for p in passwords_raw.split(",") if p.strip()]

    if not handles or not passwords:
        raise EnvironmentError(
            "BLUESKY_HANDLES and BLUESKY_APP_PASSWORDS must be set in .env"
        )
    if len(handles) != len(passwords):
        raise EnvironmentError(
            "BLUESKY_HANDLES and BLUESKY_APP_PASSWORDS must have the same number of entries"
        )

    clients = []
    for handle, password in zip(handles, passwords):
        client = BlueskyClient()
        client.login(handle, password)
        clients.append(client)

    return BlueskyClientPool(clients)


def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "429" in msg or "rate limit" in msg or "ratelimit" in msg


_CDN_MAX_RETRIES = 3
_CDN_RETRY_WAIT = 10  # seconds between CDN download retries


def _upload_to_bluesky(image: Image.Image, client: BlueskyClient) -> Image.Image:
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

    try:
        for attempt in range(_CDN_MAX_RETRIES):
            try:
                response = requests.get(cdn_url, timeout=30)
                response.raise_for_status()
                return Image.open(io.BytesIO(response.content)).convert("RGB")
            except requests.RequestException as e:
                if attempt < _CDN_MAX_RETRIES - 1:
                    print(f"[bluesky] CDN download failed ({e}), retrying in {_CDN_RETRY_WAIT}s...")
                    time.sleep(_CDN_RETRY_WAIT)
                else:
                    raise
    finally:
        client.delete_post(post.uri)


def bluesky_attack(image: Image.Image, client_pool: BlueskyClientPool | None = None) -> Image.Image:
    """
    Uploads the image to Bluesky and returns the version served by their CDN,
    which applies JPEG compression and may resize the image.
    The post is deleted after downloading the result.

    Pass a pre-authenticated BlueskyClientPool to avoid login calls on every
    invocation and to enable automatic rotation on rate limit errors.
    """
    if client_pool is None:
        client_pool = create_bluesky_client_pool()

    for attempt in range(len(client_pool)):
        try:
            return _upload_to_bluesky(image, client_pool.current)
        except Exception as e:
            if _is_rate_limit_error(e) and attempt < len(client_pool) - 1:
                client_pool.rotate()
            else:
                raise


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
