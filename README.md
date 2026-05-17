# Lensure – Watermarking and Image Verification System

Lensure is a prototype system for digital image watermarking and integrity verification. It allows embedding cryptographic and perceptual information into images and evaluating their robustness under different types of attacks.

---

## Features

The system implements the following core functionalities:

- Perceptual hashing of images
- RSA-based digital signature generation and verification
- Watermark embedding using LSB (Least Significant Bit) and DWT (Discrete Wavelet Transform) techniques
- Reed-Solomon error correction encoding
- Watermark extraction and decoding
- Robustness evaluation under multiple attack categories:
  - **Signal-processing attacks**: JPEG compression, resizing, Gaussian blur, noise injection
  - **Semantic attacks**: AI-driven inpainting via Stable Diffusion (soft and hard variants)
  - **Social media attacks**: image redistribution through Bluesky and Telegram
  - **Substitution**: random image replacement
- Automatic evaluation of image authenticity based on signature validation and perceptual hash distance

---

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1) as the Python package manager and runner.

Install the system dependency required for visualization:

```bash
sudo apt install python3-tk
```

Then install project dependencies:

```bash
uv sync
```

---

## Configuration

Social media attacks require API credentials:

| Variable | Description |
|---|---|
| `BLUESKY_HANDLE` | Your Bluesky handle (e.g. `user.bsky.social`) |
| `BLUESKY_APP_PASSWORD` | App password from https://bsky.app/settings/app-passwords |
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_CHAT_ID` | Chat ID to send images to (see `.env` for instructions) |

Social media attacks are skipped automatically if the credentials are not set.

---

## Usage

### Run a single image experiment

```bash
uv run lensure run-experiment <image-path> --embed-method DWT --no-embed-og-hash
```

### Run a complete batch experiment

```bash
uv run lensure complete-execution <images-dir> <output-dir> --embed-method DWT
```

---

## Watermarking Modes

The system supports two watermarking strategies controlled by the `embed_og_hash` flag:

### Signature-only mode (`--no-embed-og-hash`)

Only the RSA signature of the perceptual hash is embedded.

**Validation**: the extracted signature must verify against the perceptual hash of the image being checked.

> Note: this mode assumes the watermarking method does not alter the perceptual hash. LSB embedding satisfies this; DWT may shift the hash by a few bits and cause verification to fail even on unmodified images.

### Full embedding mode (default)

Both the original perceptual hash and its RSA signature are embedded.

**Validation**: the embedded signature must be valid, and the perceptual hash distance between the original (embedded) hash and the current image hash must be below a threshold `δ` (default 10).

---

## Dataset Requirements

The input image must be inside a directory that contains at least two images. The system selects a second image from the same directory to simulate the substitution attack.

---

## Testing

Run the unit test suite:

```bash
uv run pytest
```

Run only the integration tests (requires credentials in `.env`):

```bash
uv run pytest -m integration
```

Integration tests are excluded from CI automatically via `.gitlab-ci.yml`.

---

## Notes

- This is a research/prototype implementation intended for experimentation and robustness evaluation, not production use.
- The Stable Diffusion model (`runwayml/stable-diffusion-inpainting`) must be downloaded before first use and is loaded from the local Hugging Face cache.
