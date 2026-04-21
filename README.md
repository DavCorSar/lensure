# Lensure – Watermarking and Image Verification System

Lensure is a prototype system for digital image watermarking and integrity verification. It allows embedding cryptographic and perceptual information into images and evaluating their robustness under different types of attacks.

---

## Features

The system implements the following core functionalities:

- Perceptual hashing of images
- RSA-based digital signature generation and verification
- Watermark embedding using LSB (Least Significant Bit) technique
- Reed-Solomon error correction encoding
- Watermark extraction and decoding
- Robustness evaluation under image attacks:
  - JPEG compression
  - Resizing
  - Gaussian blur
  - Noise injection
  - Random image replacement
- Automatic evaluation of image authenticity based on:
  - Signature validation
  - Perceptual hash distance

---

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1) as the Python package manager and runner.

Install system dependency required for visualization:

```bash
sudo apt install python3-tk
```

Then install and run dependencies using `uv`.

---

## Usage

The CLI provides an experiment runner that applies different attacks to an image and evaluates watermark robustness.

### Basic execution
```bash
sudo apt install python3-tk
```

### Disable original hash embedding

```bash
uv run lensure <image-path> --no-embed-og-hash
```

## Watermarking Modes

The system supports two watermarking strategies controlled by the `embed_og_hash` flag:

### 1. Signature-only mode (`--no-embed-og-hash`)

Only the signed perceptual hash is embedded in the image.

**Validation rule:**
- The extracted hash from the attacked image must match the signature embedded in the watermark.

---

### 2. Full embedding mode (default)

Both the original perceptual hash and its signature are embedded.

**Validation rules:**
- The original hash must correctly validate the signature.
- The perceptual hash of the attacked image must be within a threshold distance `δ` of the embedded hash.

---

## Dataset Requirements

The input image path must point to an image inside a directory that contains at least two images.

This is required because the system automatically selects a second random image from the same directory and treats it as an additional attack case. This is used to test whether the system incorrectly validates unrelated images.

---

## Notes

- This is a research/prototype implementation.
- The system is intended for experimentation and robustness evaluation rather than production use.
- Future improvements may include more robust serialization formats and alternative watermarking domains (e.g., frequency domain methods).
