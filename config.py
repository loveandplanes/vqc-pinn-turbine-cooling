"""
config.py — Secure NVIDIA NIM API key loader.

Searches for .env in this order:
  1. The current project folder
  2. The parent pythonfolder/.env (shared across all projects)

Usage in any script:
    from config import get_nvidia_client
    client = get_nvidia_client()
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# --- Load .env from multiple locations ---
local_env = Path(__file__).parent / ".env"
root_env = Path(__file__).parent.parent / ".env"

if root_env.exists():
    load_dotenv(root_env)
if local_env.exists():
    load_dotenv(local_env, override=True)

# --- Read the NVIDIA API key ---
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# --- Safety check ---
if not NVIDIA_API_KEY or NVIDIA_API_KEY == "your-nvapi-key-here":
    print("=" * 50)
    print("ERROR: No NVIDIA API key found!")
    print()
    print("1. Go to https://build.nvidia.com")
    print("2. Sign in and get a free API key")
    print("3. Paste it in your .env file:")
    print(f'   NVIDIA_API_KEY="nvapi-..."')
    print("=" * 50)
    sys.exit(1)

# --- NVIDIA NIM base URL ---
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

def get_nvidia_client():
    """Returns an OpenAI-compatible client pointing to NVIDIA NIM."""
    return OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=NVIDIA_API_KEY
    )
