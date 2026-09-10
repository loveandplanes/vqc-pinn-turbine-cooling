"""
backup_assistant.py — Terminal coding assistant powered by NVIDIA NIM.
Run this when Antigravity is on cooldown.

Usage:
    python backup_assistant.py

Commands:
    /file <path>   — Load a file into context for the AI to analyze
    /model         — Switch between available NVIDIA models
    /think         — Toggle visible reasoning (thinking) output
    /clear         — Clear conversation history
    /help          — Show available commands
    /quit          — Exit
"""

import sys
import os
from pathlib import Path
from config import get_nvidia_client

# --- Colors for reasoning vs content ---
_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
_THINKING_COLOR = "\033[90m" if _USE_COLOR else ""  # Grey
_RESET_COLOR = "\033[0m" if _USE_COLOR else ""

# --- Available NVIDIA NIM Models ---
MODELS = {
    "1": {
        "id": "z-ai/glm-5.1",
        "name": "GLM 5.1 (Thinking)",
        "thinking": True,
    },
    "2": {
        "id": "meta/llama-3.3-70b-instruct",
        "name": "Llama 3.3 70B",
        "thinking": False,
    },
    "3": {
        "id": "nvidia/llama-3.1-nemotron-70b-instruct",
        "name": "Nemotron 70B (Coding)",
        "thinking": False,
    },
    "4": {
        "id": "deepseek-ai/deepseek-r1",
        "name": "DeepSeek R1 (Reasoning)",
        "thinking": True,
    },
    "5": {
        "id": "meta/llama-3.1-8b-instruct",
        "name": "Llama 3.1 8B (Fast)",
        "thinking": False,
    },
}

# --- System prompt tailored to your work ---
SYSTEM_PROMPT = """You are an expert coding assistant specializing in:
- Python (scientific computing, NumPy, PyTorch, PennyLane)
- Quantum computing (VQC, QAOA, Qiskit, PennyLane)
- Aerospace engineering (CFD, airfoil design, turbine optimization)
- Neural networks and machine learning

You help debug code, write new features, and explain concepts.
Be concise but thorough. Include code examples when relevant."""


def print_banner():
    print()
    print("=" * 55)
    print("  NVIDIA NIM Backup Assistant")
    print("  (for when Antigravity is on cooldown)")
    print("=" * 55)
    print()


def select_model():
    print("\n  Available Models:")
    print("  " + "-" * 45)
    for key, info in MODELS.items():
        tag = " [thinking]" if info["thinking"] else ""
        print(f"  [{key}] {info['name']}{tag}")
    print()

    choice = input("  Pick a model [1]: ").strip() or "1"
    if choice in MODELS:
        info = MODELS[choice]
        print(f"  → Using: {info['name']}")
        return info
    else:
        print("  → Invalid choice, using GLM 5.1")
        return MODELS["1"]


def load_file(path_str):
    """Load a file's content to include in the next message."""
    path = Path(path_str.strip().strip('"').strip("'"))
    if not path.exists():
        alt = Path(__file__).parent.parent / path
        if alt.exists():
            path = alt
        else:
            print(f"  File not found: {path}")
            return None

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.count("\n") + 1
        print(f"  Loaded: {path.name} ({lines} lines)")
        return f"--- File: {path.name} ---\n{content}\n--- End of {path.name} ---"
    except Exception as e:
        print(f"  Error reading file: {e}")
        return None


def stream_response(client, model_info, messages, show_thinking):
    """Send request and stream the response, handling reasoning tokens."""

    # Build request kwargs
    kwargs = dict(
        model=model_info["id"],
        messages=messages,
        temperature=1,
        top_p=1,
        max_tokens=16384,
        stream=True,
    )

    # Add thinking support for models that support it
    if model_info["thinking"]:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "clear_thinking": False,
            }
        }

    completion = client.chat.completions.create(**kwargs)

    full_response = ""
    was_thinking = False

    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        if len(chunk.choices) == 0 or getattr(chunk.choices[0], "delta", None) is None:
            continue

        delta = chunk.choices[0].delta

        # --- Reasoning/thinking tokens (shown in grey) ---
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            if show_thinking:
                if not was_thinking:
                    print(f"\n  {_THINKING_COLOR}[thinking] ", end="")
                    was_thinking = True
                print(f"{_THINKING_COLOR}{reasoning}{_RESET_COLOR}", end="")

        # --- Main content tokens ---
        if getattr(delta, "content", None) is not None:
            if was_thinking:
                print(f"{_RESET_COLOR}\n\n  AI > ", end="")
                was_thinking = False
            print(delta.content, end="", flush=True)
            full_response += delta.content

    print("\n")
    return full_response


def main():
    client = get_nvidia_client()

    print_banner()
    current_model = select_model()
    show_thinking = True  # Show reasoning by default

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    file_context = None

    print(f"\n  Thinking visible: {'ON' if show_thinking else 'OFF'}")
    print("  Type your question (or /help for commands).\n")

    while True:
        try:
            user_input = input("  You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue

        # --- Commands ---
        if user_input.startswith("/"):
            cmd = user_input.lower().split()[0]

            if cmd in ("/quit", "/exit"):
                print("  Goodbye!")
                break
            elif cmd == "/clear":
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                file_context = None
                print("  Conversation cleared.\n")
                continue
            elif cmd == "/model":
                current_model = select_model()
                continue
            elif cmd == "/think":
                show_thinking = not show_thinking
                state = "ON (grey text)" if show_thinking else "OFF (hidden)"
                print(f"  Thinking output: {state}\n")
                continue
            elif cmd == "/file":
                path = user_input[5:].strip()
                if path:
                    file_context = load_file(path)
                else:
                    print('  Usage: /file path/to/your_script.py')
                continue
            elif cmd == "/help":
                print("  Commands:")
                print("    /file <path>  — Load a file for AI to analyze")
                print("    /model        — Switch NVIDIA model")
                print("    /think        — Toggle reasoning visibility")
                print("    /clear        — Clear conversation history")
                print("    /quit         — Exit")
                print()
                continue
            else:
                print(f"  Unknown command: {cmd}. Type /help")
                continue

        # --- Build message with optional file context ---
        content = user_input
        if file_context:
            content = f"{file_context}\n\n{user_input}"
            file_context = None

        messages.append({"role": "user", "content": content})

        # --- Call NVIDIA NIM ---
        try:
            print("\n  AI > ", end="", flush=True)
            full_response = stream_response(
                client, current_model, messages, show_thinking
            )
            messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "auth" in error_msg.lower():
                print(f"\n  Auth error — check your NVIDIA_API_KEY in .env")
            elif "429" in error_msg:
                print(f"\n  Rate limited — wait a moment and try again")
            else:
                print(f"\n  Error: {e}")
            print()


if __name__ == "__main__":
    main()
