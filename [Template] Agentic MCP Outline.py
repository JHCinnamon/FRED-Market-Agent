"""Desktop client for an LM Studio agent using tools from the FRED API."""

import asyncio
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict


REQUIRED_PACKAGES = {
    "openai": "openai",
    "pydantic": "pydantic",
    "requests": "requests",
}
LM_STUDIO_MODEL = "qwen3.6-27b"
LM_STUDIO_HOST = "http://localhost:1234"


def ensure_packages() -> None:
    """Install the Python dependencies when this script is started directly."""
    missing_packages = [
        package
        for module, package in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])


ensure_packages()

import tkinter as tk
from tkinter import scrolledtext, ttk

from openai import OpenAI
from LocalFREDAgent import LocalFREDAgent


class FredAgentApp(tk.Tk):
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.title("FRED Economic Data Agent")
        self.minsize(760, 560)
        self.geometry("900x680")
        self.conversation: list[Dict[str, str]] = []
        self._configure_ollmcp_theme()
        self._build_ui()

    def _configure_ollmcp_theme(self) -> None:
        colors = {
            "ink": "#10151f",
            "surface": "#18212f",
            "cyan": "#5eead4",
            "amber": "#fbbf24",
            "text": "#e5edf5",
            "muted": "#a4b1c1",
        }
        self.configure(background=colors["ink"])
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(
            "OllMCP.TButton",
            background=colors["cyan"],
            foreground=colors["ink"],
            borderwidth=0,
            font=("Segoe UI Semibold", 10),
            padding=(16, 10),
        )
        style.map(
            "OllMCP.TButton",
            background=[("active", colors["amber"]), ("disabled", colors["surface"])],
            foreground=[("disabled", colors["muted"])],
        )
        self.colors = colors

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.transcript = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Segoe UI", 11),
            background=self.colors["surface"],
            foreground=self.colors["text"],
            insertbackground=self.colors["text"],
            relief=tk.FLAT,
            borderwidth=0,
            padx=18,
            pady=18,
        )
        self.transcript.tag_configure("user", foreground=self.colors["cyan"], font=("Segoe UI Semibold", 10))
        self.transcript.tag_configure("agent", foreground=self.colors["amber"], font=("Segoe UI Semibold", 10))
        self.transcript.tag_configure("body", foreground=self.colors["text"], spacing3=16)
        self.transcript.grid(row=0, column=0, sticky="nsew", padx=14, pady=(14, 8))

        prompt_frame = tk.Frame(self, background=self.colors["ink"], padx=14, pady=14)
        prompt_frame.grid(row=1, column=0, sticky="ew")
        prompt_frame.columnconfigure(0, weight=1)
        self.prompt = tk.Text(
            prompt_frame,
            height=3,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            background=self.colors["surface"],
            foreground=self.colors["text"],
            insertbackground=self.colors["text"],
            relief=tk.FLAT,
            padx=12,
            pady=10,
        )
        self.prompt.grid(row=0, column=0, sticky="ew")
        self.prompt.bind("<Control-Return>", self._submit_from_shortcut)
        self.send_button = ttk.Button(
            prompt_frame, text="Send", style="OllMCP.TButton", command=self._submit
        )
        self.send_button.grid(row=0, column=1, sticky="ns", padx=(8, 0))

    def _submit_from_shortcut(self, _event):
        self._submit()
        return "break"

    def _submit(self) -> None:
        question = self.prompt.get("1.0", tk.END).strip()
        if not question:
            return

        self.prompt.delete("1.0", tk.END)
        self.conversation.append({"role": "user", "content": question})
        self._append("You", question)
        self.send_button.configure(state=tk.DISABLED)
        threading.Thread(
            target=self._run_agent, args=(self.conversation.copy(),), daemon=True
        ).start()

    def _run_agent(self, conversation: list[Dict[str, str]]) -> None:
        async def execute() -> str:
            # Initialize the LocalFREDAgent directly with FRED API key
            agent = LocalFREDAgent(api_key=self.api_key)
            try:
                return await agent.run(conversation)
            finally:
                pass  # No close method needed for this implementation

        try:
            answer = asyncio.run(execute())
            self.after(0, self._append_answer, answer)
        except Exception as error:
            self.after(0, self._append, "Error", str(error))
        finally:
            self.after(0, self.send_button.configure, {"state": tk.NORMAL})

    def _append(self, speaker: str, text: str) -> None:
        self.transcript.configure(state=tk.NORMAL)
        tag = "user" if speaker == "You" else "agent"
        self.transcript.insert(tk.END, f"{speaker}\n", tag)
        self.transcript.insert(tk.END, f"{text}\n\n", "body")
        self.transcript.configure(state=tk.DISABLED)
        self.transcript.see(tk.END)

    def _append_answer(self, answer: str) -> None:
        self.conversation.append({"role": "assistant", "content": answer})
        self._append("Agent", answer)


def bootstrap() -> str:
    print("FRED Agent setup")
    api_key = input("Enter your FRED API key: ").strip()
    if not api_key:
        raise RuntimeError("A FRED API key is required.")
    print("\nStarting FRED Economic Data Agent...")
    return api_key


if __name__ == "__main__":
    try:
        FredAgentApp(bootstrap()).mainloop()
    except KeyboardInterrupt:
        print("\nFRED agent stopped.")
    except Exception as error:
        print(f"Unable to start FRED agent: {error}", file=sys.stderr)
        sys.exit(1)