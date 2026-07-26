"""Desktop client for an LM Studio agent using tools from the FRED API."""

import asyncio
from datetime import date
import importlib.util
import queue
import re
import subprocess
import sys
import threading
from typing import Any, Dict, List, Tuple


REQUIRED_PACKAGES = {
    "matplotlib": "matplotlib",
    "openai": "openai",
    "requests": "requests",
}


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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from fred_agent import LocalFREDAgent


class FredAgentApp(tk.Tk):
    def __init__(self, api_key: str):
        super().__init__()
        self.api_key = api_key
        self.title("FRED Economic Data Agent")
        self.minsize(760, 560)
        self.geometry("900x680")
        self.conversation: list[Dict[str, str]] = []
        self.chart_canvas: FigureCanvasTkAgg | None = None
        self.chart_data: Dict[str, List[Dict[str, Any]]] = {}
        self.chart_queue: queue.SimpleQueue[Tuple[str, List[Dict[str, Any]]]] = queue.SimpleQueue()
        self._configure_theme()
        self._build_ui()

    def _configure_theme(self) -> None:
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
            "Fred.TButton",
            background=colors["cyan"],
            foreground=colors["ink"],
            borderwidth=0,
            font=("Segoe UI Semibold", 10),
            padding=(16, 10),
        )
        style.map(
            "Fred.TButton",
            background=[("active", colors["amber"]), ("disabled", colors["surface"])],
            foreground=[("disabled", colors["muted"])],
        )
        style.configure(
            "Fred.TNotebook",
            background=colors["ink"],
            borderwidth=0,
        )
        style.configure(
            "Fred.TNotebook.Tab",
            background=colors["surface"],
            foreground=colors["muted"],
            font=("Segoe UI Semibold", 10),
            padding=(16, 9),
        )
        style.map(
            "Fred.TNotebook.Tab",
            background=[("selected", colors["ink"])],
            foreground=[("selected", colors["cyan"])],
        )
        self.colors = colors

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self, style="Fred.TNotebook")
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=14, pady=(14, 8))

        conversation_frame = tk.Frame(self, background=self.colors["ink"])
        chart_frame = tk.Frame(self, background=self.colors["ink"])
        self.notebook.add(conversation_frame, text="Conversation")
        self.notebook.add(chart_frame, text="Charts")

        self.transcript = scrolledtext.ScrolledText(
            conversation_frame,
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
        self.transcript.tag_configure("heading", foreground=self.colors["amber"], font=("Segoe UI Semibold", 13), spacing1=12, spacing3=6)
        self.transcript.tag_configure("subheading", foreground=self.colors["cyan"], font=("Segoe UI Semibold", 11), spacing1=9, spacing3=4)
        self.transcript.tag_configure("bullet", foreground=self.colors["cyan"], lmargin1=18, lmargin2=30)
        self.transcript.tag_configure("body", foreground=self.colors["text"], spacing3=7)
        self.transcript.tag_configure("bold", font=("Segoe UI Semibold", 11))
        self.transcript.tag_configure("code", foreground=self.colors["cyan"], font=("Cascadia Mono", 10))
        conversation_frame.columnconfigure(0, weight=1)
        conversation_frame.rowconfigure(0, weight=1)
        self.transcript.grid(row=0, column=0, sticky="nsew")

        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(1, weight=1)
        chart_header = tk.Frame(chart_frame, background=self.colors["ink"], padx=12, pady=10)
        chart_header.grid(row=0, column=0, sticky="ew")
        tk.Label(
            chart_header,
            text="Economic series charts",
            background=self.colors["ink"],
            foreground=self.colors["text"],
            font=("Segoe UI Semibold", 12),
        ).pack(side=tk.LEFT)
        ttk.Button(
            chart_header,
            text="Clear",
            style="Fred.TButton",
            command=self._clear_chart,
        ).pack(side=tk.RIGHT)
        self.chart_host = tk.Frame(chart_frame, background=self.colors["surface"])
        self.chart_host.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self._show_chart_placeholder()
        self.after(200, self._poll_chart_queue)

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
            prompt_frame, text="Send", style="Fred.TButton", command=self._submit
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
            agent = LocalFREDAgent(
                api_key=self.api_key,
                chart_callback=self._queue_chart,
            )
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
        self._append_formatted_text(text)
        self.transcript.insert(tk.END, "\n", "body")
        self.transcript.configure(state=tk.DISABLED)
        self.transcript.see(tk.END)

    def _append_answer(self, answer: str) -> None:
        self.conversation.append({"role": "assistant", "content": answer})
        self._append("Agent", answer)

    def _append_formatted_text(self, text: str) -> None:
        for line in text.splitlines() or [""]:
            if line.startswith("### "):
                self.transcript.insert(tk.END, f"{line[4:]}\n", "subheading")
            elif line.startswith(("# ", "## ")):
                self.transcript.insert(tk.END, f"{line.lstrip('# ').strip()}\n", "heading")
            elif re.match(r"^\s*(?:[-*]|\d+\.)\s+", line):
                content = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line)
                self.transcript.insert(tk.END, f"  - {content}\n", "bullet")
            else:
                self._insert_inline_formatting(line)
                self.transcript.insert(tk.END, "\n", "body")

    def _insert_inline_formatting(self, line: str) -> None:
        parts = re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", line)
        for part in parts:
            if part.startswith("**") and part.endswith("**"):
                self.transcript.insert(tk.END, part[2:-2], "bold")
            elif part.startswith("`") and part.endswith("`"):
                self.transcript.insert(tk.END, part[1:-1], "code")
            else:
                self.transcript.insert(tk.END, part, "body")

    def _queue_chart(self, series_id: str, observations: List[Dict[str, Any]]) -> None:
        self.chart_queue.put((series_id, observations))

    def _poll_chart_queue(self) -> None:
        received_data = False
        while True:
            try:
                series_id, observations = self.chart_queue.get_nowait()
            except queue.Empty:
                break
            self.chart_data[series_id] = observations
            received_data = True
        if received_data:
            self._render_charts()
        self.after(200, self._poll_chart_queue)

    def _show_chart_placeholder(self) -> None:
        tk.Label(
            self.chart_host,
            text="Ask for a FRED series and its observations will appear here.",
            background=self.colors["surface"],
            foreground=self.colors["muted"],
            font=("Segoe UI", 11),
        ).place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def _clear_chart(self) -> None:
        self.chart_data.clear()
        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()
            self.chart_canvas = None
        for child in self.chart_host.winfo_children():
            child.destroy()
        self._show_chart_placeholder()

    def _render_charts(self) -> None:
        series_points: Dict[str, List[Tuple[date, float]]] = {}
        for series_id, observations in self.chart_data.items():
            points: List[Tuple[date, float]] = []
            for observation in observations:
                try:
                    points.append((date.fromisoformat(observation["date"]), float(observation["value"])))
                except (KeyError, TypeError, ValueError):
                    continue
            if points:
                series_points[series_id] = sorted(points)
        if not series_points:
            return

        if self.chart_canvas:
            self.chart_canvas.get_tk_widget().destroy()
        figure = Figure(figsize=(8.5, 5), dpi=100, facecolor=self.colors["surface"])
        axis = figure.add_subplot(111, facecolor=self.colors["surface"])
        chart_colors = (self.colors["cyan"], self.colors["amber"], "#8be28b", "#f38ba8")
        for index, (series_id, points) in enumerate(series_points.items()):
            dates, values = zip(*points)
            axis.plot(
                dates,
                values,
                color=chart_colors[index % len(chart_colors)],
                linewidth=2.4,
                marker="o",
                markersize=3,
                label=series_id,
            )
        axis.set_title("FRED observations", color=self.colors["text"], fontsize=14, fontweight="bold", loc="left")
        axis.set_ylabel("Value", color=self.colors["muted"])
        axis.tick_params(axis="x", colors=self.colors["muted"], rotation=30, labelsize=8)
        axis.tick_params(axis="y", colors=self.colors["muted"])
        axis.grid(axis="y", color="#314157", alpha=0.65, linewidth=0.7)
        for spine in axis.spines.values():
            spine.set_color("#314157")
        legend = axis.legend(facecolor=self.colors["surface"], edgecolor="#314157", labelcolor=self.colors["text"])
        for label in legend.get_texts():
            label.set_color(self.colors["text"])
        figure.tight_layout(pad=2)
        self.chart_canvas = FigureCanvasTkAgg(figure, master=self.chart_host)
        self.chart_canvas.draw()
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.notebook.select(1)


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