"""Desktop client for an LM Studio agent using tools from the FRED API."""

import asyncio
from datetime import date, datetime
import importlib.util
import re
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Tuple


REQUIRED_PACKAGES = {
    "matplotlib": "matplotlib",
    "lmstudio": "lmstudio",
    "openai": "openai",
    "requests": "requests",
}
# Refresh the provisional token display while a local-model request is running.
PROMPT_TOKEN_REFRESH_MS = 750


def ensure_packages() -> None:
    """Install the Python dependencies when this script is started directly."""
    # The GUI can run from a fresh Python environment without a separate setup step.
    missing_packages = [
        package
        for module, package in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing_packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])


ensure_packages()

# Import GUI and chart dependencies only after the lightweight dependency check.
import tkinter as tk
from tkinter import scrolledtext, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from fred_agent import LocalFREDAgent


class FredAgentApp(tk.Tk):
    def __init__(self, api_key: str, twelve_data_api_key: str = ""):
        super().__init__()
        self.api_key = api_key
        self.twelve_data_api_key = twelve_data_api_key
        self.title("Economic and Market Data Agent")
        self.minsize(760, 560)
        self.geometry("900x680")
        # Conversation messages are the durable context sent back to the agent on each turn.
        self.conversation: list[Dict[str, str]] = []
        # Keep Tk-backed canvases alive for as long as their inline chart panels are visible.
        self.chart_canvases: List[FigureCanvasTkAgg] = []
        self.run_number = 0
        self.session_tokens = 0
        self.prompt_tokens = 0
        self.prompt_started_at = 0.0
        self.prompt_seed_tokens = 0
        self.progress_percent = 0
        self.working = False
        self.dot_phase = 0
        self._configure_theme()
        self._build_ui()

    def _configure_theme(self) -> None:
        # Keep Tk widgets and generated Matplotlib charts on the same visual palette.
        colors = {
            "ink": "#10151f",
            "surface": "#18212f",
            "user_surface": "#1c3040",
            "agent_surface": "#30291a",
            "cyan": "#5eead4",
            "amber": "#fbbf24",
            "text": "#e5edf5",
            "muted": "#a4b1c1",
            "line": "#314157",
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
        # The top-level grid reserves expandable space for the transcript.
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        conversation_frame = tk.Frame(self, background=self.colors["ink"])
        conversation_frame.grid(row=0, column=0, sticky="nsew", padx=14, pady=(14, 8))
        conversation_frame.columnconfigure(0, weight=1)
        conversation_frame.rowconfigure(1, weight=1)

        header = tk.Frame(conversation_frame, background=self.colors["ink"], pady=8)
        header.grid(row=0, column=0, sticky="ew")
        tk.Label(
            header,
            text="ECONOMIC + MARKET RUNS",
            background=self.colors["ink"],
            foreground=self.colors["cyan"],
            font=("Segoe UI Semibold", 10),
        ).pack(side=tk.LEFT)
        self.token_totals = tk.Label(
            header,
            text="Session 0 tok  |  Prompt 0 tok",
            background=self.colors["ink"],
            foreground=self.colors["muted"],
            font=("Cascadia Mono", 8),
        )
        self.token_totals.pack(side=tk.RIGHT)

        # A disabled text widget acts as the append-only run feed and hosts inline chart windows.
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
            pady=12,
        )
        self.transcript.tag_configure("run", foreground=self.colors["muted"], font=("Cascadia Mono", 9), spacing1=16, spacing3=5)
        self.transcript.tag_configure("user", foreground=self.colors["cyan"], background=self.colors["user_surface"], font=("Segoe UI Semibold", 10))
        self.transcript.tag_configure("agent", foreground=self.colors["amber"], background=self.colors["agent_surface"], font=("Segoe UI Semibold", 10))
        self.transcript.tag_configure("heading", foreground=self.colors["text"], font=("Segoe UI Semibold", 14), spacing1=12, spacing3=6)
        self.transcript.tag_configure("subheading", foreground=self.colors["cyan"], font=("Segoe UI Semibold", 11), spacing1=9, spacing3=4)
        self.transcript.tag_configure("bullet", foreground=self.colors["text"], lmargin1=22, lmargin2=36, spacing3=3)
        self.transcript.tag_configure("body", foreground=self.colors["text"], lmargin1=8, lmargin2=8, spacing3=7)
        self.transcript.tag_configure("bold", font=("Segoe UI Semibold", 11))
        self.transcript.tag_configure("code", foreground=self.colors["cyan"], font=("Cascadia Mono", 10))
        self.transcript.grid(row=1, column=0, sticky="nsew")

        # This status strip is hidden between requests and shows background-agent progress.
        self.working_frame = tk.Frame(self, background=self.colors["ink"], padx=14, pady=3)
        self.working_frame.grid(row=1, column=0, sticky="ew")
        self.working_frame.columnconfigure(2, weight=1)
        self.working_label = tk.Label(
            self.working_frame,
            text="",
            background=self.colors["ink"],
            foreground=self.colors["amber"],
            font=("Segoe UI Semibold", 9),
            anchor="w",
        )
        self.working_label.grid(row=0, column=0, sticky="w")
        dot_group = tk.Frame(self.working_frame, background=self.colors["ink"], width=32, height=20)
        dot_group.grid(row=0, column=1, sticky="w", padx=(5, 0))
        dot_group.grid_propagate(False)
        self.dot_labels = []
        for index in range(3):
            dot = tk.Label(
                dot_group,
                text=".",
                background=self.colors["ink"],
                foreground=self.colors["muted"],
                font=("Segoe UI Semibold", 12),
            )
            dot.place(x=index * 9, y=3, width=9, height=16)
            self.dot_labels.append(dot)
        self.working_detail = tk.Label(
            self.working_frame,
            text="",
            background=self.colors["ink"],
            foreground=self.colors["muted"],
            font=("Cascadia Mono", 8),
            anchor="e",
        )
        self.working_detail.grid(row=0, column=3, sticky="e")
        self.working_frame.grid_remove()

        prompt_frame = tk.Frame(self, background=self.colors["ink"], padx=14, pady=14)
        prompt_frame.grid(row=2, column=0, sticky="ew")
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
        self._start_working(question)
        self.send_button.configure(state=tk.DISABLED)
        # Run the blocking local-model workflow off the Tk event loop to keep the UI responsive.
        threading.Thread(
            target=self._run_agent, args=(self.conversation.copy(),), daemon=True
        ).start()

    def _run_agent(self, conversation: list[Dict[str, str]]) -> None:
        # Each run owns its retrieved chart data until its final response is rendered.
        charts: Dict[str, List[Dict[str, Any]]] = {}

        async def execute() -> str:
            agent = LocalFREDAgent(
                api_key=self.api_key,
                twelve_data_api_key=self.twelve_data_api_key,
                activity_callback=self._queue_activity,
                # Agent callbacks occur on this worker thread; UI callbacks re-enter through after().
                chart_callback=lambda series_id, observations: charts.__setitem__(series_id, observations),
                token_callback=self._queue_token_usage,
            )
            return await agent.run(conversation)

        try:
            answer = asyncio.run(execute())
            # Tk widgets may only be updated on the main thread.
            self.after(0, self._append_answer, answer, charts)
        except Exception as error:
            self.after(0, self._append, "Error", str(error))
        finally:
            self.after(0, self.send_button.configure, {"state": tk.NORMAL})
            self.after(0, self._stop_working)

    def _start_working(self, question: str) -> None:
        # Seed live telemetry before LM Studio has returned authoritative usage metadata.
        self.working = True
        self.prompt_tokens = 0
        self.prompt_started_at = time.monotonic()
        self.prompt_seed_tokens = max(1, (len(question) + 2) // 3)
        self.progress_percent = 5
        self.working_frame.grid()
        self._set_working_detail("Preparing the model request")
        self._animate_working()
        self._refresh_prompt_token_total()

    def _stop_working(self) -> None:
        self.working = False
        self.working_frame.grid_remove()
        self._update_token_totals()

    def _queue_activity(self, detail: str) -> None:
        # Marshal agent progress from the worker thread onto Tk's event loop.
        self.after(0, self._set_working_detail, detail)

    def _set_working_detail(self, detail: str) -> None:
        if not self.working:
            return
        self.progress_percent = max(self.progress_percent, self._progress_for_activity(detail))
        self.working_label.configure(text=detail)
        self.working_detail.configure(text=self._working_text())

    def _queue_token_usage(self, usage: Dict[str, Any]) -> None:
        self.after(0, self._record_token_usage, usage)

    def _record_token_usage(self, usage: Dict[str, Any]) -> None:
        # A tool-using answer can have several model completions, so totals are accumulated per run.
        total_tokens = int(usage["total_tokens"])
        self.prompt_tokens += total_tokens
        self.session_tokens += total_tokens
        self.progress_percent = max(self.progress_percent, 35)
        self._update_token_totals()

    def _progress_for_activity(self, detail: str) -> int:
        if detail.startswith("Retrieving data"):
            return 60
        if detail.startswith("Reviewing"):
            return 45
        if detail.startswith("Synthesizing"):
            return 85
        if detail.startswith("Context"):
            return 20
        return 10

    def _working_text(self) -> str:
        elapsed = max(0.0, time.monotonic() - self.prompt_started_at)
        live_tokens = self._live_prompt_tokens()
        progress = min(95, max(self.progress_percent, int(elapsed * 2) + 5))
        tokens_per_second = live_tokens / max(elapsed, 0.1)
        return f"{progress}% est.  |  ~{live_tokens:,} tok  |  {tokens_per_second:.1f} tok/s"

    def _animate_working(self) -> None:
        if not self.working:
            return
        self.dot_phase = (self.dot_phase + 1) % 3
        for index, dot in enumerate(self.dot_labels):
            active = index == self.dot_phase
            dot.configure(foreground=self.colors["amber"] if active else self.colors["muted"])
            dot.place_configure(y=0 if active else 3)
        self.working_detail.configure(text=self._working_text())
        # Continue the animation only while a request remains active.
        self.after(300, self._animate_working)

    def _live_prompt_tokens(self) -> int:
        elapsed = max(0.0, time.monotonic() - self.prompt_started_at)
        estimated_base = self.prompt_tokens or self.prompt_seed_tokens
        return estimated_base + int(elapsed * 4)

    def _refresh_prompt_token_total(self) -> None:
        if not self.working:
            return
        self._update_token_totals(prompt_tokens=self._live_prompt_tokens())
        self.after(PROMPT_TOKEN_REFRESH_MS, self._refresh_prompt_token_total)

    def _update_token_totals(self, prompt_tokens: int | None = None) -> None:
        displayed_prompt_tokens = self.prompt_tokens if prompt_tokens is None else prompt_tokens
        self.token_totals.configure(
            text=f"Session {self.session_tokens:,} tok  |  Prompt {displayed_prompt_tokens:,} tok"
        )

    def _append(self, speaker: str, text: str) -> None:
        # Temporarily enable the transcript because user interaction remains disabled.
        self.transcript.configure(state=tk.NORMAL)
        self.run_number += 1
        tag = "user" if speaker == "You" else "agent"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.transcript.insert(tk.END, f"RUN {self.run_number:03d}  |  {timestamp}\n", "run")
        self.transcript.insert(tk.END, f"{speaker}\n", tag)
        self._append_formatted_text(text)
        self.transcript.insert(tk.END, "\n", "body")
        self.transcript.configure(state=tk.DISABLED)
        self.transcript.see(tk.END)

    def _append_answer(self, answer: str, charts: Dict[str, List[Dict[str, Any]]]) -> None:
        # Preserve the assistant answer as the next turn's conversational context.
        self.conversation.append({"role": "assistant", "content": answer})
        self._append("Agent", answer)
        if charts:
            self._append("Data synopsis", self._build_data_synopsis(charts))
            self._append_chart(charts)

    def _build_data_synopsis(self, chart_data: Dict[str, List[Dict[str, Any]]]) -> str:
        """Summarize the plotted observations when a model response is incomplete."""
        # Convert heterogeneous FRED and Twelve Data records into one date/value representation.
        labels = {
            "UNRATE": "Unemployment rate",
            "CPIAUCSL": "Consumer price index",
            "PCEPI": "Personal consumption expenditures price index",
            "GDPC1": "Real GDP",
        }
        lines = ["## Data synopsis"]
        for series_id, observations in chart_data.items():
            points: List[Tuple[date, float]] = []
            for observation in observations:
                try:
                    points.append((date.fromisoformat(observation["date"]), float(observation["value"])))
                except (KeyError, TypeError, ValueError):
                    continue
            if len(points) < 2:
                continue
            points.sort()
            latest_date, latest_value = points[-1]
            previous_value = points[-2][1]
            change = latest_value - previous_value
            name = labels.get(series_id, series_id.replace("Market: ", "Market price: "))
            if series_id == "UNRATE":
                movement = f"{change:+.1f} percentage points from the prior release"
            else:
                percentage_change = (change / previous_value * 100) if previous_value else 0
                movement = f"{percentage_change:+.2f}% from the prior observation"
            lines.append(
                f"- **{name} (`{series_id}`):** {latest_value:,.2f} as of "
                f"{latest_date:%b %Y}, {movement}."
            )
        lines.extend(
            [
                "### Reading the chart",
                "- Each panel uses its own scale, so economic indicator levels and market prices remain readable without implying direct comparability.",
                "- The synopsis uses the newest two available observations; it is descriptive, not a forecast or causal claim.",
            ]
        )
        return "\n".join(lines)

    def _append_formatted_text(self, text: str) -> None:
        # Render the small Markdown subset emitted by the model using Tk text tags.
        for line in text.splitlines() or [""]:
            if line.startswith("### "):
                self.transcript.insert(tk.END, f"{line[4:]}\n", "subheading")
            elif line.startswith(("# ", "## ")):
                self.transcript.insert(tk.END, f"{line.lstrip('# ').strip()}\n", "heading")
            elif re.match(r"^\s*(?:[-*]|\d+\.)\s+", line):
                content = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line)
                self.transcript.insert(tk.END, "  - ", "bullet")
                self._insert_inline_formatting(content)
                self.transcript.insert(tk.END, "\n", "bullet")
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

    def _append_chart(self, chart_data: Dict[str, List[Dict[str, Any]]]) -> None:
        # Normalize raw API observations before creating one independently scaled panel per series.
        series_points: Dict[str, List[Tuple[date, float]]] = {}
        for series_id, observations in chart_data.items():
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

        # Matplotlib renders into a Tk widget so the chart can live inside the transcript feed.
        figure = Figure(
            figsize=(7.2, max(3.2, 2.3 * len(series_points))),
            dpi=100,
            facecolor=self.colors["surface"],
        )
        axes = figure.subplots(len(series_points), 1, squeeze=False).flatten()
        chart_colors = (self.colors["cyan"], self.colors["amber"], "#8be28b", "#f38ba8")
        for index, (series_id, points) in enumerate(series_points.items()):
            axis = axes[index]
            axis.set_facecolor(self.colors["surface"])
            dates, values = zip(*points)
            axis.plot(
                dates,
                values,
                color=chart_colors[index % len(chart_colors)],
                linewidth=2.4,
                marker="o",
                markersize=3,
            )
            axis.set_title(series_id, color=self.colors["text"], fontsize=10, fontweight="bold", loc="left")
            axis.set_ylabel("Value", color=self.colors["muted"], fontsize=8)
            axis.tick_params(axis="x", colors=self.colors["muted"], rotation=30, labelsize=7)
            axis.tick_params(axis="y", colors=self.colors["muted"], labelsize=7)
            axis.grid(axis="y", color=self.colors["line"], alpha=0.65, linewidth=0.7)
            for spine in axis.spines.values():
                spine.set_color(self.colors["line"])
        figure.tight_layout(pad=2)
        chart_panel = tk.Frame(self.transcript, background=self.colors["surface"], padx=12, pady=10)
        tk.Label(
            chart_panel,
            text="ARTIFACT  /  DATA SERIES",
            background=self.colors["surface"],
            foreground=self.colors["muted"],
            font=("Cascadia Mono", 9),
        ).pack(anchor=tk.W)
        chart_canvas = FigureCanvasTkAgg(figure, master=chart_panel)
        chart_canvas.draw()
        chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.chart_canvases.append(chart_canvas)

        self.transcript.configure(state=tk.NORMAL)
        self.transcript.window_create(tk.END, window=chart_panel, padx=8, pady=5)
        self.transcript.insert(tk.END, "\n\n", "body")
        self.transcript.configure(state=tk.DISABLED)
        self.transcript.see(tk.END)


def bootstrap() -> Tuple[str, str]:
    # Keep credentials out of the chat history and request the FRED key at application startup.
    print("Economic and Market Data Agent setup")
    api_key = input("Enter your FRED API key: ").strip()
    if not api_key:
        raise RuntimeError("A FRED API key is required.")
    twelve_data_api_key = input(
        "Enter your Twelve Data API key (or leave blank to use TWELVE_DATA_API_KEY): "
    ).strip()
    print("\nStarting Economic and Market Data Agent...")
    return api_key, twelve_data_api_key


if __name__ == "__main__":
    try:
        FredAgentApp(*bootstrap()).mainloop()
    except KeyboardInterrupt:
        print("\nFRED agent stopped.")
    except Exception as error:
        print(f"Unable to start FRED agent: {error}", file=sys.stderr)
        sys.exit(1)