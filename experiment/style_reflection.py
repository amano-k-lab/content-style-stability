"""Four-image CSS/SSS style-reflection 2AFC behavioral experiment."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import secrets
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Literal, Sequence
from urllib.parse import urlparse
from uuid import uuid4

from PIL import Image, ImageColor, ImageOps, UnidentifiedImageError

from content_style_stability.config.paths import get_behavior_results_root
from content_style_stability.utils.io import (
    _append_log_line,
    _build_error_record,
    _collect_env,
    _collect_git,
    _iso_timestamp,
    create_run_directory,
    repo_relative_path,
    write_json,
    write_run_json,
)

Variant = Literal["css", "sss"]
Side = Literal["left", "right"]

DEFAULT_BACKGROUND_COLOR = "#2f2f2f"
DEFAULT_DISPLAY_GAMMA_RGB = (2.136305, 2.214327, 2.201226)
DEFAULT_DISPLAY_GAMMA_WHITE = 2.185815
DEFAULT_DISPLAY_GAMMA_PROFILE = "calibrated_monitor_20260604"

TRIAL_COLUMNS = (
    "condition_id",
    "content_id",
    "style_id",
    "block",
    "content_image_path",
    "style_image_path",
    "css_image_path",
    "sss_image_path",
)
REQUIRED_TRIAL_COLUMNS = (
    "condition_id",
    "content_image_path",
    "style_image_path",
    "css_image_path",
    "sss_image_path",
)
RESPONSE_COLUMNS = (
    "participant_id",
    "session_id",
    "responded_at",
    "trial_index",
    "condition_id",
    "repetition_index",
    "expanded_trial_id",
    "block",
    "content_id",
    "style_id",
    "content_image_path",
    "style_image_path",
    "css_image_path",
    "sss_image_path",
    "left_variant",
    "right_variant",
    "left_image_path",
    "right_image_path",
    "response_side",
    "response_key",
    "response_variant",
    "chose_css",
    "chose_sss",
    "reaction_time_ms",
    "seed",
)


@dataclass(frozen=True)
class Condition:
    """One content-style pair and its two synthesis candidates."""

    condition_id: str
    content_image_path: Path
    style_image_path: Path
    css_image_path: Path
    sss_image_path: Path
    content_id: str | None = None
    style_id: str | None = None
    block: str | None = None

    def candidate_path(self, variant: Variant) -> Path:
        return self.css_image_path if variant == "css" else self.sss_image_path

    def image_paths(self) -> tuple[Path, Path, Path, Path]:
        return (
            self.content_image_path,
            self.style_image_path,
            self.css_image_path,
            self.sss_image_path,
        )


@dataclass(frozen=True)
class Presentation:
    """One expanded trial with its randomized candidate positions."""

    trial_index: int
    condition: Condition
    repetition_index: int
    left_variant: Variant
    right_variant: Variant

    @property
    def expanded_trial_id(self) -> str:
        return f"{self.condition.condition_id}__rep{self.repetition_index:02d}"

    def variant_for_side(self, side: Side) -> Variant:
        return self.left_variant if side == "left" else self.right_variant

    def path_for_side(self, side: Side) -> Path:
        return self.condition.candidate_path(self.variant_for_side(side))


@dataclass(frozen=True)
class ExperimentConfig:
    """Resolved presentation and output settings."""

    trials_csv: Path
    participant_id: str
    session_id: str
    seed: int
    mode: str
    repeat_each_pair: int
    shuffle_trials: bool
    iti_ms: int
    stimulus_duration_ms: int
    break_interval_trials: int
    stimulus_size_cm: float
    monitor_width_cm: float
    monitor_width_px: int
    background_color: str
    display_gamma_rgb: tuple[float, float, float] | None
    display_gamma_profile: str | None
    display_gamma_white: float | None
    fullscreen: bool
    window_width: int
    window_height: int
    host: str
    port: int
    open_browser: bool
    results_root: Path

    @property
    def stimulus_size_px(self) -> int:
        return round(
            self.stimulus_size_cm * self.monitor_width_px / self.monitor_width_cm
        )


def _optional(value: str | None) -> str | None:
    normalized = (value or "").strip()
    return normalized or None


def _resolve_image_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve(strict=False)


def load_conditions(csv_path: Path) -> list[Condition]:
    """Load and validate the behavioral base-condition CSV."""

    path = csv_path.expanduser().resolve(strict=False)
    if not path.is_file():
        raise FileNotFoundError(f"trial CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("trial CSV must include a header row")
        missing = [
            name for name in REQUIRED_TRIAL_COLUMNS if name not in reader.fieldnames
        ]
        if missing:
            raise ValueError(
                f"trial CSV is missing required columns: {', '.join(missing)}"
            )
        conditions: list[Condition] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            values = {
                name: (row.get(name) or "").strip() for name in REQUIRED_TRIAL_COLUMNS
            }
            blank = [name for name, value in values.items() if not value]
            if blank:
                raise ValueError(
                    f"trial CSV row {row_number} has blank fields: {', '.join(blank)}"
                )
            condition_id = values["condition_id"]
            if condition_id in seen:
                raise ValueError(f"duplicate condition_id: {condition_id}")
            seen.add(condition_id)
            conditions.append(
                Condition(
                    condition_id=condition_id,
                    content_id=_optional(row.get("content_id")),
                    style_id=_optional(row.get("style_id")),
                    block=_optional(row.get("block")),
                    content_image_path=_resolve_image_path(
                        values["content_image_path"], path.parent
                    ),
                    style_image_path=_resolve_image_path(
                        values["style_image_path"], path.parent
                    ),
                    css_image_path=_resolve_image_path(
                        values["css_image_path"], path.parent
                    ),
                    sss_image_path=_resolve_image_path(
                        values["sss_image_path"], path.parent
                    ),
                )
            )
    if not conditions:
        raise ValueError("trial CSV has no base conditions")
    validate_condition_images(conditions)
    return conditions


def validate_condition_images(conditions: Sequence[Condition]) -> None:
    """Require every unique stimulus path to be a readable image."""

    for path in sorted({path for item in conditions for path in item.image_paths()}):
        if not path.is_file():
            raise FileNotFoundError(f"stimulus image not found: {path}")
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"failed to read stimulus image: {path}") from exc


def build_presentations(
    conditions: Sequence[Condition],
    *,
    repeat_each_pair: int,
    seed: int,
    shuffle_trials: bool = True,
) -> list[Presentation]:
    """Expand trials and reproduce the legacy seeded order and side assignment."""

    if repeat_each_pair <= 0:
        raise ValueError("repeat_each_pair must be positive")
    expanded = [
        (condition, repetition)
        for condition in conditions
        for repetition in range(1, repeat_each_pair + 1)
    ]
    if shuffle_trials:
        order_rng = random.Random(seed)
        order_rng.shuffle(expanded)
    side_rng = random.Random(seed)
    result: list[Presentation] = []
    for trial_index, (condition, repetition) in enumerate(expanded, start=1):
        left, right = ("css", "sss") if side_rng.getrandbits(1) == 0 else ("sss", "css")
        result.append(
            Presentation(
                trial_index=trial_index,
                condition=condition,
                repetition_index=repetition,
                left_variant=left,
                right_variant=right,
            )
        )
    return result


def build_response_row(
    presentation: Presentation,
    *,
    side: Side,
    key: str,
    reaction_time_ms: int,
    config: ExperimentConfig,
) -> dict[str, object]:
    """Create one durable response record from a randomized presentation."""

    if reaction_time_ms < 0:
        raise ValueError("reaction_time_ms must be non-negative")
    condition = presentation.condition
    variant = presentation.variant_for_side(side)
    return {
        "participant_id": config.participant_id,
        "session_id": config.session_id,
        "responded_at": _iso_timestamp(datetime.now(UTC)),
        "trial_index": presentation.trial_index,
        "condition_id": condition.condition_id,
        "repetition_index": presentation.repetition_index,
        "expanded_trial_id": presentation.expanded_trial_id,
        "block": condition.block or "",
        "content_id": condition.content_id or "",
        "style_id": condition.style_id or "",
        "content_image_path": str(condition.content_image_path),
        "style_image_path": str(condition.style_image_path),
        "css_image_path": str(condition.css_image_path),
        "sss_image_path": str(condition.sss_image_path),
        "left_variant": presentation.left_variant,
        "right_variant": presentation.right_variant,
        "left_image_path": str(presentation.path_for_side("left")),
        "right_image_path": str(presentation.path_for_side("right")),
        "response_side": side,
        "response_key": key,
        "response_variant": variant,
        "chose_css": variant == "css",
        "chose_sss": variant == "sss",
        "reaction_time_ms": reaction_time_ms,
        "seed": config.seed,
    }


class ResponseWriter:
    """Append and fsync each response to CSV and JSONL."""

    def __init__(self, csv_path: Path, jsonl_path: Path) -> None:
        self.csv_path = csv_path
        self.jsonl_path = jsonl_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS).writeheader()
            handle.flush()
            os.fsync(handle.fileno())
        with jsonl_path.open("w", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())

    def append(self, row: dict[str, object]) -> None:
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS).writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def prepare_display_image(path: Path, config: ExperimentConfig) -> Image.Image:
    """Resize, pad, and optionally inverse-gamma-correct one stimulus."""

    size = config.stimulus_size_px
    with Image.open(path) as source:
        contained = ImageOps.contain(
            source.convert("RGB"),
            (size, size),
            method=Image.Resampling.LANCZOS,
        )
    canvas = Image.new("RGB", (size, size), config.background_color)
    canvas.paste(
        contained, ((size - contained.width) // 2, (size - contained.height) // 2)
    )
    if config.display_gamma_rgb is None:
        return canvas
    table: list[int] = []
    for gamma in config.display_gamma_rgb:
        table.extend(
            round(((value / 255.0) ** (1.0 / gamma)) * 255.0) for value in range(256)
        )
    return canvas.point(table)


def _run_gui(
    config: ExperimentConfig,
    presentations: Sequence[Presentation],
    record: Callable[[Presentation, Side, str, int], None],
) -> bool:
    import tkinter as tk  # noqa: PLC0415

    from PIL import ImageTk  # noqa: PLC0415

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError(
            "failed to initialize Tk GUI; use browser or validate mode"
        ) from exc
    root.configure(bg=config.background_color)
    root.title("画像比較実験")
    if config.fullscreen:
        root.attributes("-fullscreen", True)
    else:
        root.geometry(f"{config.window_width}x{config.window_height}")
    response_var = tk.StringVar(value="")
    pressed: set[str] = set()
    image_cache: dict[Path, ImageTk.PhotoImage] = {}
    image_refs: list[ImageTk.PhotoImage] = []
    interrupted = False

    def clear() -> None:
        for widget in root.winfo_children():
            widget.destroy()
        image_refs.clear()

    def photo(path: Path) -> ImageTk.PhotoImage:
        resolved = path.resolve(strict=False)
        if resolved not in image_cache:
            image_cache[resolved] = ImageTk.PhotoImage(
                prepare_display_image(resolved, config)
            )
        return image_cache[resolved]

    for path in sorted(
        {
            path.resolve(strict=False)
            for presentation in presentations
            for path in (
                presentation.path_for_side("left"),
                presentation.path_for_side("right"),
                presentation.condition.content_image_path,
                presentation.condition.style_image_path,
            )
        }
    ):
        photo(path)
    root.update_idletasks()

    def signal(event: object) -> None:
        key = str(getattr(event, "keysym", "")).lower()
        if key in pressed:
            return
        pressed.add(key)
        mapping = {
            "f": "left",
            "4": "left",
            "left": "left",
            "j": "right",
            "6": "right",
            "right": "right",
        }
        if key == "escape":
            response_var.set("escape")
        elif key in mapping:
            response_var.set(f"{mapping[key]}:{key}")
        elif key in {"space", "return"}:
            response_var.set("continue")

    def release(event: object) -> None:
        pressed.discard(str(getattr(event, "keysym", "")).lower())

    def message(text: str) -> str:
        clear()
        tk.Label(
            root,
            text=text,
            bg=config.background_color,
            fg="white",
            font=("TkDefaultFont", 18),
            justify="center",
        ).pack(expand=True)
        response_var.set("")
        root.wait_variable(response_var)
        return response_var.get()

    root.bind("<KeyPress>", signal)
    root.bind("<KeyRelease>", release)
    try:
        if (
            message("左右の画像のうち、下の参照画像のスタイルを\nより反映している方を選んでください。\n\nSpace / Enterで開始")
            == "escape"
        ):
            return True
        total = len(presentations)
        for completed, presentation in enumerate(presentations):
            if (
                completed
                and config.break_interval_trials > 0
                and completed % config.break_interval_trials == 0
            ):
                if (
                    message(f"休憩してください。\n\n{completed} / {total}\n\nSpace / Enterで再開")
                    == "escape"
                ):
                    interrupted = True
                    break
            clear()
            frame = tk.Frame(root, bg=config.background_color)
            frame.pack(expand=True)
            paths = (
                presentation.path_for_side("left"),
                presentation.condition.content_image_path,
                presentation.condition.style_image_path,
                presentation.path_for_side("right"),
            )
            photos = [photo(path) for path in paths]
            image_refs.extend(photos)
            tk.Label(frame, image=photos[0], bg=config.background_color).grid(
                row=0, column=0, rowspan=2, padx=18
            )
            tk.Label(frame, image=photos[1], bg=config.background_color).grid(
                row=0, column=1, pady=8
            )
            tk.Label(frame, image=photos[2], bg=config.background_color).grid(
                row=1, column=1, pady=8
            )
            tk.Label(frame, image=photos[3], bg=config.background_color).grid(
                row=0, column=2, rowspan=2, padx=18
            )
            response_var.set("")
            onset = time.perf_counter()

            def hide() -> None:
                if not response_var.get():
                    clear()

            timeout = (
                root.after(config.stimulus_duration_ms, hide)
                if config.stimulus_duration_ms > 0
                else None
            )
            root.wait_variable(response_var)
            if timeout is not None:
                root.after_cancel(timeout)
            response = response_var.get()
            if response == "escape":
                interrupted = True
                break
            side_text, key = response.split(":", 1)
            reaction_time_ms = round((time.perf_counter() - onset) * 1000)
            record(presentation, side_text, key, reaction_time_ms)  # type: ignore[arg-type]
            while pressed.intersection({"f", "4", "left", "j", "6", "right"}):
                root.update()
                time.sleep(0.005)
            clear()
            tk.Label(
                root,
                text="+",
                bg=config.background_color,
                fg="white",
                font=("TkDefaultFont", 30),
            ).pack(expand=True)
            root.update()
            root.after(config.iti_ms)
        if not interrupted:
            message("実験は終了です。\n\nSpace / Enterで終了")
        return interrupted
    finally:
        root.destroy()


def _browser_html(
    config: ExperimentConfig, presentations: Sequence[Presentation]
) -> bytes:
    trials = [
        {
            "trial_index": item.trial_index,
            "left": f"/image/{item.trial_index}/left",
            "right": f"/image/{item.trial_index}/right",
            "content": f"/image/{item.trial_index}/content",
            "style": f"/image/{item.trial_index}/style",
        }
        for item in presentations
    ]
    payload = json.dumps(trials, ensure_ascii=False)
    stimulus_timeout_script = (
        f"setTimeout(()=>{{if(accept)grid.style.display='none';}},"
        f"{config.stimulus_duration_ms});"
        if config.stimulus_duration_ms > 0
        else ""
    )
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>画像比較実験</title>
<style>html,body{{margin:0;width:100%;height:100%;background:{config.background_color};color:white;font-family:sans-serif}}#stage{{height:100%;display:flex;align-items:center;justify-content:center;text-align:center}}#grid{{display:none;grid-template-columns:1fr auto 1fr;align-items:center;gap:24px}}.center{{display:grid;gap:16px}}img{{width:{config.stimulus_size_px}px;height:{config.stimulus_size_px}px}}button{{font-size:20px;padding:12px 28px}}</style></head>
<body><div id='stage'><div id='message'>左右の画像のうち、下の参照画像のスタイルを<br>より反映している方を選んでください。<br><br><button id='start'>開始</button></div><div id='grid'><img id='left'><div class='center'><img id='content'><img id='style'></div><img id='right'></div></div>
<script>const trials={payload};let i=0,onset=0,accept=false;const stage=document.getElementById('stage'),msg=document.getElementById('message'),grid=document.getElementById('grid');const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function next(){{if(i>=trials.length){{grid.style.display='none';msg.style.display='block';msg.innerHTML='実験は終了です。';return;}}if(i>0&&i%{config.break_interval_trials}===0){{grid.style.display='none';msg.style.display='block';msg.innerHTML=`休憩してください。<br><br>${{i}} / ${{trials.length}}<br><br><button id="resume">再開</button>`;await new Promise(r=>document.getElementById('resume').onclick=r);}}const t=trials[i];for(const id of ['left','right','content','style'])document.getElementById(id).src=t[id];msg.style.display='none';grid.style.display='grid';onset=performance.now();accept=true;{stimulus_timeout_script}}}
async function answer(side,key){{if(!accept)return;accept=false;const rt=Math.round(performance.now()-onset);await fetch('/response',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{trial_index:trials[i].trial_index,response_side:side,response_key:key,reaction_time_ms:rt}})}});i++;grid.style.display='none';msg.style.display='block';msg.textContent='+';await sleep({config.iti_ms});next();}}
document.getElementById('start').onclick=async()=>{{try{{await document.documentElement.requestFullscreen();}}catch(e){{}}next();}};const held=new Set();window.addEventListener('keydown',e=>{{const k=e.key.toLowerCase();if(held.has(k))return;held.add(k);if(['f','4','arrowleft'].includes(k))answer('left',k);if(['j','6','arrowright'].includes(k))answer('right',k);if(k==='escape')fetch('/interrupt',{{method:'POST'}});}});window.addEventListener('keyup',e=>held.delete(e.key.toLowerCase()));</script></body></html>"""
    return html.encode("utf-8")


def _run_browser(
    config: ExperimentConfig,
    presentations: Sequence[Presentation],
    record: Callable[[Presentation, Side, str, int], None],
) -> bool:
    completed = 0
    interrupted = False
    html = _browser_html(config, presentations)
    image_cache: dict[Path, bytes] = {}
    for presentation in presentations:
        for path in presentation.condition.image_paths():
            resolved = path.resolve(strict=False)
            if resolved not in image_cache:
                buffer = io.BytesIO()
                prepare_display_image(resolved, config).save(buffer, format="PNG")
                image_cache[resolved] = buffer.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html)
                return
            parts = parsed.path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "image":
                try:
                    trial_index = int(parts[1])
                    role = parts[2]
                    presentation = presentations[trial_index - 1]
                    paths = {
                        "left": presentation.path_for_side("left"),
                        "right": presentation.path_for_side("right"),
                        "content": presentation.condition.content_image_path,
                        "style": presentation.condition.style_image_path,
                    }
                    if role not in paths:
                        raise KeyError(role)
                    key = paths[role].resolve(strict=False)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.end_headers()
                    self.wfile.write(image_cache[key])
                    return
                except (IndexError, KeyError, ValueError):
                    pass
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            nonlocal completed, interrupted
            if self.path == "/interrupt":
                interrupted = True
                self.send_response(204)
                self.end_headers()
                threading.Thread(target=server.shutdown, daemon=True).start()
                return
            if self.path != "/response":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                data = json.loads(self.rfile.read(length).decode("utf-8"))
                presentation = presentations[completed]
                if int(data["trial_index"]) != presentation.trial_index:
                    raise ValueError("unexpected trial_index")
                side = str(data["response_side"])
                if side not in {"left", "right"}:
                    raise ValueError("invalid response_side")
                rt = int(data["reaction_time_ms"])
                record(presentation, side, str(data["response_key"]), rt)  # type: ignore[arg-type]
                completed += 1
                self.send_response(204)
                self.end_headers()
                if completed == len(presentations):
                    threading.Thread(target=server.shutdown, daemon=True).start()
            except (
                IndexError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                self.send_error(400, str(exc))

    server = ThreadingHTTPServer((config.host, config.port), Handler)
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    print(url)
    if config.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        server.server_close()
    return interrupted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the CSS/SSS style-reflection 2AFC experiment."
    )
    parser.add_argument("--trials-csv", required=True, type=Path)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=("gui", "browser", "validate"), default="gui")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--repeat-each-pair", type=int, default=10)
    shuffle = parser.add_mutually_exclusive_group()
    shuffle.add_argument(
        "--shuffle-trials", dest="shuffle_trials", action="store_true", default=True
    )
    shuffle.add_argument(
        "--no-shuffle-trials", dest="shuffle_trials", action="store_false"
    )
    parser.add_argument("--iti-ms", type=int, default=500)
    parser.add_argument(
        "--fixation-duration-ms", type=int, dest="iti_ms", help=argparse.SUPPRESS
    )
    parser.add_argument("--stimulus-duration-ms", type=int, default=1000)
    parser.add_argument("--break-interval-trials", type=int, default=100)
    parser.add_argument("--stimulus-size-cm", type=float, default=5.0)
    parser.add_argument("--monitor-width-cm", type=float, default=54.4)
    parser.add_argument("--monitor-width-px", type=int, default=1920)
    parser.add_argument("--background-color", default=DEFAULT_BACKGROUND_COLOR)
    gamma = parser.add_mutually_exclusive_group()
    gamma.add_argument("--display-gamma-rgb", nargs=3, type=float)
    gamma.add_argument("--no-display-gamma", action="store_true")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--window-width", type=int, default=1600)
    parser.add_argument("--window-height", type=int, default=900)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-open-browser", action="store_true")
    parser.add_argument("--results-root", type=Path)
    return parser


def _resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    mode = "validate" if args.validate_only else args.mode
    positive = {
        "repeat_each_pair": args.repeat_each_pair,
        "stimulus_size_cm": args.stimulus_size_cm,
        "monitor_width_cm": args.monitor_width_cm,
        "monitor_width_px": args.monitor_width_px,
    }
    invalid = [
        name
        for name, value in positive.items()
        if not math.isfinite(float(value)) or value <= 0
    ]
    if invalid:
        raise ValueError(f"positive values required for: {', '.join(invalid)}")
    if args.iti_ms < 0 or args.stimulus_duration_ms < 0:
        raise ValueError("iti_ms and stimulus_duration_ms must be non-negative")
    if args.break_interval_trials < 0:
        raise ValueError("break_interval_trials must be non-negative")
    if not 0 <= args.port <= 65535:
        raise ValueError("port must be in [0, 65535]")
    if not args.participant_id.strip() or not args.session_id.strip():
        raise ValueError("participant_id and session_id must not be blank")
    try:
        ImageColor.getrgb(args.background_color)
    except ValueError as exc:
        raise ValueError(f"invalid background color: {args.background_color}") from exc
    gamma = (
        None
        if args.no_display_gamma
        else tuple(args.display_gamma_rgb or DEFAULT_DISPLAY_GAMMA_RGB)
    )
    if gamma is not None and any(
        not math.isfinite(value) or value <= 0 for value in gamma
    ):
        raise ValueError("display gamma values must be positive")
    return ExperimentConfig(
        trials_csv=args.trials_csv.expanduser().resolve(strict=False),
        participant_id=args.participant_id.strip(),
        session_id=args.session_id.strip(),
        seed=args.seed if args.seed is not None else secrets.randbits(32),
        mode=mode,
        repeat_each_pair=args.repeat_each_pair,
        shuffle_trials=args.shuffle_trials,
        iti_ms=args.iti_ms,
        stimulus_duration_ms=args.stimulus_duration_ms,
        break_interval_trials=args.break_interval_trials,
        stimulus_size_cm=args.stimulus_size_cm,
        monitor_width_cm=args.monitor_width_cm,
        monitor_width_px=args.monitor_width_px,
        background_color=args.background_color,
        display_gamma_rgb=gamma,  # type: ignore[arg-type]
        display_gamma_profile=(
            None
            if gamma is None
            else "manual_cli"
            if args.display_gamma_rgb
            else DEFAULT_DISPLAY_GAMMA_PROFILE
        ),
        display_gamma_white=(
            DEFAULT_DISPLAY_GAMMA_WHITE
            if gamma is not None and not args.display_gamma_rgb
            else None
        ),
        fullscreen=not args.windowed,
        window_width=args.window_width,
        window_height=args.window_height,
        host=args.host,
        port=args.port,
        open_browser=not args.no_open_browser,
        results_root=args.results_root or get_behavior_results_root(),
    )


def _config_record(config: ExperimentConfig) -> dict[str, object]:
    return {
        "mode": config.mode,
        "participant_id": config.participant_id,
        "session_id": config.session_id,
        "seed": config.seed,
        "repeat_each_pair": config.repeat_each_pair,
        "shuffle_trials": config.shuffle_trials,
        "iti_ms": config.iti_ms,
        "stimulus_duration_ms": config.stimulus_duration_ms,
        "break_interval_trials": config.break_interval_trials,
        "stimulus_size_cm": config.stimulus_size_cm,
        "stimulus_size_px": config.stimulus_size_px,
        "monitor_width_cm": config.monitor_width_cm,
        "monitor_width_px": config.monitor_width_px,
        "background_color": config.background_color,
        "display_gamma": {
            "enabled": config.display_gamma_rgb is not None,
            "profile_name": config.display_gamma_profile,
            "gamma_rgb": config.display_gamma_rgb,
            "white_gamma": config.display_gamma_white,
            "model": "inverse_power_rgb",
            "application_stage": "after_resize_before_presentation",
        },
        "fullscreen": config.fullscreen,
        "window_width": config.window_width,
        "window_height": config.window_height,
        "results_root": str(config.results_root),
    }


def _summary_record(
    *,
    run_id: str,
    config: ExperimentConfig,
    status: str,
    base_condition_count: int,
    total_trials: int,
    completed_trials: int,
    interrupted: bool,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "task": "behavior_experiment",
        "mode": config.mode,
        "status": status,
        "participant_id": config.participant_id,
        "session_id": config.session_id,
        "seed": config.seed,
        "repeat_each_pair": config.repeat_each_pair,
        "shuffle_trials": config.shuffle_trials,
        "base_condition_count": base_condition_count,
        "total_trials": total_trials,
        "completed_trials": completed_trials,
        "interrupted": interrupted,
        "aborted": interrupted or status != "success",
    }


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    try:
        args = build_parser().parse_args(argv_list)
        config = _resolve_config(args)
    except (SystemExit, ValueError) as exc:
        if isinstance(exc, ValueError):
            print(str(exc), file=sys.stderr)
            return 2
        return int(exc.code)
    started = datetime.now(UTC)
    run_id = uuid4().hex[:8]
    run_dir = create_run_directory(
        "behavior_experiment",
        results_root=config.results_root,
        run_id=run_id,
        timestamp=started,
    )
    writer = ResponseWriter(run_dir / "responses.csv", run_dir / "responses.jsonl")
    log_path = run_dir / "run.log"
    _append_log_line(
        log_path, "INFO", f"starting behavior experiment mode={config.mode}"
    )
    response_rows: list[dict[str, object]] = []
    status = "success"
    errors: list[dict[str, object]] = []
    total_trials = 0
    base_conditions = 0
    interrupted = False
    config_record = _config_record(config)
    git_record = _collect_git()
    env_record = _collect_env()
    output_paths = [
        ("responses_csv", run_dir / "responses.csv"),
        ("responses_jsonl", run_dir / "responses.jsonl"),
        ("presentation_plan", run_dir / "presentation_plan.json"),
        ("resolved_config", run_dir / "resolved_config.json"),
        ("summary", run_dir / "summary.json"),
        ("run_log", log_path),
    ]

    def persist_run(
        current_status: str,
        *,
        ended_at: datetime | None,
        current_errors: Sequence[dict[str, object]],
    ) -> None:
        write_run_json(
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "task": "behavior_experiment",
                "status": current_status,
                "started_at": _iso_timestamp(started),
                "ended_at": _iso_timestamp(ended_at) if ended_at else None,
                "duration_sec": (
                    (ended_at - started).total_seconds() if ended_at else None
                ),
                "command": {
                    "entrypoint": "experiment.run_style_reflection_experiment",
                    "argv": argv_list,
                    "cwd": str(Path.cwd()),
                },
                "git": git_record,
                "config": config_record,
                "env": env_record,
                "inputs": [
                    {
                        "role": "behavior_trials",
                        "path": str(config.trials_csv),
                        "exists": config.trials_csv.is_file(),
                        "format": "csv",
                    }
                ],
                "outputs": [
                    {
                        "role": role,
                        "path": repo_relative_path(path),
                        "format": path.suffix.lstrip("."),
                        "exists": path.is_file(),
                        "primary": role == "responses_csv",
                        "digest": None,
                    }
                    for role, path in output_paths
                ],
                "metrics": {
                    "base_condition_count": base_conditions,
                    "total_trials": total_trials,
                    "completed_trials": len(response_rows),
                    "interrupted": interrupted,
                },
                "errors": list(current_errors),
            },
            run_dir / "run.json",
            overwrite=True,
        )

    write_json(run_dir / "resolved_config.json", config_record, overwrite=True)
    write_json(
        run_dir / "summary.json",
        _summary_record(
            run_id=run_id,
            config=config,
            status="partial",
            base_condition_count=0,
            total_trials=0,
            completed_trials=0,
            interrupted=False,
        ),
        overwrite=True,
    )
    try:
        conditions = load_conditions(config.trials_csv)
        base_conditions = len(conditions)
        presentations = build_presentations(
            conditions,
            repeat_each_pair=config.repeat_each_pair,
            seed=config.seed,
            shuffle_trials=config.shuffle_trials,
        )
        total_trials = len(presentations)
        config_record.update(
            {
                "base_condition_count": base_conditions,
                "trial_count": total_trials,
            }
        )
        write_json(run_dir / "resolved_config.json", config_record, overwrite=True)
        write_json(
            run_dir / "presentation_plan.json",
            [
                {
                    "trial_index": item.trial_index,
                    "condition_id": item.condition.condition_id,
                    "repetition_index": item.repetition_index,
                    "expanded_trial_id": item.expanded_trial_id,
                    "block": item.condition.block,
                    "content_id": item.condition.content_id,
                    "style_id": item.condition.style_id,
                    "left_variant": item.left_variant,
                    "right_variant": item.right_variant,
                }
                for item in presentations
            ],
            overwrite=True,
        )
        persist_run("partial", ended_at=None, current_errors=[])

        def record(presentation: Presentation, side: Side, key: str, rt: int) -> None:
            row = build_response_row(
                presentation,
                side=side,
                key=key,
                reaction_time_ms=rt,
                config=config,
            )
            writer.append(row)
            response_rows.append(row)
            write_json(
                run_dir / "summary.json",
                _summary_record(
                    run_id=run_id,
                    config=config,
                    status="partial",
                    base_condition_count=base_conditions,
                    total_trials=total_trials,
                    completed_trials=len(response_rows),
                    interrupted=False,
                ),
                overwrite=True,
            )
            persist_run("partial", ended_at=None, current_errors=[])

        if config.mode == "gui":
            interrupted = _run_gui(config, presentations, record)
        elif config.mode == "browser":
            interrupted = _run_browser(config, presentations, record)
        if interrupted:
            status = "partial"
    except KeyboardInterrupt as exc:
        status = "partial"
        interrupted = True
        errors.append(
            _build_error_record(
                exc,
                component="experiment.style_reflection",
                operation="run_experiment",
                recoverable=True,
            )
        )
        _append_log_line(log_path, "ERROR", "experiment interrupted by user")
    except Exception as exc:
        status = "failed"
        errors.append(
            _build_error_record(
                exc,
                component="experiment.style_reflection",
                operation="run_experiment",
                recoverable=False,
            )
        )
        _append_log_line(log_path, "ERROR", str(exc))
        print(str(exc), file=sys.stderr)
    ended = datetime.now(UTC)
    _append_log_line(
        log_path,
        "INFO",
        f"finished status={status} completed={len(response_rows)}/{total_trials}",
    )
    write_json(
        run_dir / "summary.json",
        _summary_record(
            run_id=run_id,
            config=config,
            status=status,
            base_condition_count=base_conditions,
            total_trials=total_trials,
            completed_trials=len(response_rows),
            interrupted=interrupted,
        ),
        overwrite=True,
    )
    if errors:
        write_json(run_dir / "error.json", {"errors": errors}, overwrite=True)
    persist_run(status, ended_at=ended, current_errors=errors)
    if status == "failed":
        return 1
    print(run_dir)
    return 0
