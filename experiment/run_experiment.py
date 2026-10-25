from __future__ import annotations

import argparse
import csv
import json
import os
import random
import secrets
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageTk, UnidentifiedImageError

RESULTS_ROOT = Path(__file__).resolve().parent / "results"
RESPONSE_COLUMNS = (
    "participant_id",
    "session_id",
    "responded_at",
    "trial_index",
    "trial_id",
    "block",
    "target_path",
    "option_a_path",
    "option_b_path",
    "left_option",
    "right_option",
    "left_image_path",
    "right_image_path",
    "left_label",
    "right_label",
    "response_side",
    "response_key",
    "response_option",
    "response_label",
    "reaction_time_ms",
    "correct_option",
    "is_correct",
    "seed",
)


@dataclass(frozen=True)
class Trial:
    trial_id: str
    target_path: Path
    option_a_path: Path
    option_b_path: Path
    option_a_label: str
    option_b_label: str
    correct_option: str | None
    block: str | None

    def option_path(self, option: str) -> Path:
        return self.option_a_path if option == "a" else self.option_b_path

    def option_label(self, option: str) -> str:
        return self.option_a_label if option == "a" else self.option_b_label


@dataclass(frozen=True)
class Presentation:
    trial_index: int
    trial: Trial
    left_option: str
    right_option: str


@dataclass(frozen=True)
class Response:
    side: str
    key: str
    reaction_time_ms: int
    responded_at: datetime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2AFC画像類似度判断実験")
    parser.add_argument("--trials-csv", required=True, type=Path)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="CSVと刺激画像を検証し、GUIを起動せず終了する",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="デバッグ用のウィンドウ表示で起動する",
    )
    return parser


def load_trials(csv_path: Path) -> list[Trial]:
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"trial CSVが見つかりません: {csv_path}")

    required = {
        "trial_id",
        "target_path",
        "option_a_path",
        "option_b_path",
        "option_a_label",
        "option_b_label",
    }
    trials: list[Trial] = []
    seen_ids: set[str] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("trial CSVにヘッダーがありません")
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"trial CSVの必須列がありません: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            values = {name: _required_value(row, name, row_number) for name in required}
            trial_id = values["trial_id"]
            if trial_id in seen_ids:
                raise ValueError(f"trial_idが重複しています: {trial_id}")
            seen_ids.add(trial_id)
            correct_option = _optional_value(row.get("correct_option"))
            if correct_option is not None:
                correct_option = correct_option.lower()
                if correct_option not in {"a", "b"}:
                    raise ValueError(f"{row_number}行目のcorrect_optionはa、b、空欄のいずれかです")
            trials.append(
                Trial(
                    trial_id=trial_id,
                    target_path=_resolve_image_path(
                        values["target_path"], csv_path.parent
                    ),
                    option_a_path=_resolve_image_path(
                        values["option_a_path"], csv_path.parent
                    ),
                    option_b_path=_resolve_image_path(
                        values["option_b_path"], csv_path.parent
                    ),
                    option_a_label=values["option_a_label"],
                    option_b_label=values["option_b_label"],
                    correct_option=correct_option,
                    block=_optional_value(row.get("block")),
                )
            )

    if not trials:
        raise ValueError("trial CSVにtrialがありません")
    validate_images(trials)
    return trials


def _required_value(row: dict[str, str | None], name: str, row_number: int) -> str:
    value = _optional_value(row.get(name))
    if value is None:
        raise ValueError(f"{row_number}行目の{name}が空です")
    return value


def _optional_value(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _resolve_image_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def validate_images(trials: Sequence[Trial]) -> None:
    paths = {
        path
        for trial in trials
        for path in (trial.target_path, trial.option_a_path, trial.option_b_path)
    }
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(f"刺激画像が見つかりません: {path}")
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"刺激画像を読み込めません: {path}") from exc


def build_presentations(trials: Sequence[Trial], seed: int) -> list[Presentation]:
    rng = random.Random(seed)
    ordered = list(trials)
    rng.shuffle(ordered)
    presentations: list[Presentation] = []
    for index, trial in enumerate(ordered, start=1):
        left, right = ("a", "b") if rng.getrandbits(1) == 0 else ("b", "a")
        presentations.append(Presentation(index, trial, left, right))
    return presentations


def build_response_row(
    presentation: Presentation,
    response: Response,
    *,
    participant_id: str,
    session_id: str,
    seed: int,
) -> dict[str, object]:
    trial = presentation.trial
    response_option = (
        presentation.left_option
        if response.side == "left"
        else presentation.right_option
    )
    is_correct: bool | None = None
    if trial.correct_option is not None:
        is_correct = response_option == trial.correct_option
    return {
        "participant_id": participant_id,
        "session_id": session_id,
        "responded_at": response.responded_at.isoformat(),
        "trial_index": presentation.trial_index,
        "trial_id": trial.trial_id,
        "block": trial.block or "",
        "target_path": str(trial.target_path),
        "option_a_path": str(trial.option_a_path),
        "option_b_path": str(trial.option_b_path),
        "left_option": presentation.left_option,
        "right_option": presentation.right_option,
        "left_image_path": str(trial.option_path(presentation.left_option)),
        "right_image_path": str(trial.option_path(presentation.right_option)),
        "left_label": trial.option_label(presentation.left_option),
        "right_label": trial.option_label(presentation.right_option),
        "response_side": response.side,
        "response_key": response.key,
        "response_option": response_option,
        "response_label": trial.option_label(response_option),
        "reaction_time_ms": response.reaction_time_ms,
        "correct_option": trial.correct_option or "",
        "is_correct": "" if is_correct is None else is_correct,
        "seed": seed,
    }


class ResponseWriter:
    def __init__(self, path: Path) -> None:
        self._handle = path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=RESPONSE_COLUMNS)
        self._writer.writeheader()
        self._sync()

    def append(self, row: dict[str, object]) -> None:
        self._writer.writerow(row)
        self._sync()

    def close(self) -> None:
        self._handle.close()

    def _sync(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())


class TwoAFCWindow:
    def __init__(self, *, windowed: bool) -> None:
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            raise RuntimeError(
                "tkinter GUIを開始できません。画面のある環境で実行するか、" "--validate-onlyで刺激を検証してください。"
            ) from exc
        self.root.title("2AFC Similarity Experiment")
        self.root.configure(bg="#2f2f2f")
        self.root.protocol("WM_DELETE_WINDOW", lambda: self._signal("escape"))
        self.root.bind("<Key>", self._on_key)
        if windowed:
            self.root.geometry("1600x900")
            self.width, self.height = 1600, 900
        else:
            self.root.attributes("-fullscreen", True)
            self.width = self.root.winfo_screenwidth()
            self.height = self.root.winfo_screenheight()
        self._mode = "idle"
        self._wait_var: tk.StringVar | None = None
        self._response: Response | None = None
        self._started_at = 0.0
        self._image_refs: list[ImageTk.PhotoImage] = []
        self._build_layout()

    def _build_layout(self) -> None:
        background = "#2f2f2f"
        self.container = tk.Frame(self.root, bg=background)
        self.container.pack(fill="both", expand=True)
        self.title = tk.Label(
            self.container,
            bg=background,
            fg="white",
            font=("TkDefaultFont", 18, "bold"),
        )
        self.title.pack(pady=(24, 12))
        self.target = tk.Label(self.container, bg=background)
        self.target.pack(expand=True, pady=(0, 12))
        self.message = tk.Label(
            self.container,
            bg=background,
            fg="white",
            font=("TkDefaultFont", 14),
            justify="center",
            wraplength=int(self.width * 0.8),
        )
        self.message.pack(pady=(0, 12))
        choices = tk.Frame(self.container, bg=background)
        choices.pack(fill="both", expand=True, padx=40, pady=(0, 20))
        self.left = tk.Label(choices, bg=background)
        self.left.pack(side="left", fill="both", expand=True, padx=20)
        self.right = tk.Label(choices, bg=background)
        self.right.pack(side="right", fill="both", expand=True, padx=20)
        self.footer = tk.Label(
            self.container,
            bg=background,
            fg="#dddddd",
            font=("TkDefaultFont", 12),
        )
        self.footer.pack(pady=(0, 20))

    def wait_for_start(self) -> None:
        self._mode = "start"
        self._clear_images()
        self.title.configure(text="画像比較実験")
        self.message.configure(
            text=("上のtarget画像に、より似ている候補を選んでください。\n\n" "左: F または ←　　右: J または →")
        )
        self.footer.configure(text="Space / Enterで開始　　Escで終了")
        if self._wait() == "escape":
            raise KeyboardInterrupt

    def present_trial(self, presentation: Presentation) -> Response:
        self._mode = "trial"
        self._response = None
        trial = presentation.trial
        self.title.configure(text="上のtargetにより似ている方を選んでください")
        self.message.configure(text="")
        self.footer.configure(text="左: F / ←　　右: J / →　　終了: Esc")
        self._image_refs = [
            self._photo(
                trial.target_path, (int(self.width * 0.65), int(self.height * 0.32))
            ),
            self._photo(
                trial.option_path(presentation.left_option),
                (int(self.width * 0.38), int(self.height * 0.28)),
            ),
            self._photo(
                trial.option_path(presentation.right_option),
                (int(self.width * 0.38), int(self.height * 0.28)),
            ),
        ]
        self.target.configure(image=self._image_refs[0])
        self.left.configure(image=self._image_refs[1])
        self.right.configure(image=self._image_refs[2])
        self.root.update_idletasks()
        self._started_at = time.perf_counter()
        if self._wait() == "escape":
            raise KeyboardInterrupt
        if self._response is None:
            raise RuntimeError("回答が記録されていません")
        response = self._response
        self._show_iti()
        return response

    def show_end(self, completed: int) -> None:
        self._mode = "end"
        self._clear_images()
        self.title.configure(text="実験終了")
        self.message.configure(text=f"ご協力ありがとうございました。\n\n完了試行数: {completed}")
        self.footer.configure(text="Space / Enter / Escで終了")
        self._wait()

    def close(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _photo(self, path: Path, bounds: tuple[int, int]) -> ImageTk.PhotoImage:
        with Image.open(path) as image:
            prepared = image.convert("RGB")
            prepared.thumbnail(bounds, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(prepared)

    def _show_iti(self) -> None:
        self._mode = "iti"
        self._clear_images()
        self.title.configure(text="")
        self.message.configure(text="+")
        self.footer.configure(text="")
        self.root.update_idletasks()
        self._wait_var = tk.StringVar(value="")
        self.root.after(500, lambda: self._signal("done"))
        if self._wait() == "escape":
            raise KeyboardInterrupt

    def _wait(self) -> str:
        if self._wait_var is None:
            self._wait_var = tk.StringVar(value="")
        self.root.focus_force()
        self.root.wait_variable(self._wait_var)
        value = self._wait_var.get()
        self._wait_var = None
        return value

    def _signal(self, value: str) -> None:
        if self._wait_var is not None:
            self._wait_var.set(value)

    def _on_key(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key == "escape":
            self._signal("escape")
        elif self._mode in {"start", "end"} and key in {"space", "return"}:
            self._signal("continue")
        elif self._mode == "trial" and key in {"f", "j", "left", "right"}:
            side = "left" if key in {"f", "left"} else "right"
            elapsed = int(round((time.perf_counter() - self._started_at) * 1000))
            self._response = Response(side, key, max(0, elapsed), datetime.now(UTC))
            self._signal("response")

    def _clear_images(self) -> None:
        self._image_refs = []
        self.target.configure(image="")
        self.left.configure(image="")
        self.right.configure(image="")


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _run_record(
    *,
    args: argparse.Namespace,
    trials_csv: Path,
    started_at: datetime,
    status: str,
    total_trials: int,
    completed_trials: int,
    responses_path: Path,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "started_at": started_at.isoformat(),
        "finished_at": None if status == "running" else datetime.now(UTC).isoformat(),
        "participant_id": args.participant_id,
        "session_id": args.session_id,
        "seed": args.seed,
        "trials_csv": str(trials_csv),
        "windowed": args.windowed,
        "total_trials": total_trials,
        "completed_trials": completed_trials,
        "responses_csv": responses_path.name,
        "error": error,
    }


def _new_run_directory() -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = RESULTS_ROOT / f"{timestamp}_{secrets.token_hex(4)}"
    path.mkdir()
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.participant_id.strip() or not args.session_id.strip():
        parser.error("participant-idとsession-idには空でない値を指定してください")
    args.participant_id = args.participant_id.strip()
    args.session_id = args.session_id.strip()

    try:
        trials_csv = args.trials_csv.expanduser().resolve()
        trials = load_trials(trials_csv)
        presentations = build_presentations(trials, args.seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.validate_only:
        blocks = sorted({trial.block for trial in trials if trial.block is not None})
        print(
            json.dumps(
                {
                    "status": "valid",
                    "trials_csv": str(trials_csv),
                    "total_trials": len(trials),
                    "blocks": blocks,
                    "seed": args.seed,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    run_dir = _new_run_directory()
    responses_path = run_dir / "responses.csv"
    run_path = run_dir / "run.json"
    started_at = datetime.now(UTC)
    completed = 0
    window: TwoAFCWindow | None = None
    writer = ResponseWriter(responses_path)
    _write_json(
        run_path,
        _run_record(
            args=args,
            trials_csv=trials_csv,
            started_at=started_at,
            status="running",
            total_trials=len(presentations),
            completed_trials=completed,
            responses_path=responses_path,
        ),
    )

    try:
        window = TwoAFCWindow(windowed=args.windowed)
        window.wait_for_start()
        for presentation in presentations:
            response = window.present_trial(presentation)
            writer.append(
                build_response_row(
                    presentation,
                    response,
                    participant_id=args.participant_id,
                    session_id=args.session_id,
                    seed=args.seed,
                )
            )
            completed += 1
            _write_json(
                run_path,
                _run_record(
                    args=args,
                    trials_csv=trials_csv,
                    started_at=started_at,
                    status="running",
                    total_trials=len(presentations),
                    completed_trials=completed,
                    responses_path=responses_path,
                ),
            )
        _write_json(
            run_path,
            _run_record(
                args=args,
                trials_csv=trials_csv,
                started_at=started_at,
                status="completed",
                total_trials=len(presentations),
                completed_trials=completed,
                responses_path=responses_path,
            ),
        )
        window.show_end(completed)
        print(run_dir)
        return 0
    except KeyboardInterrupt:
        _write_json(
            run_path,
            _run_record(
                args=args,
                trials_csv=trials_csv,
                started_at=started_at,
                status="partial",
                total_trials=len(presentations),
                completed_trials=completed,
                responses_path=responses_path,
                error="experiment interrupted",
            ),
        )
        print(f"実験を中断しました。結果: {run_dir}", file=sys.stderr)
        return 130
    except Exception as exc:
        status = "partial" if completed else "failed"
        _write_json(
            run_path,
            _run_record(
                args=args,
                trials_csv=trials_csv,
                started_at=started_at,
                status=status,
                total_trials=len(presentations),
                completed_trials=completed,
                responses_path=responses_path,
                error=f"{type(exc).__name__}: {exc}",
            ),
        )
        print(f"error: {exc}\n結果: {run_dir}", file=sys.stderr)
        return 1
    finally:
        writer.close()
        if window is not None:
            window.close()


if __name__ == "__main__":
    raise SystemExit(main())
