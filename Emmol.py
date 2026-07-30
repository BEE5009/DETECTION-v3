from __future__ import annotations

import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import cv2

from console_utils import configure_console_encoding, safe_print

configure_console_encoding()

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "วีดีโอภาษามือ"
SUPPORTED_SUFFIXES = {".mkv", ".mp4", ".avi", ".mov", ".m4v", ".webm"}
MAX_PREVIEW_WIDTH = 960
MAX_PREVIEW_HEIGHT = 540
FFPLAY_PATH = shutil.which("ffplay")


def normalize_text(value: str) -> str:
    return re.sub(r"[\s\W_]+", "", value.strip(), flags=re.UNICODE)


@dataclass(frozen=True)
class SearchResult:
    query: str
    normalized_query: str
    matched_terms: list[str]
    missing_terms: list[str]
    playback_paths: list[Path]
    exact_match: bool


class VideoLibrary:
    def __init__(self, video_dir: Path):
        self.video_dir = video_dir
        self.term_to_paths: dict[str, list[Path]] = {}
        self.terms_in_search_order: list[str] = []
        self.refresh()

    def refresh(self) -> None:
        self.term_to_paths.clear()

        if not self.video_dir.exists():
            self.video_dir.mkdir(parents=True, exist_ok=True)

        for video_path in sorted(self.video_dir.rglob("*")):
            if not video_path.is_file() or video_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue

            normalized_stem = normalize_text(video_path.stem)
            if not normalized_stem:
                continue

            self.term_to_paths.setdefault(normalized_stem, []).append(video_path)

        for paths in self.term_to_paths.values():
            paths.sort()

        self.terms_in_search_order = sorted(self.term_to_paths.keys(), key=len, reverse=True)

    def all_paths(self) -> list[Path]:
        ordered_paths: list[Path] = []
        for term in sorted(self.term_to_paths.keys()):
            ordered_paths.extend(self.term_to_paths[term])
        return ordered_paths

    def segment_query(self, normalized_query: str) -> tuple[list[str], list[str]]:
        matched_terms: list[str] = []
        missing_terms: list[str] = []

        if not normalized_query:
            return matched_terms, missing_terms

        index = 0
        while index < len(normalized_query):
            current_char = normalized_query[index]
            candidates = [term for term in self.terms_in_search_order if term[0] == current_char]

            matched_term = None
            for candidate in candidates:
                if normalized_query.startswith(candidate, index):
                    matched_term = candidate
                    break

            if matched_term is not None:
                matched_terms.append(matched_term)
                index += len(matched_term)
                continue

            start = index
            index += 1
            while index < len(normalized_query):
                next_char = normalized_query[index]
                next_candidates = [term for term in self.terms_in_search_order if term[0] == next_char]
                if any(normalized_query.startswith(candidate, index) for candidate in next_candidates):
                    break
                index += 1

            missing_terms.append(normalized_query[start:index])

        return matched_terms, missing_terms

    def search(self, query: str) -> SearchResult:
        normalized_query = normalize_text(query)
        if not normalized_query:
            return SearchResult(query, normalized_query, [], [], [], False)

        if normalized_query in self.term_to_paths:
            matched_paths = list(self.term_to_paths[normalized_query])
            return SearchResult(
                query=query,
                normalized_query=normalized_query,
                matched_terms=[normalized_query],
                missing_terms=[],
                playback_paths=matched_paths,
                exact_match=True,
            )

        matched_terms, missing_terms = self.segment_query(normalized_query)
        playback_paths: list[Path] = []
        for term in matched_terms:
            playback_paths.extend(self.term_to_paths.get(term, []))

        return SearchResult(
            query=query,
            normalized_query=normalized_query,
            matched_terms=matched_terms,
            missing_terms=missing_terms,
            playback_paths=playback_paths,
            exact_match=False,
        )


class EmmolApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Emmol")
        self.root.geometry("1180x720")
        self.root.minsize(980, 620)

        self.library = VideoLibrary(VIDEO_DIR)
        self.playback_thread: threading.Thread | None = None
        self.active_process: subprocess.Popen | None = None
        self.stop_event = threading.Event()
        self.ffplay_path = FFPLAY_PATH

        self.query_var = tk.StringVar()
        self.status_var = tk.StringVar(value="พร้อมค้นหาวิดีโอ")
        self.current_video_var = tk.StringVar(value="ยังไม่ได้เลือกวิดีโอ")

        self._build_ui()
        self._refresh_library_view()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        self.root.configure(bg="#111827")

        top_frame = tk.Frame(self.root, bg="#111827")
        top_frame.pack(fill="x", padx=18, pady=(18, 10))

        title_label = tk.Label(
            top_frame,
            text="Emmol",
            font=("Tahoma", 24, "bold"),
            fg="#F9FAFB",
            bg="#111827",
        )
        title_label.pack(anchor="w")

        subtitle_label = tk.Label(
            top_frame,
            text="พิมพ์คำหรือประโยค แล้วระบบจะค้นหาชื่อวิดีโอในโฟลเดอร์ วีดีโอภาษามือ",
            font=("Tahoma", 12),
            fg="#CBD5E1",
            bg="#111827",
        )
        subtitle_label.pack(anchor="w", pady=(6, 0))

        input_frame = tk.Frame(self.root, bg="#111827")
        input_frame.pack(fill="x", padx=18, pady=(6, 10))

        entry = ttk.Entry(input_frame, textvariable=self.query_var, font=("Tahoma", 16))
        entry.pack(side="left", fill="x", expand=True, ipady=8)
        entry.bind("<Return>", lambda event: self.on_search())

        search_button = ttk.Button(input_frame, text="รัน", command=self.on_search)
        search_button.pack(side="left", padx=(10, 0))

        stop_button = ttk.Button(input_frame, text="หยุดเล่น", command=self.stop_playback)
        stop_button.pack(side="left", padx=(8, 0))

        refresh_button = ttk.Button(input_frame, text="รีเฟรชรายการ", command=self.on_refresh_library)
        refresh_button.pack(side="left", padx=(8, 0))

        body_frame = tk.Frame(self.root, bg="#111827")
        body_frame.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        left_panel = tk.Frame(body_frame, bg="#1F2937", highlightbackground="#334155", highlightthickness=1)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))

        right_panel = tk.Frame(body_frame, bg="#1F2937", highlightbackground="#334155", highlightthickness=1)
        right_panel.pack(side="right", fill="both", expand=False)
        right_panel.configure(width=360)
        right_panel.pack_propagate(False)

        result_header = tk.Label(
            left_panel,
            text="ผลการค้นหา",
            font=("Tahoma", 14, "bold"),
            fg="#F9FAFB",
            bg="#1F2937",
        )
        result_header.pack(anchor="w", padx=14, pady=(14, 6))

        self.result_text = tk.Text(
            left_panel,
            wrap="word",
            height=14,
            font=("Tahoma", 13),
            bg="#0F172A",
            fg="#E2E8F0",
            insertbackground="#E2E8F0",
            relief="flat",
            padx=12,
            pady=12,
        )
        self.result_text.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self._set_result_text("พิมพ์คำแล้วกด รัน\n\nถ้าพบชื่อวิดีโอ ระบบจะเล่นให้ทันที\nถ้าไม่พบ จะขึ้นข้อความว่า ไม่มี ยังไม่มีคำ")

        status_frame = tk.Frame(left_panel, bg="#1F2937")
        status_frame.pack(fill="x", padx=14, pady=(0, 14))

        status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            font=("Tahoma", 12),
            fg="#93C5FD",
            bg="#1F2937",
            anchor="w",
        )
        status_label.pack(fill="x")

        current_label = tk.Label(
            status_frame,
            textvariable=self.current_video_var,
            font=("Tahoma", 11),
            fg="#FDE68A",
            bg="#1F2937",
            anchor="w",
        )
        current_label.pack(fill="x", pady=(4, 0))

        library_header = tk.Label(
            right_panel,
            text="วิดีโอที่พบในโฟลเดอร์",
            font=("Tahoma", 14, "bold"),
            fg="#F9FAFB",
            bg="#1F2937",
        )
        library_header.pack(anchor="w", padx=14, pady=(14, 6))

        self.library_list = tk.Listbox(
            right_panel,
            font=("Tahoma", 13),
            bg="#0F172A",
            fg="#E2E8F0",
            selectbackground="#2563EB",
            highlightthickness=0,
            relief="flat",
        )
        self.library_list.pack(fill="both", expand=True, padx=14, pady=(0, 14))

    def _set_result_text(self, text: str) -> None:
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("1.0", text)
        self.result_text.configure(state="disabled")

    def _refresh_library_view(self) -> None:
        self.library.refresh()
        self.library_list.delete(0, "end")

        all_paths = self.library.all_paths()
        if not all_paths:
            self.library_list.insert("end", "ยังไม่มีวิดีโอในโฟลเดอร์นี้")
            self.status_var.set("ยังไม่มีไฟล์วิดีโอในโฟลเดอร์ วีดีโอภาษามือ")
            return

        for path in all_paths:
            self.library_list.insert("end", path.stem)

        self.status_var.set(f"พบวิดีโอทั้งหมด {len(all_paths)} ไฟล์")

    def on_refresh_library(self) -> None:
        self._refresh_library_view()
        self._set_result_text("รีเฟรชรายการวิดีโอเรียบร้อยแล้ว")

    def on_search(self) -> None:
        if self.playback_thread and self.playback_thread.is_alive():
            self.status_var.set("กำลังเล่นวิดีโออยู่ กรุณาหยุดก่อนหรือรอให้จบ")
            return

        query = self.query_var.get().strip()
        result = self.library.search(query)

        if not result.normalized_query:
            self.status_var.set("กรุณาพิมพ์คำหรือประโยคก่อน")
            self._set_result_text("ไม่มี ยังไม่มีคำ")
            self.current_video_var.set("ยังไม่ได้เลือกวิดีโอ")
            return

        summary_lines = [f"คำที่พิมพ์: {result.query}"]

        if result.exact_match:
            summary_lines.append(f"พบวิดีโอตรงตัว: {result.normalized_query}")
        elif result.matched_terms:
            summary_lines.append("พบคำที่ตรงกับวิดีโอ: " + " + ".join(result.matched_terms))
        else:
            summary_lines.append("ไม่มี ยังไม่มีคำ")

        if result.missing_terms:
            summary_lines.append("คำที่ไม่พบ: " + " | ".join(result.missing_terms))

        if result.playback_paths:
            summary_lines.append("")
            summary_lines.append("รายการที่จะเล่น:")
            for index, video_path in enumerate(result.playback_paths, start=1):
                summary_lines.append(f"{index}. {video_path.stem}")
            self.status_var.set(f"พร้อมเล่น {len(result.playback_paths)} วิดีโอ")
            self._set_result_text("\n".join(summary_lines))
            self.start_playback(result.playback_paths)
        else:
            self.status_var.set("ไม่มี ยังไม่มีคำ")
            self.current_video_var.set("ยังไม่ได้เลือกวิดีโอ")
            self._set_result_text("\n".join(summary_lines))

    def start_playback(self, playback_paths: list[Path]) -> None:
        if not playback_paths:
            return

        self.stop_event.clear()
        self.playback_thread = threading.Thread(
            target=self._playback_worker,
            args=(playback_paths,),
            daemon=True,
        )
        self.playback_thread.start()

    def _playback_worker(self, playback_paths: list[Path]) -> None:
        window_name = "Emmol Preview"
        try:
            if not self.ffplay_path:
                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

            for index, video_path in enumerate(playback_paths, start=1):
                if self.stop_event.is_set():
                    break

                self.root.after(
                    0,
                    lambda path=video_path, number=index, total=len(playback_paths): self.current_video_var.set(
                        f"กำลังเล่น: {path.stem} ({number}/{total})"
                    ),
                )

                self._speak_clip_name(path_name=video_path.stem)

                if self.ffplay_path:
                    self._play_with_ffplay(video_path, window_name)
                    continue

                self._play_with_opencv(video_path, window_name)

        finally:
            self.active_process = None
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass

            self.root.after(0, self._playback_finished)

    def _play_with_ffplay(self, video_path: Path, window_name: str) -> None:
        command = [
            self.ffplay_path,
            "-autoexit",
            "-loglevel",
            "error",
            "-x",
            str(MAX_PREVIEW_WIDTH),
            "-y",
            str(MAX_PREVIEW_HEIGHT),
            "-window_title",
            window_name,
            str(video_path),
        ]
        try:
            process = subprocess.Popen(command)
            self.active_process = process

            while not self.stop_event.is_set():
                try:
                    process.wait(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    continue

            if self.stop_event.is_set() and process.poll() is None:
                process.terminate()
                process.wait(timeout=2)
        except FileNotFoundError:
            safe_print("ไม่พบ ffplay จึงใช้โหมดเล่นแบบไม่มีเสียง")
            self.ffplay_path = None
            self._play_with_opencv(video_path, window_name)
        except Exception as exc:
            safe_print(f"เกิดข้อผิดพลาดระหว่างเล่นวิดีโอด้วย ffplay: {exc}")
        finally:
            self.active_process = None

    def _play_with_opencv(self, video_path: Path, window_name: str) -> None:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            safe_print(f"ไม่สามารถเปิดไฟล์วิดีโอได้: {video_path}")
            return

        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_delay = capture.get(cv2.CAP_PROP_FPS)
        if not frame_delay or frame_delay <= 0:
            frame_delay = 30.0
        wait_time_ms = max(1, int(1000 / frame_delay))

        scale = 1.0
        if frame_width > 0 and frame_height > 0:
            width_scale = MAX_PREVIEW_WIDTH / frame_width
            height_scale = MAX_PREVIEW_HEIGHT / frame_height
            scale = min(1.0, width_scale, height_scale)

        while not self.stop_event.is_set():
            success, frame = capture.read()
            if not success or frame is None:
                break

            if scale < 1.0:
                resized_width = max(1, int(frame.shape[1] * scale))
                resized_height = max(1, int(frame.shape[0] * scale))
                frame = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(wait_time_ms) & 0xFF
            if key in (27, ord("q")):
                self.stop_event.set()
                break

        capture.release()

    def _speak_clip_name(self, path_name: str) -> None:
        if not path_name:
            return

        text = path_name.replace("_", " ").strip()
        if not text:
            return

        escaped_text = text.replace("'", "''")
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speaker.Speak('{escaped_text}')"
        )

        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            pass

    def _playback_finished(self) -> None:
        self.playback_thread = None
        if self.stop_event.is_set():
            self.status_var.set("หยุดเล่นวิดีโอแล้ว")
        else:
            self.status_var.set("เล่นวิดีโอเสร็จแล้ว")

        if not self.query_var.get().strip():
            self.current_video_var.set("ยังไม่ได้เลือกวิดีโอ")

    def stop_playback(self) -> None:
        self.stop_event.set()
        self.status_var.set("กำลังหยุดวิดีโอ...")
        if self.active_process and self.active_process.poll() is None:
            try:
                self.active_process.terminate()
            except Exception:
                pass

    def on_close(self) -> None:
        self.stop_event.set()
        if self.active_process and self.active_process.poll() is None:
            try:
                self.active_process.terminate()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.root.destroy()


def main() -> None:
    if not VIDEO_DIR.exists():
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass

    EmmolApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()