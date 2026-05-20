#!/usr/bin/env python3
"""
TikTok Video Transcription Script — geen ffmpeg nodig
Werkt in Jupyter Notebook én via de terminal.

Installatie (eenmalig):
    pip install faster-whisper yt-dlp openpyxl pandas tqdm
"""

import os
import tempfile
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

# ============================================================
#  INSTELLINGEN — pas dit aan naar jouw situatie
# ============================================================
path = Path("output-single/dataenrichment.xlsx")
INPUT_FILE       = path                     # pad naar je Excel-bestand
LINK_COLUMN      = "link"                   # kolomnaam met TikTok URLs
OUTPUT_COLUMN    = "transcriptie"           # kolomnaam voor de transcriptie
WHISPER_MODEL    = "small"                  # tiny | base | small | medium | large-v2
WORKERS          = 4                        # 1 worker per CPU core
DEVICE           = "auto"                   # auto | cpu | cuda
SKIP_DONE        = True                     # sla rijen over die al een transcriptie hebben
CHECKPOINT_EVERY = 50                       # tussentijds opslaan elke N rijen

# Beperk taaldetectie tot deze talen
LANGUAGES        = ["nl", "en"]
# ============================================================


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("transcription.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def download_video(url: str, output_path: str) -> bool:
    import subprocess
    cmd = [
        "yt-dlp",
        "--quiet", "--no-warnings",
        "--format", "best[ext=mp4]/best",
        "--no-part",
        "--output", output_path,
        "--socket-timeout", "15",
        "--retries", "2",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=90)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception as e:
        log.debug(f"Download fout: {e}")
        return False


def find_downloaded_file(tmpdir: str) -> str | None:
    for f in os.listdir(tmpdir):
        full = os.path.join(tmpdir, f)
        if os.path.isfile(full) and not f.endswith(".part"):
            return full
    return None


def transcribe_file(file_path: str, model) -> str:
    segments, info = model.transcribe(
        file_path,
        beam_size=1,                        # snelste decodering
        language=None,                      # auto-detect
        language_detection_segments=1,      # detecteer taal op basis van 1 segment (sneller)
        language_detection_threshold=0.5,
        condition_on_previous_text=False,   # geen context bijhouden = sneller
        vad_filter=True,                    # stilte overslaan
        vad_parameters={"min_silence_duration_ms": 500},
    )
    # Filter op verwachte talen; markeer als onbekend als taal niet herkend
    if info.language not in LANGUAGES:
        log.debug(f"Onverwachte taal gedetecteerd: {info.language} — toch transcriberen")
    return " ".join(seg.text.strip() for seg in segments).strip()


def process_row(args):
    idx, url, model = args
    if not isinstance(url, str) or not url.strip():
        return idx, ""

    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, "video.mp4")

        if not download_video(url.strip(), target):
            log.debug(f"[{idx}] Niet beschikbaar: {url}")
            return idx, "VIDEO_NOT_AVAILABLE"

        video_file = find_downloaded_file(tmpdir) or target
        if not os.path.exists(video_file):
            return idx, "VIDEO_NOT_AVAILABLE"

        try:
            text = transcribe_file(video_file, model)
        except Exception as e:
            log.warning(f"[{idx}] Transcriptie fout: {e}")
            return idx, "TRANSCRIPTION_ERROR"

    return idx, text if text else "NO_SPEECH_DETECTED"


def run():
    input_path = Path(INPUT_FILE)
    if not input_path.exists():
        log.error(f"Bestand niet gevonden: {input_path}")
        return

    output_path = input_path.with_stem(input_path.stem + "_transcribed")

    log.info(f"Excel laden: {input_path}")
    df = pd.read_excel(input_path, dtype=str)

    if LINK_COLUMN not in df.columns:
        log.error(f"Kolom '{LINK_COLUMN}' niet gevonden. Beschikbare kolommen: {df.columns.tolist()}")
        return

    if OUTPUT_COLUMN not in df.columns:
        df[OUTPUT_COLUMN] = ""

    if SKIP_DONE:
        todo_mask = df[OUTPUT_COLUMN].isna() | (df[OUTPUT_COLUMN].str.strip() == "")
    else:
        todo_mask = pd.Series([True] * len(df))

    todo_indices = df.index[todo_mask].tolist()
    log.info(f"Totaal: {len(df)} | Te verwerken: {len(todo_indices)} | Al klaar: {len(df) - len(todo_indices)}")

    log.info(f"Whisper model laden: '{WHISPER_MODEL}' ...")
    from faster_whisper import WhisperModel

    device = DEVICE
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    compute_type = "int8" if device == "cpu" else "float16"
    model = WhisperModel(
        WHISPER_MODEL,
        device=device,
        compute_type=compute_type,
        cpu_threads=4,          # gebruik alle 4 cores per transcriptie
        num_workers=WORKERS,    # prefetch workers
    )
    log.info(f"Model geladen op {device} ({compute_type})")

    processed = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(process_row, (idx, df.at[idx, LINK_COLUMN], model)): idx
            for idx in todo_indices
        }

        with tqdm(total=len(todo_indices), desc="Transcriberen", unit="video") as pbar:
            for future in as_completed(futures):
                try:
                    idx, text = future.result()
                    df.at[idx, OUTPUT_COLUMN] = text
                    processed += 1

                    if text in ("VIDEO_NOT_AVAILABLE", "TRANSCRIPTION_ERROR"):
                        errors += 1

                    if processed % CHECKPOINT_EVERY == 0:
                        df.to_excel(output_path, index=False)
                        log.info(f"Checkpoint: {processed} verwerkt → {output_path}")

                except Exception as e:
                    log.error(f"Onverwachte fout: {e}")
                    errors += 1

                pbar.update(1)
                pbar.set_postfix({"fouten": errors})

    df.to_excel(output_path, index=False)
    log.info(f"Klaar! {processed} verwerkt, {errors} fouten/niet-beschikbaar")
    log.info(f"Resultaat: {output_path}")


run()