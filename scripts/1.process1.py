#!/usr/bin/env python3
"""
TikTok kijkgeschiedenis pipeline
Stap 1 – CSV parsen & sessie-features
Stap 2 – Video's downloaden & analyseren (YOLO)
Stap 3 – Alles samenvoegen
Stap 4 – Opschonen & engagement-features
"""

from __future__ import annotations

import io
import json
import shutil
import time
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import cv2
import pandas as pd
import pyktok as pyk
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Instellingen – pas hier aan
# ---------------------------------------------------------------------------

# Zoek de invoerbestanden: eerst in de huidige werkmap, dan één map hoger
def _find_input_file(name: str) -> Path:
    for candidate in [Path.cwd() / name, Path.cwd().parent / name, Path(name)]:
        if candidate.exists():
            return candidate
    print(f"⚠ '{name}' niet gevonden. Huidige werkmap: {Path.cwd()}")
    print(f"  Zorg dat het bestand in dezelfde map staat als dit script/notebook.")
    return Path(name)  # fallback; geeft later een duidelijke fout

HISTORY_CSV   = _find_input_file("Kijkgeschiedenis.txt")
LIKES_CSV     = _find_input_file("Likelijst.txt")
OUTPUT_DIR    = Path("output-single")           # uitvoermap

SESSION_GAP_MIN    = 7     # minuten tussen video's voor nieuwe sessie
MIN_SESSION_SIZE   = 10    # minimaal aantal video's per sessie (stap 4)
HISTORY_LIMIT      = None  # bijv. 100 om te testen; None = alles
BATCH_SIZE         = 40    # video's per batch
FRAME_INTERVAL     = 15    # elke N frames analyseren
MAX_SAMPLED_FRAMES = 120   # maximaal gesamplede frames per video
YOLO_MODEL         = "yolov8n.pt"

COLUMNS_TO_DROP = [
    "video", "link_analysis", "link_metadata", "metadata",
    "video_timestamp", "video_locationcreated",
    "video_diggcount", "video_sharecount", "video_commentcount",
    "video_playcount", "author_name", "author_followercount",
    "author_followingcount", "author_heartcount", "author_videocount",
    "author_diggcount", "author_verified",
    "poi_name", "poi_address", "poi_city",
]

# Afgeleide paden (niet aanpassen)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_OUT  = OUTPUT_DIR / "geschiedenis.csv"
LIKES_OUT    = OUTPUT_DIR / "likes.csv"
METADATA_OUT = OUTPUT_DIR / "video_metadata.csv"
ANALYSIS_OUT = OUTPUT_DIR / "video_analysis.csv"
COMBINED_OUT = OUTPUT_DIR / "combined.csv"
FINAL_CSV    = OUTPUT_DIR / "dataenrichment.csv"
FINAL_XLSX   = OUTPUT_DIR / "dataenrichment.xlsx"
VIDEO_DIR    = OUTPUT_DIR / "Videodata"
PROGRESS_FILE = OUTPUT_DIR / "progress.txt"


# ---------------------------------------------------------------------------
# Hulpfuncties
# ---------------------------------------------------------------------------

def extract_video_id(url: str) -> str:
    """Haal numerieke video_id uit TikTok-URL; leeg als niet gevonden."""
    match = pd.Series([str(url)]).str.extract(r"/(\d+)/?$")[0].iloc[0]
    return "" if pd.isna(match) else str(match)


def read_csv_file(path: Path) -> pd.DataFrame:
    """
    Lees een TikTok-export tekstbestand.
    Geeft DataFrame met kolommen ['datum', 'link'].
    """
    if not path.exists():
        print(f"⚠ Bestand niet gevonden: {path}")
        return pd.DataFrame(columns=["datum", "link"])

    # Lees ruwe regels, strip alle aanhalingstekens per regel
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    cleaned = []
    for line in raw_lines:
        # Verwijder aanhalingstekens aan begin/einde en splits op '" "' of gewone spatie-blokken
        line = line.strip().strip('"')
        cleaned.append(line)

    # Zoek de header en dataregels
    if not cleaned:
        print(f"✗ Leeg bestand: {path}")
        return pd.DataFrame(columns=["datum", "link"])

    # Splits elke regel op '"' als scheidingsteken, filter lege delen
    records = []
    for line in cleaned:
        parts = [p.strip() for p in line.split('"') if p.strip() and p.strip() != ' ']
        if parts:
            records.append(parts)

    if len(records) < 2:
        print(f"✗ Niet genoeg regels in {path}")
        return pd.DataFrame(columns=["datum", "link"])

    # Eerste rij = headers, rest = data
    headers = [h.lower().strip() for h in records[0]]
    data_rows = records[1:]

    # Bepaal welke kolom datum en link is
    date_idx = next((i for i, h in enumerate(headers) if "date" in h), None)
    link_idx = next((i for i, h in enumerate(headers) if "video" in h), None)

    if date_idx is None or link_idx is None:
        print(f"✗ Geen datum/video kolommen gevonden. Headers: {headers}")
        return pd.DataFrame(columns=["datum", "link"])

    rows = []
    for parts in data_rows:
        if len(parts) > max(date_idx, link_idx):
            rows.append({
                "datum": parts[date_idx].strip(),
                "link": parts[link_idx].strip(),
            })

    df = pd.DataFrame(rows)
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce", utc=True)
    df = df.dropna(subset=["datum", "link"]).reset_index(drop=True)
    print(f"  ✓ {path.name}: {len(df)} records ingelezen")
    return df


def safe_merge(df_main: pd.DataFrame, path: Path, label: str) -> pd.DataFrame:
    """Voeg een CSV samen met df_main op video_id; overslaan bij problemen."""
    if not path.exists():
        print(f"⚠ {path.name} niet gevonden — overgeslagen.")
        return df_main
    try:
        df_extra = pd.read_csv(path)

        # video_id kan als float opgeslagen zijn (bijv. 7.437520e+18) — zet om naar int-string
        if "video_id" in df_extra.columns:
            df_extra["video_id"] = (
                df_extra["video_id"]
                .apply(lambda x: str(int(float(x))) if pd.notna(x) else "")
                .str.strip()
            )
        elif "link" in df_extra.columns:
            df_extra["video_id"] = (
                df_extra["link"].astype(str).str.extract(r"/(\d+)/?$")[0].fillna("")
            )
        else:
            print(f"⚠ Geen video_id of link kolom in {path.name} — overgeslagen.")
            return df_main

        # Zorg dat df_main video_id ook een schone int-string is
        df_main["video_id"] = (
            df_main["video_id"]
            .apply(lambda x: str(int(float(x))) if pd.notna(x) and str(x) not in ("", "nan") else "")
        )

        matches = df_main["video_id"].isin(df_extra["video_id"]).sum()
        print(f"  {label}: {matches}/{len(df_main)} matches")
        return df_main.merge(df_extra, on="video_id", how="left",
                             suffixes=("", f"_{label}"))
    except Exception as e:
        print(f"✗ Fout bij samenvoegen {label}: {e}")
        return df_main


# ---------------------------------------------------------------------------
# Stap 1 – CSV parsen & sessie-features
# ---------------------------------------------------------------------------

def stap1_parse():
    print("\n=== Stap 1: Parsen ===")

    # Kijkgeschiedenis
    history = read_csv_file(HISTORY_CSV)
    if not history.empty:
        history = history.sort_values("datum").reset_index(drop=True)
        history.insert(0, "nr", range(1, len(history) + 1))
        history["sessietijd"] = history["datum"].dt.time

        delta_min = history["datum"].diff().dt.total_seconds().div(60)
        history["sessie_nr"] = (delta_min.isna() | (delta_min >= SESSION_GAP_MIN)).cumsum()

        volgende = history["datum"].shift(-1)
        zelfde_sessie = history["sessie_nr"] == history["sessie_nr"].shift(-1)
        history["kijktijd_sec"] = (volgende - history["datum"]).dt.total_seconds().where(zelfde_sessie)

        history["video_id"] = history["link"].map(extract_video_id)

        if HISTORY_LIMIT is not None:
            history = history.head(HISTORY_LIMIT).copy()

    history.to_csv(HISTORY_OUT, index=False)

    # Likes
    likes = read_csv_file(LIKES_CSV)
    if not likes.empty:
        likes["video_id"] = likes["link"].map(extract_video_id)
    likes.to_csv(LIKES_OUT, index=False)

    print(f"Geschiedenis: {len(history)} records | Likes: {len(likes)} records")
    return history, likes


# ---------------------------------------------------------------------------
# Stap 2 – Downloaden & analyseren
# ---------------------------------------------------------------------------

def download_video(link: str) -> Path | None:
    """Download één TikTok-video; geeft pad naar mp4 of None bij fout."""
    video_id = extract_video_id(link)
    if not video_id:
        return None

    expected = Path(f"share_video_{video_id}_.mp4")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        pyk.save_tiktok(link, True, str(METADATA_OUT))

    if not expected.exists():
        print(f"  ⚠ Verwacht bestand niet gevonden: {expected}")
        return None

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    dst = VIDEO_DIR / expected.name
    shutil.move(str(expected), str(dst))
    return dst


def analyze_video(video_path: Path, model: YOLO) -> tuple[str, dict]:
    """Detecteer objecten op gesamplede frames; geeft (hoofdonderwerp, tellingen)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Kan video niet openen: {video_path}")

    labels = []
    frame_idx = 0
    sampled = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % FRAME_INTERVAL == 0:
                results = model(frame, verbose=False)
                for box in results[0].boxes:
                    labels.append(model.names[int(box.cls[0])])
                sampled += 1
                if MAX_SAMPLED_FRAMES and sampled >= MAX_SAMPLED_FRAMES:
                    break
            frame_idx += 1
    finally:
        cap.release()

    if not labels:
        return "no data", {}

    counts = Counter(labels)
    return counts.most_common(1)[0][0], dict(counts)


def stap2_download_en_analyseer():
    print("\n=== Stap 2: Downloaden & analyseren ===")

    # Zorg dat analysis CSV de juiste headers heeft
    if not ANALYSIS_OUT.exists():
        pd.DataFrame(columns=["video", "subject", "objects", "link", "video_id"]).to_csv(
            ANALYSIS_OUT, index=False
        )
    if not METADATA_OUT.exists():
        pd.DataFrame(columns=["link", "metadata"]).to_csv(METADATA_OUT, index=False)

    data = pd.read_csv(HISTORY_OUT)
    link_col = "link" if "link" in data.columns else data.columns[2]

    # Links die al geanalyseerd zijn overslaan
    completed = set(pd.read_csv(ANALYSIS_OUT)["link"].dropna().astype(str))

    model = YOLO(YOLO_MODEL)
    total = len(data)
    start_index = 0
    if PROGRESS_FILE.exists():
        try:
            start_index = int(PROGRESS_FILE.read_text().strip())
        except Exception:
            start_index = 0

    print(f"Start vanaf index {start_index} | totaal: {total}")
    t0 = time.perf_counter()

    for batch_start in range(start_index, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_results = []
        elapsed = time.perf_counter() - t0
        print(f"\nBatch {batch_start//BATCH_SIZE + 1} ({batch_start}–{batch_end-1}) | {elapsed:.0f}s verstreken")

        for i in range(batch_start, batch_end):
            link = str(data.iloc[i][link_col]).strip()

            if not link or link.lower() == "nan":
                print("  - Lege link overgeslagen")
                continue

            if link in completed:
                print("  - Al geanalyseerd, overgeslagen")
                continue

            video_id = extract_video_id(link)
            row = {"video": "no data", "subject": "no data",
                   "objects": "no data", "link": link, "video_id": video_id}

            for attempt in range(2):
                try:
                    video_path = download_video(link)
                    if not video_path:
                        break

                    print(f"  ✓ Download: {video_path.name}")
                    subject, counts = analyze_video(video_path, model)
                    print(f"  ✓ Analyse: {subject}")

                    row = {
                        "video": video_path.name,
                        "subject": subject,
                        "objects": json.dumps(counts, ensure_ascii=True),
                        "link": link,
                        "video_id": video_id,
                    }
                    completed.add(link)
                    break

                except KeyError as e:
                    # Niet-herprobeerbare downloadfout (ontbrekende TikTok API-velden)
                    if e.args and e.args[0] in {"itemInfo", "downloadAddr"}:
                        print(f"  ✗ Niet-herprobeerbaar: {e}")
                        pd.DataFrame([{"link": link, "metadata": "no data"}]).to_csv(
                            METADATA_OUT, mode="a", header=False, index=False
                        )
                        break
                    if attempt == 0:
                        print(f"  ! Fout, opnieuw proberen: {e}")
                        continue
                    print(f"  ✗ Mislukt: {e}")

                except Exception as e:
                    if attempt == 0:
                        print(f"  ! Fout, opnieuw proberen: {e}")
                        continue
                    print(f"  ✗ Mislukt: {e}")
                    pd.DataFrame([{"link": link, "metadata": "no data"}]).to_csv(
                        METADATA_OUT, mode="a", header=False, index=False
                    )

            batch_results.append(row)

        if batch_results:
            pd.DataFrame(batch_results).to_csv(ANALYSIS_OUT, mode="a", header=False, index=False)

        PROGRESS_FILE.write_text(str(batch_end))

        if VIDEO_DIR.exists():
            shutil.rmtree(VIDEO_DIR)
            print("✓ Tijdelijke videomap verwijderd")

    print("ALLE VIDEO'S VERWERKT")


# ---------------------------------------------------------------------------
# Stap 3 – Samenvoegen
# ---------------------------------------------------------------------------

def stap3_samenvoegen() -> pd.DataFrame:
    print("\n=== Stap 3: Samenvoegen ===")

    df = pd.read_csv(HISTORY_OUT)
    df["video_id"] = df["link"].astype(str).str.extract(r"/(\d+)/?$")[0].astype(str)

    df = safe_merge(df, LIKES_OUT,    "likes")
    df = safe_merge(df, ANALYSIS_OUT, "analysis")
    df = safe_merge(df, METADATA_OUT, "metadata")

    df.to_csv(COMBINED_OUT, index=False)
    print(f"Gecombineerd opgeslagen: {COMBINED_OUT}")
    return df


# ---------------------------------------------------------------------------
# Stap 4 – Opschonen & engagement-features
# ---------------------------------------------------------------------------

def stap4_opschonen(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Stap 4: Opschonen & features ===")

    # Verwijder overbodige kolommen
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")

    # Filter rijen zonder bruikbare combined_text
    if "combined_text" in df.columns:
        voor = len(df)
        df = df[df["combined_text"].notna() & (df["combined_text"] != "no data")].copy()
        print(f"  combined_text filter: {voor} → {len(df)} rijen")
    else:
        print(f"⚠ Kolom 'combined_text' niet gevonden — filter overgeslagen.")
        print(f"  Beschikbare kolommen: {list(df.columns)}")

    if df.empty:
        print("✗ Geen rijen over na combined_text filter — stap 4 gestopt.")
        return df

    # Filter sessies met te weinig video's
    if "sessie_nr" not in df.columns:
        print(f"✗ Kolom 'sessie_nr' niet gevonden. Beschikbare kolommen: {list(df.columns)}")
        return df

    if "nr" not in df.columns:
        df["nr"] = df.groupby("sessie_nr").cumcount() + 1

    session_counts = df.groupby("sessie_nr")["nr"].transform("count")
    df = df[session_counts >= MIN_SESSION_SIZE].copy()
    print(f"Na sessiefilter: {len(df)} rijen | {df['sessie_nr'].nunique()} sessies")

    # Liked-vlag per video
    if "datum_likes" in df.columns:
        df["liked"] = df["datum_likes"].notna().astype(int)
    else:
        print("⚠ Kolom 'datum_likes' niet gevonden — liked = 0.")
        df["liked"] = 0

    # Engagement-typology: 1 als sessie ≥ 2 likes heeft
    likes_per_sessie = df.groupby("sessie_nr")["liked"].transform("sum")
    df["engagementtypology"] = (likes_per_sessie > 1).astype(int)

    # Kijktijdscore (0–1)
    if "video_duration" in df.columns:
        df["kijktijd_score"] = df["kijktijd_sec"].div(df["video_duration"]).clip(upper=1.0)
    else:
        print("⚠ Kolom 'video_duration' niet gevonden — kijktijd_score overgeslagen.")

    df.to_csv(FINAL_CSV, index=False)
    df.to_excel(FINAL_XLSX, index=False)
    print(f"Klaar! {len(df)} rijen, {len(df.columns)} kolommen → {FINAL_XLSX}")
    return df


# ---------------------------------------------------------------------------
# Hoofdprogramma
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    stap1_parse()
    stap2_download_en_analyseer()
    df = stap3_samenvoegen()
    stap4_opschonen(df)