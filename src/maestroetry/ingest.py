"""Download SDD and FMA datasets for training."""

from __future__ import annotations

import csv
import io
import logging
import random
import subprocess
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve

logger = logging.getLogger(__name__)

# Song Describer Dataset CSV on Zenodo
SDD_CSV_URL = (
    "https://zenodo.org/records/10072001/files/" "song_describer.csv?download=1"
)

# FMA metadata repository (raw CSV files)
_FMA_META_BASE = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

# FMA audio archives on archive.org
_FMA_AUDIO_URLS: dict[str, str] = {
    "small": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
    "medium": "https://os.unil.cloud.switch.ch/fma/fma_medium.zip",
    "large": "https://os.unil.cloud.switch.ch/fma/fma_large.zip",
}

# Caption templates for programmatic FMA captions.
# Each template is a format string accepting genre and tags.
_CAPTION_TEMPLATES: list[str] = [
    "A {genre} track with {tags}",
    "An energetic {genre} piece featuring {tags}",
    "A {genre} song characterized by {tags}",
    "{genre} music with {tags}",
    "A {genre} composition with elements of {tags}",
    "A track in the {genre} style, featuring {tags}",
    "{genre} recording showcasing {tags}",
    "A {genre} number driven by {tags}",
]


def build_caption(
    genres: list[str],
    title: str,
    tags: list[str],
) -> str:
    """Build a natural-language caption from FMA metadata.

    Produces descriptive captions that stylistically match
    the human-written SDD captions so the model sees a
    uniform distribution.

    Args:
        genres: Genre labels for the track.
        title: Track title (currently unused but available
            for future template enrichment).
        tags: Instrumentation or mood tags.

    Returns:
        A single descriptive sentence.
    """
    genre_str = genres[0] if genres else "unknown genre"
    if tags:
        tags_str = ", ".join(tags[:3])
    else:
        tags_str = "varied instrumentation"
    template = random.choice(_CAPTION_TEMPLATES)
    return template.format(genre=genre_str, tags=tags_str)


def _convert_to_wav(src: Path, dst: Path) -> bool:
    """Convert audio to 16kHz mono wav via ffmpeg.

    Args:
        src: Source audio file path.
        dst: Destination wav file path.

    Returns:
        True if conversion succeeded.
    """
    if dst.exists():
        return True
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(dst),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=30,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.warning("Failed to convert %s", src)
        return False
    return dst.exists()


def _download_file(url: str, dest: Path) -> Path:
    """Download a file if it does not already exist.

    Args:
        url: URL to download from.
        dest: Local destination path.

    Returns:
        The destination path.
    """
    if dest.exists():
        logger.debug("Already exists: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s ...", url)
    urlretrieve(url, dest)
    return dest


class _HttpFile:
    """Seekable file-like object backed by HTTP range requests.

    Lets ``zipfile.ZipFile`` read specific members from a remote
    zip without downloading the entire archive.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        req = Request(url, method="HEAD")
        with urlopen(req) as resp:
            self._size = int(resp.headers["Content-Length"])
            if resp.headers.get("Accept-Ranges") != "bytes":
                raise RuntimeError(
                    "Server does not support range requests"
                )
        self._pos = 0

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._size + offset
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if self._pos >= self._size:
            return b""
        if size == -1 or size is None:
            size = self._size - self._pos
        if size <= 0:
            return b""
        end = min(self._pos + size - 1, self._size - 1)
        req = Request(self.url)
        req.add_header("Range", f"bytes={self._pos}-{end}")
        with urlopen(req) as resp:
            data = resp.read()
        self._pos += len(data)
        return data


def _fetch_tracks_from_remote_zip(
    track_ids: set[int],
    dest_dir: Path,
    zip_url: str,
    archive_name: str,
) -> dict[int, Path]:
    """Download specific MP3s from a remote FMA zip via range requests.

    Uses HTTP range requests so only the zip central directory and
    the requested tracks are transferred, not the full archive.

    Args:
        track_ids: Set of FMA track IDs to fetch.
        dest_dir: Local directory to write MP3 files into.
        zip_url: URL of the remote FMA zip archive.
        archive_name: Name prefix inside the zip (e.g. "fma_large").

    Returns:
        Mapping of track ID to local MP3 path for successfully
        fetched tracks.
    """
    logger.info(
        "Fetching %d tracks from remote %s archive...",
        len(track_ids),
        archive_name,
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    http_file = _HttpFile(zip_url)
    fetched: dict[int, Path] = {}

    with zipfile.ZipFile(http_file) as zf:
        for tid in track_ids:
            fname = f"{tid:06d}.mp3"
            member = f"{archive_name}/{fname[:3]}/{fname}"
            try:
                data = zf.read(member)
            except KeyError:
                logger.debug(
                    "Track %d not in %s", tid, archive_name
                )
                continue

            mp3_path = dest_dir / fname
            mp3_path.write_bytes(data)
            fetched[tid] = mp3_path

            if len(fetched) % 50 == 0:
                logger.info(
                    "[%d] tracks fetched...", len(fetched)
                )

    logger.info(
        "Fetched %d of %d requested tracks",
        len(fetched),
        len(track_ids),
    )
    return fetched


def _read_fma_tracks_csv(
    meta_zip_path: Path,
) -> dict[int, dict[str, str]]:
    """Parse tracks.csv from the FMA metadata zip.

    Returns a dict keyed by track_id with keys
    'title' and 'genre_top'.

    Args:
        meta_zip_path: Path to fma_metadata.zip.

    Returns:
        Mapping of track ID to metadata dict.
    """
    tracks: dict[int, dict[str, str]] = {}
    with zipfile.ZipFile(meta_zip_path) as zf:
        # tracks.csv has two header rows; we need the second
        with zf.open("fma_metadata/tracks.csv") as f:
            text = io.TextIOWrapper(f, encoding="utf-8")
            reader = csv.reader(text)
            header_row_0 = next(reader)
            header_row_1 = next(reader)
            # Build combined headers from the two-level header
            headers = [
                f"{h0}.{h1}" if h0 else h1
                for h0, h1 in zip(header_row_0, header_row_1)
            ]
            # Find relevant column indices
            id_idx = 0  # first column is track_id
            title_idx = (
                headers.index("track.title")
                if "track.title" in headers
                else None
            )
            genre_idx = (
                headers.index("track.genre_top")
                if "track.genre_top" in headers
                else None
            )
            for row in reader:
                if not row or not row[0].strip():
                    continue
                try:
                    tid = int(row[id_idx])
                except ValueError:
                    continue
                title = row[title_idx] if title_idx is not None else ""
                genre = row[genre_idx] if genre_idx is not None else ""
                tracks[tid] = {"title": title, "genre_top": genre}
    return tracks


def _fma_track_id_to_path(track_id: int, audio_root: Path) -> Path | None:
    """Resolve an FMA track ID to its file path inside the archive.

    FMA stores tracks as ``audio_root/XXX/XXXXXX.mp3`` where
    the subdirectory is the zero-padded track ID truncated to
    the first three digits.

    Args:
        track_id: Numeric FMA track ID.
        audio_root: Root directory of the extracted FMA audio.

    Returns:
        Path to the mp3 file, or None if not found.
    """
    fname = f"{track_id:06d}.mp3"
    subdir = fname[:3]
    path = audio_root / subdir / fname
    return path if path.exists() else None


def ingest_sdd(
    data_dir: str | Path = "data",
    max_samples: int | None = None,
) -> list[dict[str, str]]:
    """Download the Song Describer Dataset and matching FMA audio.

    SDD is a CSV mapping FMA track IDs to human-written
    descriptions. This function downloads the CSV, then
    resolves audio for each track: first checking any
    locally-extracted FMA subsets, then fetching missing
    tracks individually from the remote fma_large archive
    via HTTP range requests.

    Args:
        data_dir: Root data directory.
        max_samples: Cap on number of samples to process.

    Returns:
        List of manifest row dicts with audio_path, text, source.
    """
    data_dir = Path(data_dir)
    audio_dir = data_dir / "audio" / "sdd"
    audio_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = data_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    # Download SDD CSV
    sdd_csv_path = downloads_dir / "song_describer.csv"
    _download_file(SDD_CSV_URL, sdd_csv_path)

    # Parse SDD CSV to collect (track_id, caption) pairs
    sdd_entries: list[tuple[int, str]] = []
    with sdd_csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples is not None and i >= max_samples:
                break
            track_id_str = row.get(
                "track_id", row.get("id", "")
            )
            if not track_id_str:
                continue
            try:
                track_id = int(track_id_str)
            except ValueError:
                continue
            caption = row.get(
                "caption", row.get("description", "")
            )
            if not caption:
                continue
            sdd_entries.append((track_id, caption))

    if not sdd_entries:
        logger.warning("No valid entries found in SDD CSV")
        return []

    # Resolve MP3 sources: check local FMA extractions first
    needed_ids: set[int] = set()
    local_mp3s: dict[int, Path] = {}

    for tid, _ in sdd_entries:
        wav_path = audio_dir / f"{tid:06d}.wav"
        if wav_path.exists():
            continue  # already converted

        mp3_path = None
        for subset in ("small", "medium", "large", "full"):
            fma_root = downloads_dir / f"fma_{subset}"
            candidate = _fma_track_id_to_path(tid, fma_root)
            if candidate is not None:
                mp3_path = candidate
                break

        if mp3_path is not None:
            local_mp3s[tid] = mp3_path
        else:
            needed_ids.add(tid)

    # Fetch missing tracks from remote fma_large via range requests
    if needed_ids:
        logger.info(
            "%d of %d SDD tracks not found locally, "
            "fetching from remote archive...",
            len(needed_ids),
            len(sdd_entries),
        )
        mp3_dir = downloads_dir / "sdd_tracks"
        fetched = _fetch_tracks_from_remote_zip(
            needed_ids,
            mp3_dir,
            _FMA_AUDIO_URLS["large"],
            "fma_large",
        )
        local_mp3s.update(fetched)

    # Convert to wav and build manifest rows
    rows: list[dict[str, str]] = []
    for tid, caption in sdd_entries:
        wav_path = audio_dir / f"{tid:06d}.wav"
        if not wav_path.exists():
            mp3_path = local_mp3s.get(tid)
            if mp3_path is None:
                continue
            if not _convert_to_wav(mp3_path, wav_path):
                continue

        rows.append(
            {
                "audio_path": str(wav_path),
                "text": caption,
                "source": "sdd",
            }
        )
        if len(rows) % 100 == 0:
            logger.info("[%d] SDD tracks processed...", len(rows))

    logger.info("SDD ingest complete: %d samples", len(rows))
    return rows


def ingest_fma(
    data_dir: str | Path = "data",
    subset: str = "small",
    max_samples: int | None = None,
) -> list[dict[str, str]]:
    """Download FMA audio and generate captions from metadata.

    Downloads the FMA audio archive and metadata, then builds
    descriptive captions from genre, title, and tag fields.

    Args:
        data_dir: Root data directory.
        subset: FMA subset to use ("small" or "medium").
        max_samples: Cap on number of samples to process.

    Returns:
        List of manifest row dicts with audio_path, text, source.
    """
    if subset not in _FMA_AUDIO_URLS:
        raise ValueError(
            f"Unknown FMA subset {subset!r}; "
            f"choose from {list(_FMA_AUDIO_URLS)}"
        )

    data_dir = Path(data_dir)
    audio_dir = data_dir / "audio" / "fma"
    audio_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = data_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    # Download metadata
    meta_zip_path = downloads_dir / "fma_metadata.zip"
    _download_file(_FMA_META_BASE, meta_zip_path)

    # Download audio archive
    audio_zip_name = f"fma_{subset}.zip"
    audio_zip_path = downloads_dir / audio_zip_name
    _download_file(_FMA_AUDIO_URLS[subset], audio_zip_path)

    # Extract audio
    fma_audio_root = downloads_dir / f"fma_{subset}"
    if not fma_audio_root.exists():
        logger.info("Extracting FMA %s audio...", subset)
        with zipfile.ZipFile(audio_zip_path) as zf:
            zf.extractall(downloads_dir)

    # Parse metadata
    tracks_meta = _read_fma_tracks_csv(meta_zip_path)

    # Process tracks
    rows: list[dict[str, str]] = []
    count = 0
    for tid, meta in tracks_meta.items():
        if max_samples is not None and count >= max_samples:
            break

        mp3_path = _fma_track_id_to_path(tid, fma_audio_root)
        if mp3_path is None:
            continue

        genre = meta.get("genre_top", "")
        title = meta.get("title", "")
        genres = [genre] if genre else []
        caption = build_caption(genres, title, [])

        wav_path = audio_dir / f"{tid:06d}.wav"
        if not _convert_to_wav(mp3_path, wav_path):
            continue

        rows.append(
            {
                "audio_path": str(wav_path),
                "text": caption,
                "source": "fma",
            }
        )
        count += 1
        if count % 100 == 0:
            logger.info("[%d] FMA tracks processed...", count)

    logger.info("FMA ingest complete: %d samples", len(rows))
    return rows


def ingest(
    data_dir: str | Path = "data",
    fma_subset: str = "small",
    max_samples_sdd: int | None = None,
    max_samples_fma: int | None = None,
    sdd_only: bool = False,
    fma_only: bool = False,
) -> None:
    """Orchestrate SDD and FMA ingest into a unified manifest.

    Calls ingest_sdd and/or ingest_fma and writes a combined
    manifest.csv with audio_path, text, and source columns.

    Args:
        data_dir: Root data directory.
        fma_subset: FMA subset name ("small" or "medium").
        max_samples_sdd: Cap on SDD samples.
        max_samples_fma: Cap on FMA samples.
        sdd_only: If True, only ingest SDD.
        fma_only: If True, only ingest FMA.
    """
    data_dir = Path(data_dir)
    manifest_path = data_dir / "manifest.csv"
    all_rows: list[dict[str, str]] = []

    if not fma_only:
        all_rows.extend(ingest_sdd(data_dir, max_samples=max_samples_sdd))

    if not sdd_only:
        all_rows.extend(
            ingest_fma(
                data_dir,
                subset=fma_subset,
                max_samples=max_samples_fma,
            )
        )

    fieldnames = ["audio_path", "text", "source"]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info(
        "Done. %d total samples. Manifest: %s",
        len(all_rows),
        manifest_path,
    )
