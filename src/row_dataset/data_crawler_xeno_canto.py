"""Build a diverse, DB3V-disjoint Xeno-canto dataset for the DB3V species.

The script downloads a configurable set of quality grades, prioritizes country
and recording-session diversity, and guarantees that DB3V source recording IDs
are excluded. It writes per-recording metadata so training/validation splits
can later use recording, location, or country groups rather than audio segments.
XENO_CANTO_API_KEY, when set, overrides the stored legacy key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import wave
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import filetype
import requests
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError


API_URL = "https://xeno-canto.org/api/3/recordings"
API_KEY = "0d1823264f3d866a05df71132f35b54ae10391cf"
REQUEST_TIMEOUT = (15, 90)
PAGE_SIZE = 500
DEFAULT_QUALITY_GRADES = ("A", "B", "C", "D", "E")
DEFAULT_TARGET_PER_SPECIES = 300
DEFAULT_OUTPUT_DIR = Path("src/row_dataset") / "row_bird_dataset_A_next"
DEFAULT_DB3V_ROOT = Path("src/row_dataset") / "DB3V" / "extracted" / "data_wav_8s_2"
DEFAULT_OVERLAP_LOG = Path("src/row_dataset") / "xeno_removed_db3v_overlap.json"
DEFAULT_MIN_DURATION_SECONDS = 1.0
DEFAULT_REQUEST_DELAY_SECONDS = 0.25
MAX_REQUEST_ATTEMPTS = 4
PROGRESS_INTERVAL = 10
DB3V_SPECIES = (
    "Agelaius phoeniceus",
    "Cardinalis cardinalis",
    "Certhia americana",
    "Corvus brachyrhynchos",
    "Molothrus ater",
    "Setophaga aestiva",
    "Setophaga ruticilla",
    "Spinus tristis",
    "Tringa semipalmata",
    "Turdus migratorius",
)
ELIGIBLE_TRAINING_SPECIES = (
    "Agelaius phoeniceus",
    "Cardinalis cardinalis",
    "Certhia americana",
    "Corvus brachyrhynchos",
    "Setophaga aestiva",
    "Setophaga ruticilla",
    "Spinus tristis",
    "Turdus migratorius",
)


def request_with_retries(url: str, **kwargs: Any) -> requests.Response:
    """Return a successful response, backing off on transient API failures."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
        try:
            response = requests.get(url, **kwargs)
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt == MAX_REQUEST_ATTEMPTS:
                break
            delay = min(30.0, 2.0 ** (attempt - 1)) + random.uniform(0.0, 0.5)
            print(f"Request attempt {attempt}/{MAX_REQUEST_ATTEMPTS} failed: {error}; retrying in {delay:.1f}s.")
            time.sleep(delay)
    raise RuntimeError(f"Request failed after {MAX_REQUEST_ATTEMPTS} attempts: {last_error}")


def fetch_recordings(
    species: str,
    api_key: str,
    request_delay_seconds: float,
    quality_grades: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Fetch species metadata then retain only the requested quality grades."""
    query = f'sp:"{species}" grp:birds'
    common_params = {"query": query, "key": api_key, "per_page": PAGE_SIZE}
    first_page = request_with_retries(
        API_URL,
        params={**common_params, "page": 1},
        timeout=REQUEST_TIMEOUT,
    ).json()

    recordings = list(first_page.get("recordings", []))
    page_count = int(first_page.get("numPages", 1))
    for page in range(2, page_count + 1):
        time.sleep(request_delay_seconds)
        response = request_with_retries(
            API_URL,
            params={**common_params, "page": page},
            timeout=REQUEST_TIMEOUT,
        )
        recordings.extend(response.json().get("recordings", []))

    return [
        recording
        for recording in recordings
        if str(recording.get("q") or "").upper() in quality_grades
    ]


def recording_id(recording: dict[str, Any]) -> str:
    return str(recording.get("id", ""))


def recording_session_key(recording: dict[str, Any]) -> str:
    """Return a conservative key for recordings likely made in one session."""
    country = str(recording.get("cnt") or "Unknown").strip().casefold()
    location = str(recording.get("loc") or "Unknown").strip().casefold()
    date = str(recording.get("date") or "Unknown").strip()
    latitude = str(recording.get("lat") or "").strip()
    longitude = str(recording.get("lng") or "").strip()
    coordinate = f"{latitude},{longitude}" if latitude or longitude else ""
    return "|".join((country, location, date, coordinate))


def balanced_candidates(
    recordings: list[dict[str, Any]], excluded_ids: set[str], seed: int
) -> list[dict[str, Any]]:
    """Return a reproducible country-round-robin ordering of eligible recordings."""
    countries: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    seen_ids: set[str] = set()
    for recording in recordings:
        identifier = recording_id(recording)
        if (
            not identifier
            or identifier in seen_ids
            or identifier in excluded_ids
            or not recording.get("file")
        ):
            continue
        seen_ids.add(identifier)
        country = str(recording.get("cnt") or "Unknown")
        countries[country].append(recording)

    randomizer = random.Random(seed)
    country_names = sorted(countries)
    randomizer.shuffle(country_names)
    for country in country_names:
        country_records = list(countries[country])
        random.Random(f"{seed}:{country}").shuffle(country_records)
        countries[country] = deque(country_records)

    selected: list[dict[str, Any]] = []
    while country_names:
        next_countries: list[str] = []
        for country in country_names:
            if countries[country]:
                selected.append(countries[country].popleft())
            if countries[country]:
                next_countries.append(country)
        country_names = next_countries
    return selected


def load_overlap_log_ids(path: Path) -> set[str]:
    """Load legacy overlap IDs when the historical removal log is available."""
    if not path.exists():
        return set()
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Could not parse overlap log: {path}") from error
    identifiers: set[str] = set()
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, dict) and entry.get("id"):
            identifiers.add(str(entry["id"]))
        elif isinstance(entry, str):
            identifiers.add(entry)
    return identifiers


def load_db3v_source_ids(db3v_root: Path) -> set[str]:
    """Extract every original Xeno-canto ID represented by DB3V WAV files."""
    if not db3v_root.exists():
        raise FileNotFoundError(
            f"DB3V source directory is required to protect the independent test set: {db3v_root}"
        )
    identifiers: set[str] = set()
    for audio_path in db3v_root.glob("*/*/*.wav"):
        match = re.match(r"(\d+)_", audio_path.stem)
        if match:
            identifiers.add(match.group(1))
    if not identifiers:
        raise ValueError(f"No DB3V source IDs were found under {db3v_root}")
    return identifiers


class DuplicateAudioError(RuntimeError):
    """Raised when two different Xeno-canto IDs normalize to identical WAV data."""


def wav_paths_by_id(species_dir: Path) -> dict[str, Path]:
    """Return local WAV paths indexed by a Xeno-canto ID embedded in their name."""
    paths: dict[str, Path] = {}
    for audio_path in species_dir.glob("*.wav"):
        match = re.search(r"XC(\d+)", audio_path.stem)
        if match:
            paths[match.group(1)] = audio_path
    return paths


def metadata_recording_ids(metadata_path: Path) -> set[str]:
    """Read IDs already represented in the metadata manifest."""
    if not metadata_path.exists():
        return set()
    identifiers: set[str] = set()
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        try:
            identifier = json.loads(line).get("id")
        except json.JSONDecodeError:
            continue
        if identifier:
            identifiers.add(str(identifier))
    return identifiers


def existing_recording_ids(species_dir: Path) -> set[str]:
    """Read only IDs with a local WAV, so an interrupted run can resume safely."""
    identifiers = set(wav_paths_by_id(species_dir))
    metadata_path = species_dir / "metadata.jsonl"
    if not metadata_path.exists():
        return identifiers
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        try:
            metadata = json.loads(line)
            identifier = str(metadata["id"])
        except (json.JSONDecodeError, KeyError):
            continue
        local_name = Path(str(metadata.get("local_file") or "")).name
        if local_name and (species_dir / local_name).exists():
            identifiers.add(identifier)
    return identifiers


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a normalized WAV file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_audio_hashes(species_dir: Path) -> set[str]:
    return {file_sha256(path) for path in species_dir.glob("*.wav")}


def append_metadata(
    metadata_path: Path,
    known_metadata_ids: set[str],
    recording: dict[str, Any],
    output_path: Path,
    duration_seconds: float,
    digest: str,
) -> None:
    """Append auditable metadata only after a WAV was fully validated."""
    identifier = recording_id(recording)
    if identifier in known_metadata_ids:
        return
    fields = (
        "id",
        "gen",
        "sp",
        "en",
        "rec",
        "cnt",
        "loc",
        "lat",
        "lng",
        "date",
        "type",
        "length",
        "q",
        "lic",
        "file",
        "file-name",
    )
    metadata = {field: recording.get(field) for field in fields}
    metadata["local_file"] = output_path.name
    metadata["duration_seconds"] = round(duration_seconds, 3)
    metadata["sha256"] = digest
    metadata["session_key"] = recording_session_key(recording)
    with metadata_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
    known_metadata_ids.add(identifier)


def append_failure(failures_path: Path, recording: dict[str, Any], error: Exception) -> None:
    """Persist failed attempts for auditability without marking them as complete."""
    failure = {
        "id": recording_id(recording),
        "country": recording.get("cnt"),
        "file": recording.get("file"),
        "error": str(error),
    }
    with failures_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(failure, ensure_ascii=False) + "\n")


def verify_standardized_wav(path: Path, min_duration_seconds: float) -> float:
    """Validate the format required by the MFCC preprocessing pipeline."""
    with wave.open(str(path), "rb") as audio:
        frame_rate = audio.getframerate()
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        frame_count = audio.getnframes()
        compression = audio.getcomptype()
    if frame_rate != 16_000 or channels != 1 or sample_width != 2 or compression != "NONE":
        raise RuntimeError(
            f"Invalid WAV format for {path.name}: {frame_rate}Hz, {channels} channel(s), "
            f"{sample_width * 8}-bit, compression={compression}."
        )
    duration_seconds = frame_count / frame_rate
    if duration_seconds < min_duration_seconds:
        raise RuntimeError(
            f"Recording {path.name} is shorter than {min_duration_seconds:.1f} seconds after conversion."
        )
    return duration_seconds


def download_and_convert(
    recording: dict[str, Any],
    species_dir: Path,
    known_hashes: set[str],
    min_duration_seconds: float,
) -> tuple[Path, float, str]:
    """Download atomically, standardize to 16 kHz mono PCM WAV, and verify it."""
    identifier = recording_id(recording)
    if not identifier:
        raise RuntimeError("Recording is missing a Xeno-canto ID.")
    wav_path = species_dir / f"XC{identifier}.wav"
    raw_path = species_dir / f".XC{identifier}.download"
    temporary_wav_path = species_dir / f".XC{identifier}.tmp.wav"
    try:
        response = request_with_retries(
            str(recording["file"]), timeout=REQUEST_TIMEOUT, stream=True
        )
        try:
            with raw_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        finally:
            response.close()
        if raw_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded an empty response for XC{identifier}.")

        kind = filetype.guess(raw_path)
        if kind is None:
            raise RuntimeError(f"Could not determine file type for XC{identifier}.")
        audio = AudioSegment.from_file(raw_path, format=kind.extension)
        if len(audio) / 1000 < min_duration_seconds:
            raise RuntimeError(
                f"XC{identifier} is shorter than {min_duration_seconds:.1f} seconds before conversion."
            )
        audio.set_frame_rate(16_000).set_channels(1).export(
            temporary_wav_path,
            format="wav",
            parameters=["-acodec", "pcm_s16le"],
        )
        temporary_wav_path.replace(wav_path)
        duration_seconds = verify_standardized_wav(wav_path, min_duration_seconds)
        digest = file_sha256(wav_path)
        if digest in known_hashes:
            wav_path.unlink(missing_ok=True)
            raise DuplicateAudioError(f"XC{identifier} is byte-identical to an existing normalized WAV.")
        known_hashes.add(digest)
        return wav_path, duration_seconds, digest
    finally:
        raw_path.unlink(missing_ok=True)
        temporary_wav_path.unlink(missing_ok=True)


def prioritize_session_diversity(
    candidates: list[dict[str, Any]], max_per_session: int
) -> list[dict[str, Any]]:
    """Prefer unique country/location/date sessions before falling back to repeats."""
    session_counts: dict[str, int] = defaultdict(int)
    preferred: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for recording in candidates:
        session_key = recording_session_key(recording)
        if session_counts[session_key] < max_per_session:
            preferred.append(recording)
            session_counts[session_key] += 1
        else:
            deferred.append(recording)
    return preferred + deferred


def download_species(
    species: str,
    output_dir: Path,
    api_key: str,
    excluded_ids: set[str],
    target_per_species: int,
    seed: int,
    max_per_session: int,
    min_duration_seconds: float,
    request_delay_seconds: float,
    quality_grades: tuple[str, ...],
    dry_run: bool,
) -> dict[str, Any]:
    """Download enough valid recordings to reach the requested final target."""
    species_dir = output_dir / species.replace(" ", "_")
    if not dry_run:
        species_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = species_dir / "metadata.jsonl"
    failures_path = species_dir / "failures.jsonl"
    existing_ids = existing_recording_ids(species_dir)
    unsafe_existing_ids = existing_ids & excluded_ids
    if unsafe_existing_ids:
        raise RuntimeError(
            f"{species}: output already contains {len(unsafe_existing_ids)} DB3V-overlapping recording ID(s)."
        )
    needed = max(0, target_per_species - len(existing_ids))
    print(f"{species}: {len(existing_ids)} validated recording(s) already present; target={target_per_species}.")
    print(f"{species}: fetching quality grades {','.join(quality_grades)} metadata.")
    recordings = fetch_recordings(species, api_key, request_delay_seconds, quality_grades)
    candidates = balanced_candidates(recordings, existing_ids | excluded_ids, seed)
    candidates = prioritize_session_diversity(candidates, max_per_session)
    candidate_countries = {str(recording.get("cnt") or "Unknown") for recording in candidates}
    candidate_sessions = {recording_session_key(recording) for recording in candidates}
    candidate_quality_counts = Counter(str(recording.get("q") or "Unknown") for recording in candidates)
    print(
        f"{species}: {len(candidates)} eligible candidate(s), {len(candidate_countries)} country/countries, "
        f"{len(candidate_sessions)} location-date session(s)."
    )

    summary: dict[str, Any] = {
        "target_recordings": target_per_species,
        "existing_recordings": len(existing_ids),
        "eligible_candidates": len(candidates),
        "eligible_countries": len(candidate_countries),
        "eligible_sessions": len(candidate_sessions),
        "eligible_quality_counts": dict(sorted(candidate_quality_counts.items())),
        "downloaded_this_run": 0,
        "failed_this_run": 0,
        "duplicate_audio_this_run": 0,
        "final_recordings": len(existing_ids),
    }
    if dry_run or needed == 0:
        if len(existing_ids) + len(candidates) < target_per_species:
            print(f"{species}: WARNING: the candidate pool cannot reach the target.")
        return summary

    known_metadata_ids = metadata_recording_ids(metadata_path)
    known_hashes = existing_audio_hashes(species_dir)
    for candidate_rank, recording in enumerate(candidates, start=1):
        if len(existing_ids) >= target_per_species:
            break
        identifier = recording_id(recording)
        try:
            output_path, duration_seconds, digest = download_and_convert(
                recording,
                species_dir,
                known_hashes,
                min_duration_seconds,
            )
            append_metadata(
                metadata_path,
                known_metadata_ids,
                recording,
                output_path,
                duration_seconds,
                digest,
            )
            existing_ids.add(identifier)
            summary["downloaded_this_run"] += 1
            if len(existing_ids) % PROGRESS_INTERVAL == 0 or len(existing_ids) == target_per_species:
                print(
                    f"{species}: accepted {len(existing_ids)}/{target_per_species} "
                    f"from candidate {candidate_rank} ({recording.get('cnt') or 'Unknown'}).",
                    flush=True,
                )
        except DuplicateAudioError as error:
            summary["duplicate_audio_this_run"] += 1
            append_failure(failures_path, recording, error)
            print(f"{species}: duplicate XC{identifier}; trying the next candidate.")
        except (CouldntDecodeError, OSError, wave.Error, requests.RequestException, RuntimeError) as error:
            summary["failed_this_run"] += 1
            append_failure(failures_path, recording, error)
            print(f"{species}: skipped XC{identifier}: {error}")
        time.sleep(request_delay_seconds)

    summary["final_recordings"] = len(existing_ids)
    if len(existing_ids) < target_per_species:
        print(f"{species}: WARNING: finished with {len(existing_ids)}/{target_per_species} recording(s).")
    return summary


def validate_dataset(
    output_dir: Path,
    species_list: tuple[str, ...],
    db3v_ids: set[str],
    min_per_species: int,
    min_duration_seconds: float,
) -> dict[str, Any]:
    """Validate counts, format, manifests, and DB3V isolation before promotion."""
    results: dict[str, Any] = {}
    errors: list[str] = []
    for species in species_list:
        species_dir = output_dir / species.replace(" ", "_")
        paths_by_id = wav_paths_by_id(species_dir)
        identifiers = set(paths_by_id)
        overlap = identifiers & db3v_ids
        metadata_ids = metadata_recording_ids(species_dir / "metadata.jsonl")
        for audio_path in paths_by_id.values():
            try:
                verify_standardized_wav(audio_path, min_duration_seconds)
            except (OSError, wave.Error, RuntimeError) as error:
                errors.append(f"{species}: invalid WAV {audio_path.name}: {error}")
        if len(identifiers) < min_per_species:
            errors.append(f"{species}: only {len(identifiers)} recordings; minimum is {min_per_species}.")
        if overlap:
            errors.append(f"{species}: {len(overlap)} recording ID(s) overlap DB3V.")
        if identifiers - metadata_ids:
            errors.append(f"{species}: {len(identifiers - metadata_ids)} WAV file(s) lack metadata.")
        results[species] = {
            "recordings": len(identifiers),
            "metadata_records": len(metadata_ids),
            "countries": len(
                {
                    json.loads(line).get("cnt")
                    for line in (species_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()
                }
            )
            if (species_dir / "metadata.jsonl").exists()
            else 0,
            "db3v_overlap_ids": sorted(overlap),
        }
    if errors:
        raise RuntimeError("Dataset validation failed:\n- " + "\n- ".join(errors))
    return results


def write_manifest(
    output_dir: Path,
    arguments: argparse.Namespace,
    species_list: tuple[str, ...],
    db3v_ids: set[str],
    legacy_overlap_ids: set[str],
    species_results: dict[str, Any],
    validation: dict[str, Any] | None = None,
) -> None:
    """Write an auditable summary that stays valid after staging promotion."""
    manifest: dict[str, Any] = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "Xeno-canto API v3",
        "quality_grades": list(arguments.quality_grades),
        "target_per_species": arguments.target_per_species,
        "minimum_per_species": arguments.min_per_species,
        "seed": arguments.seed,
        "max_per_session": arguments.max_per_session,
        "min_duration_seconds": arguments.min_duration_seconds,
        "db3v_excluded_source_ids": len(db3v_ids),
        "legacy_overlap_log_ids": len(legacy_overlap_ids),
        "species_list": list(species_list),
        "species": species_results,
    }
    if validation is not None:
        manifest["validation"] = validation
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_quality_grades(value: str) -> tuple[str, ...]:
    grades = tuple(grade.strip().upper() for grade in value.split(",") if grade.strip())
    allowed_grades = {"A", "B", "C", "D", "E"}
    if not grades or any(grade not in allowed_grades for grade in grades):
        raise argparse.ArgumentTypeError("--quality-grades must be a comma-separated subset of A,B,C,D,E.")
    return tuple(dict.fromkeys(grades))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-per-species",
        "--max-per-species",
        dest="target_per_species",
        type=int,
        default=DEFAULT_TARGET_PER_SPECIES,
        help="Final number of successful, validated recordings per species.",
    )
    parser.add_argument(
        "--min-per-species",
        type=int,
        default=300,
        help="Minimum count required for a dataset to pass final validation.",
    )
    species_group = parser.add_mutually_exclusive_group()
    species_group.add_argument(
        "--species",
        nargs="+",
        choices=DB3V_SPECIES,
        help="One or more DB3V species to crawl. Defaults to all ten species.",
    )
    species_group.add_argument(
        "--eligible-eight",
        action="store_true",
        help="Use the eight species with at least 300 all-quality candidates after DB3V exclusion.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quality-grades",
        type=parse_quality_grades,
        default=DEFAULT_QUALITY_GRADES,
        help="Comma-separated Xeno-canto quality grades to retain (default: A,B,C,D,E).",
    )
    parser.add_argument("--max-per-session", type=int, default=1)
    parser.add_argument("--min-duration-seconds", type=float, default=DEFAULT_MIN_DURATION_SECONDS)
    parser.add_argument("--request-delay-seconds", type=float, default=DEFAULT_REQUEST_DELAY_SECONDS)
    parser.add_argument("--db3v-root", type=Path, default=DEFAULT_DB3V_ROOT)
    parser.add_argument("--overlap-log", type=Path, default=DEFAULT_OVERLAP_LOG)
    parser.add_argument("--dry-run", action="store_true", help="Query and audit candidates without downloading.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow downloading when the preflight cannot reach every target; disabled by default.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    selected_species = (
        ELIGIBLE_TRAINING_SPECIES
        if arguments.eligible_eight
        else tuple(dict.fromkeys(arguments.species or DB3V_SPECIES))
    )
    if arguments.target_per_species < arguments.min_per_species:
        raise ValueError("--target-per-species must be greater than or equal to --min-per-species.")
    if arguments.min_per_species < 1 or arguments.max_per_session < 1:
        raise ValueError("--min-per-species and --max-per-session must both be at least 1.")
    if arguments.min_duration_seconds < 1.0:
        raise ValueError("--min-duration-seconds must be at least 1.0 for the MFCC pipeline.")
    if arguments.request_delay_seconds < 0:
        raise ValueError("--request-delay-seconds must not be negative.")

    api_key = os.environ.get("XENO_CANTO_API_KEY", API_KEY)
    db3v_ids = load_db3v_source_ids(arguments.db3v_root)
    legacy_overlap_ids = load_overlap_log_ids(arguments.overlap_log)
    excluded_ids = db3v_ids | legacy_overlap_ids
    print(
        f"DB3V protection: excluding {len(db3v_ids)} source ID(s) "
        f"({len(legacy_overlap_ids)} additional historical-log ID(s))."
    )
    print(f"Accepted Xeno-canto quality grades: {','.join(arguments.quality_grades)}.")

    if not arguments.dry_run and not arguments.allow_partial:
        print("Running a no-download preflight before creating the staging dataset.")
        preflight: dict[str, dict[str, Any]] = {}
        for species in selected_species:
            preflight[species] = download_species(
                species,
                arguments.output_dir,
                api_key,
                excluded_ids,
                arguments.target_per_species,
                arguments.seed,
                arguments.max_per_session,
                arguments.min_duration_seconds,
                arguments.request_delay_seconds,
                arguments.quality_grades,
                True,
            )
        shortfalls = {
            species: result
            for species, result in preflight.items()
            if result["existing_recordings"] + result["eligible_candidates"]
            < arguments.target_per_species
        }
        if shortfalls:
            details = ", ".join(
                f"{species} ({result['existing_recordings'] + result['eligible_candidates']}/"
                f"{arguments.target_per_species})"
                for species, result in shortfalls.items()
            )
            raise RuntimeError(
                "Preflight found insufficient eligible Xeno-canto recordings: "
                f"{details}. No download was started. Use --allow-partial only for an intentional "
                "partial crawl."
            )

    species_results: dict[str, Any] = {}
    for species in selected_species:
        species_results[species] = download_species(
            species,
            arguments.output_dir,
            api_key,
            excluded_ids,
            arguments.target_per_species,
            arguments.seed,
            arguments.max_per_session,
            arguments.min_duration_seconds,
            arguments.request_delay_seconds,
            arguments.quality_grades,
            arguments.dry_run,
        )

    if arguments.dry_run:
        return
    validation = validate_dataset(
        arguments.output_dir,
        selected_species,
        db3v_ids,
        arguments.min_per_species,
        arguments.min_duration_seconds,
    )
    write_manifest(
        arguments.output_dir,
        arguments,
        selected_species,
        db3v_ids,
        legacy_overlap_ids,
        species_results,
        validation,
    )
    print(f"Validated staging dataset: {arguments.output_dir}")


if __name__ == "__main__":
    main()
