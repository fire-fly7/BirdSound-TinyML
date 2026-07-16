import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from pydub import AudioSegment
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


API_URL = "https://xeno-canto.org/api/3/recordings"
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2
DATASET_SOURCE = "xeno-canto"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR / "xeno_canto"
DATASET_MARKER = ".dataset_source.json"

METADATA_FIELDS = [
    "dataset_source",
    "recording_id",
    "requested_species",
    "scientific_name",
    "common_name",
    "country",
    "location",
    "latitude",
    "longitude",
    "sound_type",
    "quality",
    "length",
    "recording_date",
    "recording_time",
    "recordist",
    "license",
    "source_url",
    "audio_url",
    "wav_path",
    "sample_rate",
    "channels",
]


def create_session():
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "RegionalBioacousticsResearch/1.0"
        }
    )

    return session


def safe_name(text):
    text = text.strip().replace(" ", "_")
    text = re.sub(r'[<>:"/\\|?*]', "_", text)
    return text


def prepare_output_directory(output_root):
    marker_path = output_root / DATASET_MARKER

    if marker_path.exists():
        try:
            with marker_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                marker = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"无法读取数据目录标记：{marker_path}"
            ) from error

        if marker.get("dataset_source") != DATASET_SOURCE:
            raise RuntimeError(
                f"输出目录属于其他数据集：{output_root}"
            )

        return

    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(
            f"输出目录非空且没有Xeno-canto标记：{output_root}"
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with marker_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {"dataset_source": DATASET_SOURCE},
            file,
            ensure_ascii=False,
            indent=2,
        )


def normalize_url(url):
    if not url:
        return ""

    if url.startswith("//"):
        return "https:" + url

    return url


def load_existing_metadata(metadata_path):
    existing_ids = set()
    species_counts = {}

    if not metadata_path.exists():
        return existing_ids, species_counts

    with metadata_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            recording_id = row.get("recording_id", "")
            species = row.get("requested_species", "")

            if recording_id:
                existing_ids.add(recording_id)

            if species:
                species_counts[species] = (
                    species_counts.get(species, 0) + 1
                )

    return existing_ids, species_counts


def append_metadata(metadata_path, row):
    file_exists = metadata_path.exists()

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with metadata_path.open(
        "a",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=METADATA_FIELDS
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def download_audio(
    session,
    audio_url,
    output_path
):
    if output_path.exists() and output_path.stat().st_size > 0:
        return

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    response = session.get(
        audio_url,
        stream=True,
        timeout=(15, 120)
    )
    response.raise_for_status()

    with temporary_path.open("wb") as file:
        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):
            if chunk:
                file.write(chunk)

    temporary_path.replace(output_path)


def convert_to_wav(
    source_path,
    wav_path
):
    if wav_path.exists() and wav_path.stat().st_size > 0:
        return

    audio = AudioSegment.from_file(source_path)

    audio = audio.set_frame_rate(SAMPLE_RATE)
    audio = audio.set_channels(CHANNELS)
    audio = audio.set_sample_width(SAMPLE_WIDTH)

    audio.export(
        wav_path,
        format="wav"
    )


def get_source_extension(recording, audio_url):
    file_name = recording.get("file-name", "")
    extension = Path(file_name).suffix.lower()

    if not extension:
        parsed_path = urlparse(audio_url).path
        extension = Path(parsed_path).suffix.lower()

    if not extension:
        extension = ".audio"

    return extension


def build_metadata_row(
    recording,
    requested_species,
    audio_url,
    wav_path,
    output_root
):
    genus = recording.get("gen", "")
    species = recording.get("sp", "")
    scientific_name = f"{genus} {species}".strip()

    if not scientific_name:
        scientific_name = requested_species

    source_url = normalize_url(
        recording.get("url", "")
    )

    return {
        "dataset_source": DATASET_SOURCE,
        "recording_id": str(recording.get("id", "")),
        "requested_species": requested_species,
        "scientific_name": scientific_name,
        "common_name": recording.get("en", ""),
        "country": recording.get("cnt", ""),
        "location": recording.get("loc", ""),
        "latitude": recording.get("lat", ""),
        "longitude": recording.get("lng", ""),
        "sound_type": recording.get("type", ""),
        "quality": recording.get("q", ""),
        "length": recording.get("length", ""),
        "recording_date": recording.get("date", ""),
        "recording_time": recording.get("time", ""),
        "recordist": recording.get("rec", ""),
        "license": recording.get("lic", ""),
        "source_url": source_url,
        "audio_url": audio_url,
        "wav_path": wav_path.relative_to(
            output_root
        ).as_posix(),
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
    }


def download_species(
    session,
    api_key,
    scientific_name,
    output_root,
    metadata_path,
    existing_ids,
    existing_count,
    quality="C",
    max_downloads=200,
    keep_original=False,
    request_interval=0.5,
):
    species_dir = output_root / safe_name(scientific_name)
    species_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    total_saved = existing_count

    if max_downloads is not None and total_saved >= max_downloads:
        print(
            f"[{scientific_name}] 已存在 {total_saved} 条录音，"
            "不需要继续下载。"
        )
        return total_saved

    query_parts = [
        f'sp:"{scientific_name}"',
        "grp:birds",
    ]

    if quality:
        query_parts.append(f"q:{quality}")

    query = " ".join(query_parts)
    page = 1

    while True:
        if max_downloads is not None and total_saved >= max_downloads:
            break

        print(
            f"[{scientific_name}] 正在获取第 {page} 页，"
            f"当前已有 {total_saved} 条录音。"
        )

        response = session.get(
            API_URL,
            params={
                "query": query,
                "page": page,
                "key": api_key,
            },
            timeout=(15, 120),
        )
        response.raise_for_status()

        result = response.json()
        recordings = result.get("recordings", [])

        if not recordings:
            print(
                f"[{scientific_name}] 没有找到更多录音。"
            )
            break

        for recording in recordings:
            if max_downloads is not None and total_saved >= max_downloads:
                break

            recording_id = str(
                recording.get("id", "")
            )

            if not recording_id:
                continue

            if recording_id in existing_ids:
                continue

            audio_url = normalize_url(
                recording.get("file", "")
            )

            if not audio_url:
                print(
                    f"[{scientific_name}] 录音 {recording_id} "
                    "没有声音文件地址，已跳过。"
                )
                continue

            species_name = safe_name(scientific_name)
            extension = get_source_extension(
                recording,
                audio_url
            )

            source_path = species_dir / (
                f"{species_name}_{recording_id}"
                f".source{extension}"
            )

            wav_path = species_dir / (
                f"{species_name}_{recording_id}.wav"
            )

            try:
                print(
                    f"[{scientific_name}] 正在下载录音 "
                    f"{recording_id}。"
                )

                download_audio(
                    session,
                    audio_url,
                    source_path
                )

                convert_to_wav(
                    source_path,
                    wav_path
                )

                metadata_row = build_metadata_row(
                    recording=recording,
                    requested_species=scientific_name,
                    audio_url=audio_url,
                    wav_path=wav_path,
                    output_root=output_root,
                )

                append_metadata(
                    metadata_path,
                    metadata_row
                )

                existing_ids.add(recording_id)
                total_saved += 1

                if not keep_original and source_path.exists():
                    source_path.unlink()

                print(
                    f"[{scientific_name}] 已保存："
                    f"{wav_path.name}"
                )

                time.sleep(request_interval)

            except Exception as error:
                print(
                    f"[{scientific_name}] 录音 {recording_id} "
                    f"处理失败：{error}"
                )

                part_path = source_path.with_suffix(
                    source_path.suffix + ".part"
                )

                if part_path.exists():
                    part_path.unlink()

        total_pages = int(
            result.get("numPages", page)
        )

        if page >= total_pages:
            break

        page += 1

    print(
        f"[{scientific_name}] 处理完成，"
        f"当前共保存 {total_saved} 条录音。"
    )

    return total_saved


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "从Xeno-canto下载鸟类声音，并转换为"
            "16 kHz单声道WAV文件。"
        )
    )

    parser.add_argument(
        "--species",
        nargs="+",
        default=[
            "Columba livia",
            "Passer domesticus",
            "Turdus merula",
        ],
        help="需要下载的鸟类学名。",
    )

    parser.add_argument(
        "--quality",
        default="C",
        help="Xeno-canto录音质量等级，例如A、B或C。",
    )

    parser.add_argument(
        "--max-per-species",
        type=int,
        default=200,
        help="每个类别最多保存的录音数量。",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Xeno-canto数据保存目录；"
            "默认按质量等级保存到row_dataset/xeno_canto。"
        ),
    )

    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="转换为WAV后保留原始声音文件。",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    api_key = os.getenv("XENO_CANTO_API_KEY")

    if not api_key:
        raise RuntimeError(
            "没有找到XENO_CANTO_API_KEY环境变量。"
            "请先设置Xeno-canto API密钥。"
        )

    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = (
            DEFAULT_DATA_ROOT
            / f"quality_{safe_name(args.quality)}"
        )

    prepare_output_directory(output_root)

    metadata_path = output_root / "metadata.csv"

    existing_ids, species_counts = (
        load_existing_metadata(metadata_path)
    )

    session = create_session()

    for scientific_name in args.species:
        existing_count = species_counts.get(
            scientific_name,
            0
        )

        download_species(
            session=session,
            api_key=api_key,
            scientific_name=scientific_name,
            output_root=output_root,
            metadata_path=metadata_path,
            existing_ids=existing_ids,
            existing_count=existing_count,
            quality=args.quality,
            max_downloads=args.max_per_species,
            keep_original=args.keep_original,
        )


if __name__ == "__main__":
    main()
