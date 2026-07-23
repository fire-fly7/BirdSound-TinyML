"""Download the DB3V bird-vocalisation dataset from Zenodo.

Dataset DOI: https://doi.org/10.5281/zenodo.11544734
License: CC BY 4.0 (cite the dataset authors when using the data).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

import requests


DATASET_URL = "https://zenodo.org/records/11544734/files/db3v.rar?download=1"
ARCHIVE_NAME = "db3v.rar"
EXPECTED_MD5 = "fa8b368b793b54ffc98be8688dbeac0c"
CHUNK_SIZE = 1024 * 1024


def calculate_md5(path: Path) -> str:
    """Return the MD5 digest of a file without loading it all into memory."""
    digest = hashlib.md5()
    with path.open("rb") as archive:
        for chunk in iter(lambda: archive.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_db3v(output_dir: Path, force: bool = False) -> Path:
    """Download DB3V and verify its published MD5 checksum."""
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / ARCHIVE_NAME

    if archive_path.exists() and not force:
        print(f"Using existing archive: {archive_path}")
    else:
        temporary_path = archive_path.with_suffix(".rar.part")
        print("Downloading DB3V archive (approximately 1.9 GB)...")

        try:
            downloaded = temporary_path.stat().st_size if temporary_path.exists() else 0
            headers = {"Range": f"bytes={downloaded}-"} if downloaded else None
            with requests.get(
                DATASET_URL,
                headers=headers,
                stream=True,
                timeout=(15, 120),
            ) as response:
                response.raise_for_status()
                if downloaded and response.status_code != 206:
                    print("Server did not accept resume request; restarting download.")
                    downloaded = 0

                total_size = int(response.headers.get("content-length", 0)) + downloaded
                write_mode = "ab" if downloaded else "wb"

                with temporary_path.open(write_mode) as archive:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        archive.write(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            progress = downloaded * 100 / total_size
                            print(
                                f"Downloaded {downloaded / 1024 / 1024:.1f} MiB "
                                f"({progress:.1f}%)",
                                end="\r",
                                flush=True,
                            )
        except requests.RequestException as error:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(f"DB3V download failed: {error}") from error

        print()
        temporary_path.replace(archive_path)
        print(f"Archive saved to: {archive_path}")

    actual_md5 = calculate_md5(archive_path)
    if actual_md5 != EXPECTED_MD5:
        raise RuntimeError(
            "Checksum mismatch for DB3V archive. "
            f"Expected {EXPECTED_MD5}, received {actual_md5}."
        )

    print("Checksum verified.")
    return archive_path


def extract_rar(archive_path: Path, destination: Path) -> None:
    """Extract a RAR archive with an installed 7-Zip, unrar, or rar command."""
    destination.mkdir(parents=True, exist_ok=True)

    for command_name in ("7z", "7za"):
        command_path = shutil.which(command_name)
        if command_path:
            subprocess.run(
                [command_path, "x", "-y", f"-o{destination}", str(archive_path)],
                check=True,
            )
            print(f"Archive extracted to: {destination}")
            return

    for command_name in ("unrar", "rar"):
        command_path = shutil.which(command_name)
        if command_path:
            subprocess.run(
                [command_path, "x", "-o+", str(archive_path), str(destination)],
                check=True,
            )
            print(f"Archive extracted to: {destination}")
            return

    raise RuntimeError(
        "No RAR extractor was found. Install 7-Zip and add '7z' to PATH, "
        "then run the script again with --extract."
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the DB3V bird-vocalisation dataset from Zenodo."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("row_dataset") / "DB3V",
        help="Directory for db3v.rar and extracted files.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract the downloaded RAR archive using 7-Zip, unrar, or rar.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even if db3v.rar already exists.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    archive_path = download_db3v(arguments.output_dir, force=arguments.force)
    if arguments.extract:
        extract_rar(archive_path, arguments.output_dir / "extracted")


if __name__ == "__main__":
    main()
