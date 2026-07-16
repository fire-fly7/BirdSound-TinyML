import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import datasets
import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset, load_dataset_builder
from packaging.version import Version


DATASET_ID = "DBD-research-group/BirdSet"
SAMPLE_RATE = 16000
DATASET_SOURCE = "birdset"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR / "birdset"
DEFAULT_CACHE_DIR = DEFAULT_DATA_ROOT / "cache"
DATASET_MARKER = ".dataset_source.json"

REGIONAL_SITES = {
    "PER", "NES", "UHH", "HSN",
    "NBP", "POW", "SSW", "SNE",
}

SUPPORTED_CONFIGS = {
    f"{name}_scape"
    for name in REGIONAL_SITES
}

METADATA_FIELDS = [
    "dataset_source",
    "sample_key",
    "dataset_config",
    "split",
    "original_filepath",
    "primary_label",
    "multilabels",
    "start_time",
    "end_time",
    "latitude",
    "longitude",
    "quality",
    "microphone",
    "license",
    "source",
    "recordist",
    "wav_path",
    "sample_rate",
]


def safe_name(value):
    value = str(value).strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    return value or "unknown"


def prepare_output_directory(output_root, config):
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

        if marker.get("dataset_config") != config:
            raise RuntimeError(
                f"输出目录属于其他BirdSet配置：{output_root}"
            )

        return

    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(
            f"输出目录非空且没有BirdSet标记：{output_root}"
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
            {
                "dataset_source": DATASET_SOURCE,
                "dataset_config": config,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


def paths_overlap(first_path, second_path):
    first_path = first_path.resolve()
    second_path = second_path.resolve()

    return (
        first_path == second_path
        or first_path in second_path.parents
        or second_path in first_path.parents
    )


def decode_label(value, label_names):
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    try:
        index = int(value)
    except (TypeError, ValueError):
        return ""

    if 0 <= index < len(label_names):
        return label_names[index]

    return ""


def decode_multilabel(values, label_names):
    if values is None:
        return []

    labels = []

    for value in values:
        label = decode_label(value, label_names)

        if label and label not in labels:
            labels.append(label)

    return labels


def make_sample_key(config, split, filepath, start_time, end_time):
    content = (
        f"{config}|{split}|{filepath}|"
        f"{start_time}|{end_time}"
    )

    return hashlib.sha1(
        content.encode("utf-8")
    ).hexdigest()


def get_label_names(config, cache_dir):
    builder = load_dataset_builder(
        DATASET_ID,
        config,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
    )

    features = builder.info.features

    primary_names = list(
        features["ebird_code"].names
    )

    multilabel_feature = features.get(
        "ebird_code_multilabel"
    )

    if multilabel_feature is not None:
        multilabel_names = list(
            multilabel_feature.feature.names
        )
    else:
        multilabel_names = primary_names

    return primary_names, multilabel_names


def load_existing_metadata(metadata_path):
    existing_keys = set()
    label_counts = {}

    if not metadata_path.exists():
        return existing_keys, label_counts

    with metadata_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            sample_key = row.get("sample_key", "")

            if sample_key:
                existing_keys.add(sample_key)

            labels = []

            primary_label = row.get(
                "primary_label",
                "",
            )

            if primary_label:
                labels.append(primary_label)

            try:
                multilabels = json.loads(
                    row.get("multilabels", "[]")
                )
                labels.extend(multilabels)
            except json.JSONDecodeError:
                pass

            for label in set(labels):
                label_counts[label] = (
                    label_counts.get(label, 0) + 1
                )

    return existing_keys, label_counts


def append_metadata(metadata_path, row):
    file_exists = metadata_path.exists()

    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with metadata_path.open(
        "a",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=METADATA_FIELDS,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_labels(
    output_root,
    config,
    all_labels,
    selected_labels,
):
    label_information = {
        "dataset_source": DATASET_SOURCE,
        "dataset_id": DATASET_ID,
        "dataset_config": config,
        "label_type": "ebird_code",
        "all_labels": all_labels,
        "selected_labels": sorted(selected_labels),
    }

    label_path = output_root / "labels.json"

    with label_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            label_information,
            file,
            ensure_ascii=False,
            indent=2,
        )


def save_wav(audio_data, wav_path):
    audio_array = np.asarray(
        audio_data["array"],
        dtype=np.float32,
    )

    if audio_array.ndim > 1:
        audio_array = np.mean(
            audio_array,
            axis=0,
        )

    wav_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = wav_path.with_suffix(
        ".wav.part"
    )

    sf.write(
        temporary_path,
        audio_array,
        SAMPLE_RATE,
        subtype="PCM_16",
        format="WAV",
    )

    temporary_path.replace(wav_path)


def check_split(config, split):
    if split == "train":
        raise ValueError(
            f"{config}只读取声景测试数据，"
            "不支持train划分。"
        )


def export_birdset(args):
    if Version(datasets.__version__) >= Version("4.0.0"):
        raise RuntimeError(
            "当前BirdSet需要datasets 4.0以下版本，"
            "建议安装datasets==3.6.0。"
        )

    if args.config not in SUPPORTED_CONFIGS:
        raise ValueError(
            f"不支持的BirdSet配置：{args.config}"
        )

    check_split(args.config, args.split)

    requested_labels = set(args.labels or [])

    cache_dir = Path(args.cache_dir)

    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = DEFAULT_DATA_ROOT / args.config

    if paths_overlap(output_root, cache_dir):
        raise ValueError(
            "BirdSet输出目录和缓存目录不能相同或相互包含。"
        )

    prepare_output_directory(
        output_root,
        args.config,
    )

    audio_root = output_root / args.split
    metadata_path = (
        output_root
        / f"metadata_{args.split}.csv"
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary_names, multilabel_names = (
        get_label_names(
            args.config,
            cache_dir,
        )
    )

    unknown_labels = (
        requested_labels
        - set(primary_names)
    )

    if unknown_labels:
        raise ValueError(
            "以下eBird代码不属于当前配置："
            + ", ".join(
                sorted(unknown_labels)
            )
        )

    selected_labels = (
        requested_labels
        or set(primary_names)
    )

    save_labels(
        output_root,
        args.config,
        primary_names,
        selected_labels,
    )

    existing_keys, label_counts = (
        load_existing_metadata(
            metadata_path
        )
    )

    max_per_label = args.max_per_label

    if max_per_label <= 0:
        max_per_label = None

    print(f"BirdSet配置：{args.config}")
    print(f"数据划分：{args.split}")
    print(f"类别数量：{len(selected_labels)}")

    dataset = load_dataset(
        DATASET_ID,
        args.config,
        split=args.split,
        streaming=True,
        cache_dir=str(cache_dir),
        trust_remote_code=True,
    )

    dataset = dataset.cast_column(
        "audio",
        Audio(
            sampling_rate=SAMPLE_RATE,
            mono=True,
            decode=True,
        ),
    )

    saved_count = 0

    for sample in dataset:
        if (
            args.max_total is not None
            and saved_count >= args.max_total
        ):
            break

        filepath = str(
            sample.get(
                "filepath",
                "unknown_audio",
            )
        )

        start_time = sample.get(
            "start_time"
        )

        end_time = sample.get(
            "end_time"
        )

        sample_key = make_sample_key(
            args.config,
            args.split,
            filepath,
            start_time,
            end_time,
        )

        if sample_key in existing_keys:
            continue

        primary_label = decode_label(
            sample.get("ebird_code"),
            primary_names,
        )

        multilabels = decode_multilabel(
            sample.get(
                "ebird_code_multilabel"
            ),
            multilabel_names,
        )

        sample_labels = []

        if primary_label:
            sample_labels.append(
                primary_label
            )

        for label in multilabels:
            if label not in sample_labels:
                sample_labels.append(label)

        matched_labels = [
            label
            for label in sample_labels
            if label in selected_labels
        ]

        if not matched_labels:
            if not (
                args.include_background
                and not sample_labels
            ):
                continue

            matched_labels = ["background"]

        if max_per_label is not None:
            available_labels = [
                label
                for label in matched_labels
                if label_counts.get(label, 0)
                < max_per_label
            ]

            if not available_labels:
                continue

            counted_labels = available_labels
        else:
            counted_labels = matched_labels

        class_name = safe_name(
            counted_labels[0]
        )

        original_stem = safe_name(
            Path(filepath).stem
        )

        wav_name = (
            f"{original_stem}_"
            f"{sample_key[:12]}.wav"
        )

        wav_path = (
            audio_root
            / class_name
            / wav_name
        )

        try:
            save_wav(
                sample["audio"],
                wav_path,
            )

            metadata_row = {
                "dataset_source": DATASET_SOURCE,
                "sample_key": sample_key,
                "dataset_config": args.config,
                "split": args.split,
                "original_filepath": filepath,
                "primary_label": primary_label,
                "multilabels": json.dumps(
                    multilabels,
                    ensure_ascii=False,
                ),
                "start_time": start_time,
                "end_time": end_time,
                "latitude": sample.get(
                    "lat",
                    "",
                ),
                "longitude": sample.get(
                    "long",
                    "",
                ),
                "quality": sample.get(
                    "quality",
                    "",
                ),
                "microphone": sample.get(
                    "microphone",
                    "",
                ),
                "license": sample.get(
                    "license",
                    "",
                ),
                "source": sample.get(
                    "source",
                    "",
                ),
                "recordist": sample.get(
                    "recordist",
                    "",
                ),
                "wav_path": wav_path.relative_to(
                    output_root
                ).as_posix(),
                "sample_rate": SAMPLE_RATE,
            }

            append_metadata(
                metadata_path,
                metadata_row,
            )

            existing_keys.add(sample_key)

            for label in set(counted_labels):
                label_counts[label] = (
                    label_counts.get(label, 0)
                    + 1
                )

            saved_count += 1

            print(
                f"已保存{saved_count}："
                f"{wav_path.name}，"
                f"标签={matched_labels}"
            )

        except Exception as error:
            print(
                f"处理失败：{filepath}，"
                f"错误：{error}"
            )

    print(
        f"本次共保存{saved_count}个样本。"
    )
    print(f"音频目录：{audio_root}")
    print(f"元数据文件：{metadata_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "从BirdSet声景数据筛选并导出"
            "16 kHz单声道WAV数据。"
        )
    )

    parser.add_argument(
        "--config",
        default="HSN_scape",
        help=(
            "BirdSet声景配置，例如"
            "HSN_scape或PER_scape。"
        ),
    )

    parser.add_argument(
        "--split",
        default="test_5s",
        choices=[
            "test_5s",
            "test",
        ],
        help="需要导出的BirdSet数据划分。",
    )

    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help=(
            "需要保留的eBird代码；"
            "不填写时使用配置中的全部类别。"
        ),
    )

    parser.add_argument(
        "--max-per-label",
        type=int,
        default=200,
        help=(
            "每个类别最多保存的样本数；"
            "设置为0表示不限制。"
        ),
    )

    parser.add_argument(
        "--max-total",
        type=int,
        default=None,
        help="本次运行最多新增的样本数。",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "BirdSet WAV音频和元数据输出目录；"
            "默认保存到row_dataset/birdset/<配置名>。"
        ),
    )

    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=(
            "BirdSet专用Hugging Face缓存目录；"
            "默认保存到row_dataset/birdset/cache。"
        ),
    )

    parser.add_argument(
        "--include-background",
        action="store_true",
        help="保留test_5s中的无标签背景片段。",
    )

    return parser.parse_args()


if __name__ == "__main__":
    export_birdset(
        parse_arguments()
    )
