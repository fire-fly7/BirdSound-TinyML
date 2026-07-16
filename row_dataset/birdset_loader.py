#!/usr/bin/env python3
"""下载并导出BirdSet音频。

训练集使用detected_events截取固定长度的焦点鸟声，输出为单标签目录，
可直接交给dataset_processing/data_sugment_MFCC.py处理。

test_5s保持BirdSet官方的5秒多标签声景格式，统一存放在
_multilabel目录，并通过metadata_test_5s.csv保存全部标签，避免将
多标签样本错误地归入某一个鸟种目录。
"""

import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path

import datasets
from datasets import load_dataset, load_dataset_builder
from packaging.version import Version
from pydub import AudioSegment


DATASET_ID = "DBD-research-group/BirdSet"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SITE_CONFIGS = {"PER", "NES", "UHH", "HSN", "NBP", "POW", "SSW", "SNE"}
TRAIN_ONLY_CONFIGS = {"XCM", "XCL"}
SUPPORTED_CONFIGS = (
    SITE_CONFIGS
    | TRAIN_ONLY_CONFIGS
    | {f"{name}_xc" for name in SITE_CONFIGS}
    | {f"{name}_scape" for name in SITE_CONFIGS}
)

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2

METADATA_FIELDS = [
    "sample_key",
    "dataset_config",
    "split",
    "label_mode",
    "original_filepath",
    "primary_label",
    "multilabels",
    "event_index",
    "event_start_time",
    "event_end_time",
    "clip_start_time",
    "clip_end_time",
    "latitude",
    "longitude",
    "quality",
    "source",
    "license",
    "recordist",
    "wav_path",
    "sample_rate",
    "channels",
]


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def safe_name(text):
    text = str(text).strip().replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]", "_", text)
    return text or "unknown"


def available_splits(config_name):
    if config_name in TRAIN_ONLY_CONFIGS or config_name.endswith("_xc"):
        return {"train"}
    if config_name.endswith("_scape"):
        return {"test", "test_5s"}
    return {"train", "test", "test_5s"}


def validate_config_and_split(config_name, split_name):
    if config_name not in SUPPORTED_CONFIGS:
        supported = ", ".join(sorted(SUPPORTED_CONFIGS))
        raise ValueError(
            f"不支持的BirdSet配置：{config_name}。可用配置：{supported}"
        )

    valid_splits = available_splits(config_name)
    if split_name not in valid_splits:
        raise ValueError(
            f"配置{config_name}不包含{split_name}，"
            f"可用split：{', '.join(sorted(valid_splits))}"
        )


def decode_label(value, names):
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    try:
        index = int(value)
    except (TypeError, ValueError):
        return str(value)

    if index < 0 or index >= len(names):
        return ""

    return names[index]


def decode_multilabel(values, names):
    if values is None:
        return []

    labels = []
    for value in values:
        label = decode_label(value, names)
        if label and label not in labels:
            labels.append(label)

    return labels


def optional_float(value):
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def normalize_detected_events(values):
    events = []
    if values is None:
        return events

    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue

        start_time = optional_float(value[0])
        end_time = optional_float(value[1])
        if (
            start_time is not None
            and end_time is not None
            and start_time >= 0
            and end_time > start_time
        ):
            events.append((start_time, end_time))

    return events


def centered_clip_bounds(
    event_start,
    event_end,
    clip_duration,
    total_duration,
):
    center_time = (event_start + event_end) / 2
    clip_start = center_time - clip_duration / 2
    clip_start = max(0.0, clip_start)
    clip_end = clip_start + clip_duration

    if clip_end > total_duration:
        clip_end = total_duration
        clip_start = max(0.0, clip_end - clip_duration)

    return clip_start, clip_end


def make_sample_key(
    config_name,
    split_name,
    filepath,
    event_index,
    clip_start,
    clip_end,
):
    raw_key = (
        f"{config_name}|{split_name}|{filepath}|{event_index}|"
        f"{clip_start:.6f}|{clip_end:.6f}"
    )
    return hashlib.sha1(raw_key.encode("utf-8")).hexdigest()


def load_existing_metadata(metadata_path):
    existing_keys = set()
    label_counts = {}

    if not metadata_path.exists():
        return existing_keys, label_counts

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        existing_fields = set(reader.fieldnames or [])
        required_fields = set(METADATA_FIELDS)

        if existing_fields != required_fields:
            raise RuntimeError(
                f"元数据格式已经更新，旧文件不能继续追加：{metadata_path}。"
                "请将旧输出目录改名备份，或使用新的--output-dir。"
            )

        for row in reader:
            sample_key = row.get("sample_key", "")
            if sample_key:
                existing_keys.add(sample_key)

            labels = []
            primary_label = row.get("primary_label", "")
            if primary_label:
                labels.append(primary_label)

            try:
                labels.extend(json.loads(row.get("multilabels", "[]")))
            except json.JSONDecodeError:
                pass

            for label in set(labels):
                label_counts[label] = label_counts.get(label, 0) + 1

    return existing_keys, label_counts


def append_metadata(metadata_path, row):
    file_exists = metadata_path.exists()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_FIELDS)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def get_audio_segment(audio_value, original_filepath):
    if isinstance(audio_value, dict):
        audio_bytes = audio_value.get("bytes")
        audio_path = audio_value.get("path")

        if audio_bytes:
            suffix = Path(original_filepath).suffix.lower().lstrip(".")
            audio_file = io.BytesIO(audio_bytes)
            if suffix:
                return AudioSegment.from_file(audio_file, format=suffix)
            return AudioSegment.from_file(audio_file)

        if audio_path:
            return AudioSegment.from_file(audio_path)

    if isinstance(audio_value, (str, Path)):
        return AudioSegment.from_file(audio_value)

    raise ValueError("BirdSet样本中没有可读取的音频路径或字节数据。")


def prepare_audio(audio_value, original_filepath):
    audio = get_audio_segment(audio_value, original_filepath)
    return (
        audio.set_frame_rate(SAMPLE_RATE)
        .set_channels(CHANNELS)
        .set_sample_width(SAMPLE_WIDTH)
    )


def slice_with_padding(audio, start_time, duration):
    start_ms = max(0, int(round(start_time * 1000)))
    duration_ms = int(round(duration * 1000))
    segment = audio[start_ms:start_ms + duration_ms]

    if len(segment) < duration_ms:
        segment += AudioSegment.silent(
            duration=duration_ms - len(segment),
            frame_rate=SAMPLE_RATE,
        )

    return segment[:duration_ms]


def save_wav(segment, wav_path):
    if wav_path.exists() and wav_path.stat().st_size > 0:
        return

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = wav_path.with_suffix(".wav.part")
    segment.export(temporary_path, format="wav")
    temporary_path.replace(wav_path)


def feature_names(feature):
    names = getattr(feature, "names", None)
    if names is not None:
        return list(names)

    nested_feature = getattr(feature, "feature", None)
    nested_names = getattr(nested_feature, "names", None)
    if nested_names is not None:
        return list(nested_names)

    return []


def get_label_names(config_name, cache_dir, revision):
    builder_arguments = {
        "path": DATASET_ID,
        "name": config_name,
        "cache_dir": str(cache_dir),
        "trust_remote_code": True,
    }
    if revision:
        builder_arguments["revision"] = revision

    builder = load_dataset_builder(**builder_arguments)
    features = builder.info.features
    primary_names = feature_names(features["ebird_code"])
    multilabel_names = feature_names(features["ebird_code_multilabel"])

    if not primary_names:
        raise RuntimeError("无法从BirdSet读取ebird_code类别名称。")
    if not multilabel_names:
        multilabel_names = primary_names

    return primary_names, multilabel_names


def save_label_information(
    output_root,
    config_name,
    primary_names,
    selected_labels,
):
    label_path = output_root / "labels.json"
    label_data = {
        "dataset_id": DATASET_ID,
        "dataset_config": config_name,
        "label_type": "ebird_code",
        "all_labels": primary_names,
        "selected_labels": sorted(selected_labels),
    }
    label_path.write_text(
        json.dumps(label_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def should_accept_sample(
    sample_labels,
    selected_labels,
    label_counts,
    max_per_label,
    include_background,
):
    matched_labels = [
        label for label in sample_labels if label in selected_labels
    ]

    if not sample_labels:
        if not include_background:
            return False, []
        matched_labels = ["background"]
    elif not matched_labels:
        return False, []

    if max_per_label is None:
        return True, matched_labels

    labels_below_limit = [
        label
        for label in matched_labels
        if label_counts.get(label, 0) < max_per_label
    ]
    return bool(labels_below_limit), labels_below_limit


def all_selected_labels_complete(selected_labels, label_counts, max_per_label):
    if max_per_label is None or not selected_labels:
        return False

    return all(
        label_counts.get(label, 0) >= max_per_label
        for label in selected_labels
    )


def build_train_segments(sample, audio, args):
    events = normalize_detected_events(sample.get("detected_events"))
    if not events:
        return []

    if args.events_per_recording is not None:
        events = events[:args.events_per_recording]

    total_duration = len(audio) / 1000
    segments = []
    for event_index, (event_start, event_end) in enumerate(events):
        clip_start, clip_end = centered_clip_bounds(
            event_start,
            event_end,
            args.train_clip_duration,
            total_duration,
        )
        segments.append(
            {
                "event_index": event_index,
                "event_start": event_start,
                "event_end": event_end,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "duration": args.train_clip_duration,
            }
        )

    return segments


def build_test_segments(sample, audio, split_name):
    total_duration = len(audio) / 1000

    if split_name == "test_5s":
        duration = 5.0
        return [
            {
                "event_index": 0,
                "event_start": None,
                "event_end": None,
                "clip_start": 0.0,
                "clip_end": min(duration, total_duration),
                "duration": duration,
            }
        ]

    event_start = optional_float(sample.get("start_time"))
    event_end = optional_float(sample.get("end_time"))
    if event_start is None or event_end is None or event_end <= event_start:
        return []

    clip_start, clip_end = centered_clip_bounds(
        event_start,
        event_end,
        5.0,
        total_duration,
    )
    return [
        {
            "event_index": 0,
            "event_start": event_start,
            "event_end": event_end,
            "clip_start": clip_start,
            "clip_end": clip_end,
            "duration": 5.0,
        }
    ]


def output_class_directory(split_name, primary_label, sample_labels):
    if split_name == "test_5s":
        return "_multilabel" if sample_labels else "_background"
    if primary_label:
        return safe_name(primary_label)
    if sample_labels:
        return safe_name(sample_labels[0])
    return "background"


def make_load_arguments(args, cache_dir):
    load_arguments = {
        "path": DATASET_ID,
        "name": args.config,
        "split": args.split,
        "streaming": True,
        "cache_dir": str(cache_dir),
        "trust_remote_code": True,
    }
    if args.revision:
        load_arguments["revision"] = args.revision
    return load_arguments


def export_birdset(args):
    if Version(datasets.__version__) > Version("3.6.0"):
        raise RuntimeError(
            "当前BirdSet数据脚本要求datasets<=3.6.0，"
            f"检测到datasets=={datasets.__version__}。"
            "请运行：python -m pip install datasets==3.6.0"
        )

    validate_config_and_split(args.config, args.split)

    requested_labels = set(args.labels or [])
    if args.config in TRAIN_ONLY_CONFIGS and not requested_labels:
        raise ValueError(
            "XCL和XCM数据量很大，必须使用--labels指定eBird代码。"
        )

    cache_dir = resolve_path(args.cache_dir)
    output_root = resolve_path(
        args.output_dir or f"row_dataset/birdset_{args.config}"
    )
    split_root = output_root / args.split
    metadata_path = output_root / f"metadata_{args.split}.csv"

    cache_dir.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)

    primary_names, multilabel_names = get_label_names(
        args.config,
        cache_dir,
        args.revision,
    )
    unknown_labels = requested_labels.difference(primary_names)
    if unknown_labels:
        raise ValueError(
            "以下eBird代码不属于当前配置："
            + ", ".join(sorted(unknown_labels))
        )

    selected_labels = requested_labels or set(primary_names)
    save_label_information(
        output_root,
        args.config,
        primary_names,
        selected_labels,
    )
    existing_keys, label_counts = load_existing_metadata(metadata_path)

    print(f"BirdSet配置：{args.config}")
    print(f"数据划分：{args.split}")
    print(f"选择类别：{len(selected_labels)}")
    print(f"输出目录：{split_root}")

    dataset = load_dataset(**make_load_arguments(args, cache_dir))
    saved_this_run = 0
    skipped_without_event = 0
    failed_samples = 0

    stop_requested = False
    for sample in dataset:
        if args.max_total is not None and saved_this_run >= args.max_total:
            break
        if all_selected_labels_complete(
            selected_labels,
            label_counts,
            args.max_per_label,
        ):
            break

        original_filepath = str(sample.get("filepath", "unknown_audio"))
        primary_label = decode_label(
            sample.get("ebird_code"),
            primary_names,
        )
        multilabels = decode_multilabel(
            sample.get("ebird_code_multilabel"),
            multilabel_names,
        )

        if args.split == "train":
            sample_labels = [primary_label] if primary_label else []
        else:
            sample_labels = list(multilabels)
            if not sample_labels and primary_label:
                sample_labels = [primary_label]

        accepted, counted_labels = should_accept_sample(
            sample_labels,
            selected_labels,
            label_counts,
            args.max_per_label,
            args.include_background,
        )
        if not accepted:
            continue

        try:
            audio = prepare_audio(sample.get("audio"), original_filepath)
            if args.split == "train":
                segment_descriptions = build_train_segments(sample, audio, args)
            else:
                segment_descriptions = build_test_segments(
                    sample,
                    audio,
                    args.split,
                )

            if not segment_descriptions:
                skipped_without_event += 1
                continue

            class_directory = output_class_directory(
                args.split,
                primary_label,
                sample_labels,
            )
            original_stem = safe_name(Path(original_filepath).stem)

            for segment_description in segment_descriptions:
                if args.max_total is not None and saved_this_run >= args.max_total:
                    stop_requested = True
                    break

                segment_counted_labels = list(counted_labels)
                if args.max_per_label is not None:
                    segment_counted_labels = [
                        label
                        for label in segment_counted_labels
                        if label_counts.get(label, 0) < args.max_per_label
                    ]
                    if not segment_counted_labels:
                        break

                sample_key = make_sample_key(
                    args.config,
                    args.split,
                    original_filepath,
                    segment_description["event_index"],
                    segment_description["clip_start"],
                    segment_description["clip_end"],
                )
                if sample_key in existing_keys:
                    continue

                output_name = f"{original_stem}_{sample_key[:12]}.wav"
                wav_path = split_root / class_directory / output_name
                segment = slice_with_padding(
                    audio,
                    segment_description["clip_start"],
                    segment_description["duration"],
                )
                save_wav(segment, wav_path)

                row = {
                    "sample_key": sample_key,
                    "dataset_config": args.config,
                    "split": args.split,
                    "label_mode": (
                        "multilabel" if args.split == "test_5s" else "single_label"
                    ),
                    "original_filepath": original_filepath,
                    "primary_label": primary_label,
                    "multilabels": json.dumps(multilabels, ensure_ascii=False),
                    "event_index": segment_description["event_index"],
                    "event_start_time": segment_description["event_start"],
                    "event_end_time": segment_description["event_end"],
                    "clip_start_time": segment_description["clip_start"],
                    "clip_end_time": segment_description["clip_end"],
                    "latitude": sample.get("lat", ""),
                    "longitude": sample.get("long", ""),
                    "quality": sample.get("quality", ""),
                    "source": sample.get("source", ""),
                    "license": sample.get("license", ""),
                    "recordist": sample.get("recordist", ""),
                    "wav_path": wav_path.relative_to(output_root).as_posix(),
                    "sample_rate": SAMPLE_RATE,
                    "channels": CHANNELS,
                }
                append_metadata(metadata_path, row)
                existing_keys.add(sample_key)

                for label in set(segment_counted_labels):
                    label_counts[label] = label_counts.get(label, 0) + 1

                saved_this_run += 1
                print(
                    f"已保存 {saved_this_run}：{wav_path.name}，"
                    f"标签={sample_labels or ['background']}"
                )

                if all_selected_labels_complete(
                    selected_labels,
                    label_counts,
                    args.max_per_label,
                ):
                    stop_requested = True
                    break

        except Exception as error:
            failed_samples += 1
            print(f"处理失败：{original_filepath}，错误：{error}")

        if stop_requested:
            break

    print("\nBirdSet导出完成。")
    print(f"本次保存：{saved_this_run}")
    print(f"无有效事件而跳过：{skipped_without_event}")
    print(f"处理失败：{failed_samples}")
    print(f"音频目录：{split_root}")
    print(f"元数据：{metadata_path}")

    if args.split == "train":
        print(
            "下一步可运行MFCC预处理，并将上述train目录传给"
            "--input-dir或--train-dir。"
        )
    elif args.split == "test_5s":
        print(
            "test_5s是多标签评价集，不能传给当前单标签MFCC训练入口；"
            "请使用metadata_test_5s.csv构建multi-hot标签。"
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "导出BirdSet训练事件或5秒多标签声景为16 kHz单声道WAV。"
        )
    )
    parser.add_argument(
        "--config",
        default="HSN",
        choices=sorted(SUPPORTED_CONFIGS),
        help="BirdSet配置；建议先使用HSN。",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test", "test_5s"],
        help="导出的官方数据划分；没有valid划分。",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="保留的eBird代码；省略时使用当前配置的全部类别。",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=200,
        help="每个类别最多导出的音频片段数量。",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=None,
        help="本次运行最多新增的音频片段总数。",
    )
    parser.add_argument(
        "--events-per-recording",
        type=int,
        default=1,
        help="训练集中每条原始录音最多导出的检测事件数。",
    )
    parser.add_argument(
        "--train-clip-duration",
        type=float,
        default=1.0,
        help="训练事件导出长度，默认与MFCC模型的一秒输入一致。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="WAV、标签和元数据输出目录。",
    )
    parser.add_argument(
        "--cache-dir",
        default="row_dataset/birdset_cache",
        help="Hugging Face数据缓存目录。",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="可选的Hugging Face数据集revision，用于固定实验版本。",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="在test_5s中保留没有鸟类标签的背景片段。",
    )

    args = parser.parse_args()

    if args.max_per_label is not None and args.max_per_label <= 0:
        parser.error("--max-per-label必须大于0。")
    if args.max_total is not None and args.max_total <= 0:
        parser.error("--max-total必须大于0。")
    if args.events_per_recording <= 0:
        parser.error("--events-per-recording必须大于0。")
    if args.train_clip_duration <= 0:
        parser.error("--train-clip-duration必须大于0。")
    if args.include_background and args.split != "test_5s":
        parser.error("--include-background只用于test_5s。")

    validate_config_and_split(args.config, args.split)
    return args


if __name__ == "__main__":
    export_birdset(parse_arguments())
