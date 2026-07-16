#!/usr/bin/env python3
"""将按类别存放的音频统一转换为MFCC训练数据。

目录结构应为：

    audio_root/
        class_a/*.wav
        class_b/*.wav

脚本先在原始音频文件级别划分训练、验证和测试集合，再对每个集合
分别切片和提取MFCC，避免同一录音的相邻切片进入不同集合。
"""

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

try:
    from .acoustic_frontend import (
        CONFIG_PATH,
        config_sha256,
        extract_mfcc as extract_shared_mfcc,
        load_config,
        load_config_data,
        make_dct_matrix,
        make_mel_filterbank,
        make_periodic_hann,
    )
except ImportError:
    from acoustic_frontend import (
        CONFIG_PATH,
        config_sha256,
        extract_mfcc as extract_shared_mfcc,
        load_config,
        load_config_data,
        make_dct_matrix,
        make_mel_filterbank,
        make_periodic_hann,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_AUDIO_SUFFIXES = {".wav"}


@dataclass(frozen=True)
class AudioRecord:
    path: Path
    label: str


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def discover_audio_records(root_directory):
    if not root_directory.is_dir():
        raise FileNotFoundError(f"找不到音频目录：{root_directory}")

    records = []
    class_directories = sorted(
        path for path in root_directory.iterdir() if path.is_dir()
    )

    if not class_directories:
        raise ValueError(
            f"音频目录下没有类别子目录：{root_directory}"
        )

    for class_directory in class_directories:
        for audio_path in sorted(class_directory.rglob("*")):
            if (
                audio_path.is_file()
                and audio_path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
            ):
                records.append(
                    AudioRecord(
                        path=audio_path.resolve(),
                        label=class_directory.name,
                    )
                )

    if not records:
        raise ValueError(f"没有找到WAV音频：{root_directory}")

    return records


def calculate_split_counts(sample_count, validation_ratio, test_ratio):
    validation_count = int(sample_count * validation_ratio)
    test_count = int(sample_count * test_ratio)

    requested_splits = int(validation_ratio > 0) + int(test_ratio > 0)
    if sample_count >= requested_splits + 1:
        if validation_ratio > 0:
            validation_count = max(1, validation_count)
        if test_ratio > 0:
            test_count = max(1, test_count)

    maximum_held_out = max(0, sample_count - 1)
    while validation_count + test_count > maximum_held_out:
        if test_count >= validation_count and test_count > 0:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1

    return validation_count, test_count


def split_records(records, validation_ratio, test_ratio, seed):
    records_by_label = {}
    for record in records:
        records_by_label.setdefault(record.label, []).append(record)

    train_records = []
    validation_records = []
    test_records = []

    for label_index, label in enumerate(sorted(records_by_label)):
        label_records = list(records_by_label[label])
        label_random = random.Random(seed + label_index)
        label_random.shuffle(label_records)

        validation_count, test_count = calculate_split_counts(
            len(label_records),
            validation_ratio,
            test_ratio,
        )

        test_records.extend(label_records[:test_count])
        validation_records.extend(
            label_records[test_count:test_count + validation_count]
        )
        train_records.extend(
            label_records[test_count + validation_count:]
        )

    return {
        "train": train_records,
        "validation": validation_records,
        "test": test_records,
    }


def check_split_overlap(split_records):
    path_to_split = {}

    for split_name, records in split_records.items():
        for record in records:
            previous_split = path_to_split.get(record.path)
            if previous_split is not None:
                raise ValueError(
                    f"同一音频同时出现在{previous_split}和{split_name}中："
                    f"{record.path}"
                )
            path_to_split[record.path] = split_name


def prepare_split_records(args):
    if args.input_dir:
        if args.validation_dir or args.test_dir:
            raise ValueError(
                "使用--input-dir时不能同时使用--validation-dir或--test-dir。"
            )

        records = discover_audio_records(resolve_path(args.input_dir))
        split_data = split_records(
            records,
            args.validation_ratio,
            args.test_ratio,
            args.seed,
        )
    else:
        train_pool = discover_audio_records(resolve_path(args.train_dir))
        internal_validation_ratio = (
            0.0 if args.validation_dir else args.validation_ratio
        )
        internal_test_ratio = 0.0 if args.test_dir else args.test_ratio

        split_data = split_records(
            train_pool,
            internal_validation_ratio,
            internal_test_ratio,
            args.seed,
        )

        if args.validation_dir:
            split_data["validation"] = discover_audio_records(
                resolve_path(args.validation_dir)
            )
        if args.test_dir:
            split_data["test"] = discover_audio_records(
                resolve_path(args.test_dir)
            )

    check_split_overlap(split_data)
    return split_data


def make_label_map(split_records):
    labels = sorted(
        {
            record.label
            for records in split_records.values()
            for record in records
        }
    )

    if not labels:
        raise ValueError("数据集中没有可用类别。")

    return {label: index for index, label in enumerate(labels)}


def audio_slices(audio, clip_sample_count, drop_remainder):
    if drop_remainder:
        slice_count = len(audio) // clip_sample_count
    else:
        slice_count = math.ceil(len(audio) / clip_sample_count)

    for clip_index in range(slice_count):
        start_sample = clip_index * clip_sample_count
        unpadded_end_sample = min(
            start_sample + clip_sample_count,
            len(audio),
        )
        clip = audio[start_sample:unpadded_end_sample]
        padded_samples = clip_sample_count - len(clip)

        if padded_samples > 0:
            clip = np.pad(clip, (0, padded_samples))

        yield (
            clip_index,
            start_sample,
            unpadded_end_sample,
            padded_samples,
            clip,
        )


def extract_mfcc(
    clip,
    frontend_config,
    window,
    mel_filterbank,
    dct_matrix,
):
    feature = extract_shared_mfcc(
        clip,
        config=frontend_config,
        window=window,
        mel_filterbank=mel_filterbank,
        dct_matrix=dct_matrix,
    )
    return np.asarray(feature, dtype=np.float32)


def process_split(
    split_name,
    records,
    label_map,
    args,
    frontend_config,
    window,
    mel_filterbank,
    dct_matrix,
):
    features = []
    labels = []
    manifest_rows = []
    clip_sample_count = frontend_config.clip_sample_count

    for record in tqdm(records, desc=f"[{split_name}]", unit="file"):
        try:
            audio, _ = librosa.load(
                record.path,
                sr=frontend_config.sample_rate,
                mono=True,
            )
        except Exception as error:
            raise RuntimeError(
                f"无法读取音频：{record.path}，错误：{error}"
            ) from error

        if len(audio) == 0:
            raise ValueError(f"音频内容为空：{record.path}")

        for (
            clip_index,
            start_sample,
            end_sample,
            padded_samples,
            clip,
        ) in audio_slices(
            audio,
            clip_sample_count,
            args.drop_remainder,
        ):
            feature = extract_mfcc(
                clip,
                frontend_config,
                window,
                mel_filterbank,
                dct_matrix,
            )

            expected_shape = (
                frontend_config.number_of_frames,
                frontend_config.n_mfcc,
            )
            if feature.shape != expected_shape:
                raise RuntimeError(
                    f"MFCC形状异常：{record.path}得到{feature.shape}，"
                    f"预期{expected_shape}。"
                )

            class_id = label_map[record.label]
            features.append(feature)
            labels.append(class_id)
            manifest_rows.append(
                {
                    "split": split_name,
                    "source_file": record.path.as_posix(),
                    "class_name": record.label,
                    "class_id": class_id,
                    "clip_index": clip_index,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "padded_samples": padded_samples,
                }
            )

    if features:
        feature_array = np.stack(features).astype(np.float32, copy=False)
        label_array = np.asarray(labels, dtype=np.int64)
    else:
        feature_array = np.empty(
            (
                0,
                frontend_config.number_of_frames,
                frontend_config.n_mfcc,
            ),
            dtype=np.float32,
        )
        label_array = np.empty((0,), dtype=np.int64)

    return feature_array, label_array, manifest_rows


def save_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_manifest(path, rows):
    fieldnames = [
        "split",
        "source_file",
        "class_name",
        "class_id",
        "clip_index",
        "start_sample",
        "end_sample",
        "padded_samples",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    frontend_config_path = resolve_path(args.frontend_config)
    frontend_config_data = load_config_data(frontend_config_path)
    frontend_config = load_config(frontend_config_path)
    frontend_config_hash = config_sha256(frontend_config_data)
    window = make_periodic_hann(frontend_config)
    mel_filterbank = make_mel_filterbank(frontend_config)
    dct_matrix = make_dct_matrix(frontend_config)

    split_records_data = prepare_split_records(args)
    label_map = make_label_map(split_records_data)
    output_directory = resolve_path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    print(f"输出目录：{output_directory}")
    print(f"标签映射：{label_map}")
    print(
        "MFCC形状："
        f"({frontend_config.number_of_frames}, {frontend_config.n_mfcc})"
    )
    print(f"声学前端配置：{frontend_config_path}")
    print(f"声学前端配置SHA-256：{frontend_config_hash}")
    print(
        "原始音频数量："
        + ", ".join(
            f"{name}={len(records)}"
            for name, records in split_records_data.items()
        )
    )

    all_manifest_rows = []
    summary = {}

    for split_name in ("train", "validation", "test"):
        feature_array, label_array, manifest_rows = process_split(
            split_name,
            split_records_data[split_name],
            label_map,
            args,
            frontend_config,
            window,
            mel_filterbank,
            dct_matrix,
        )

        np.save(output_directory / f"{split_name}_data.npy", feature_array)
        np.save(output_directory / f"{split_name}_label.npy", label_array)
        all_manifest_rows.extend(manifest_rows)

        summary[split_name] = {
            "source_files": len(split_records_data[split_name]),
            "clips": len(feature_array),
            "feature_shape": list(feature_array.shape),
        }

    if summary["train"]["clips"] == 0:
        raise RuntimeError("训练集没有生成任何音频切片。")
    if summary["test"]["clips"] == 0:
        raise RuntimeError(
            "测试集为空；请增加每类原始录音数量、提供--test-dir，"
            "并确保测试音频能够生成完整或补零后的切片。"
        )

    feature_config = {
        "feature_type": "MFCC",
        "frontend_config_sha256": frontend_config_hash,
        "frontend_config": frontend_config_data,
        "drop_remainder": args.drop_remainder,
        "output_feature_shape": [
            frontend_config.number_of_frames,
            frontend_config.n_mfcc,
        ],
        "model_input_shape": list(frontend_config.model_input_shape),
        "dtype": "float32",
    }
    label_information = {
        "label_to_id": label_map,
        "id_to_label": {
            str(class_id): label
            for label, class_id in label_map.items()
        },
    }
    dataset_summary = {
        "random_seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "test_ratio": args.test_ratio,
        "number_of_classes": len(label_map),
        "splits": summary,
    }

    save_json(output_directory / "feature_config.json", feature_config)
    save_json(output_directory / "label_map.json", label_information)
    save_json(output_directory / "dataset_summary.json", dataset_summary)
    save_manifest(output_directory / "split_manifest.csv", all_manifest_rows)

    print("\nMFCC数据处理完成。")
    for split_name, split_summary in summary.items():
        print(
            f"{split_name}: {split_summary['source_files']}个原始音频，"
            f"{split_summary['clips']}个切片，"
            f"数组形状={tuple(split_summary['feature_shape'])}"
        )
    print(f"特征配置：{output_directory / 'feature_config.json'}")
    print(f"标签映射：{output_directory / 'label_map.json'}")
    print(f"切片清单：{output_directory / 'split_manifest.csv'}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "在原始音频文件级别划分数据，并统一提取MFCC训练特征。"
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--input-dir",
        help="包含全部类别音频的目录，由脚本自动划分三个集合。",
    )
    source_group.add_argument(
        "--train-dir",
        help="已经划分的训练音频目录。",
    )
    parser.add_argument(
        "--validation-dir",
        default=None,
        help="可选的独立验证音频目录；省略时从训练音频中按文件划分。",
    )
    parser.add_argument(
        "--test-dir",
        default=None,
        help="可选的独立测试音频目录；省略时从训练音频中按文件划分。",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset_processing/output/MFCC_dataset",
        help="输出.npy、标签映射和配置文件的目录。",
    )
    parser.add_argument(
        "--frontend-config",
        default=str(CONFIG_PATH),
        help="统一声学前端JSON配置；训练和STM32端必须使用相同哈希。",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
        help="未提供独立验证目录时，验证集的原始音频比例。",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="未提供独立测试目录时，测试集的原始音频比例。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--drop-remainder",
        action="store_true",
        help="丢弃不足一个完整切片的尾部；默认使用零填充。",
    )

    args = parser.parse_args()

    if args.validation_dir and not args.train_dir:
        parser.error("--validation-dir只能与--train-dir一起使用。")
    if args.test_dir and not args.train_dir:
        parser.error("--test-dir只能与--train-dir一起使用。")
    if not 0 <= args.validation_ratio < 1:
        parser.error("--validation-ratio必须在[0, 1)范围内。")
    if not 0 <= args.test_ratio < 1:
        parser.error("--test-ratio必须在[0, 1)范围内。")
    if args.validation_ratio + args.test_ratio >= 1:
        parser.error("验证集比例与测试集比例之和必须小于1。")
    return args


if __name__ == "__main__":
    run(parse_arguments())
