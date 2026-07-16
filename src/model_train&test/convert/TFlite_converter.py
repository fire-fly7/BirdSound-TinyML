import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIRECTORY = (
    REPOSITORY_ROOT
    / "src"
    / "model_train&test"
    / "TinyML_model"
)
DEFAULT_CALIBRATION_DATA = (
    REPOSITORY_ROOT
    / "dataset_processing"
    / "output"
    / "MFCC_dataset_D"
    / "train_data.npy"
)


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def get_model_paths(args):
    if args.all:
        model_paths = sorted(MODEL_DIRECTORY.glob("*.h5"))
        if not model_paths:
            raise FileNotFoundError(
                f"模型目录中没有.h5文件：{MODEL_DIRECTORY}"
            )
        return model_paths

    model_path = resolve_path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到Keras模型：{model_path}")
    return [model_path]


def load_calibration_data(data_path, input_shape):
    if not data_path.is_file():
        raise FileNotFoundError(
            "找不到代表性数据集："
            f"{data_path}\n"
            "请使用--calibration-data指定实际train_data.npy路径。"
        )

    data = np.load(data_path, mmap_mode="r")
    expected_shape = tuple(int(size) for size in input_shape)

    if tuple(data.shape[1:]) == expected_shape:
        prepared_data = data
    elif (
        expected_shape[-1] == 1
        and tuple(data.shape[1:]) == expected_shape[:-1]
    ):
        prepared_data = np.expand_dims(data, axis=-1)
    else:
        raise ValueError(
            "代表性数据形状与模型输入不一致："
            f"数据形状={data.shape}，"
            f"模型需要=(样本数, {expected_shape})"
        )

    if len(prepared_data) == 0:
        raise ValueError("代表性数据集为空。")

    return prepared_data


def build_representative_dataset(data, sample_count):
    count = min(sample_count, len(data))
    indices = np.linspace(
        0,
        len(data) - 1,
        num=count,
        dtype=np.int64,
    )

    def representative_dataset():
        for index in indices:
            sample = np.asarray(
                data[index:index + 1],
                dtype=np.float32,
            )
            yield [sample]

    return representative_dataset, indices


def convert_to_full_int8(model, representative_dataset):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    # 禁止转换器回退到Float32算子。
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.target_spec.supported_types = [tf.int8]

    # 单片机输入和输出均使用INT8。
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    return converter.convert()


def quantization_values(tensor_detail):
    scale, zero_point = tensor_detail["quantization"]
    return float(scale), int(zero_point)


def validate_full_int8(model_content, calibration_sample):
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1 or len(output_details) != 1:
        raise RuntimeError(
            "当前脚本只支持单输入、单输出模型。"
        )

    input_detail = input_details[0]
    output_detail = output_details[0]

    if input_detail["dtype"] != np.int8:
        raise RuntimeError(
            f"模型输入不是INT8：{input_detail['dtype']}"
        )

    if output_detail["dtype"] != np.int8:
        raise RuntimeError(
            f"模型输出不是INT8：{output_detail['dtype']}"
        )

    float_tensors = [
        detail["name"]
        for detail in interpreter.get_tensor_details()
        if detail["dtype"] in (np.float16, np.float32, np.float64)
    ]

    if float_tensors:
        preview = ", ".join(float_tensors[:10])
        raise RuntimeError(
            "模型中仍存在浮点张量，未完成全INT8量化："
            f"{preview}"
        )

    input_scale, input_zero_point = quantization_values(input_detail)
    output_scale, output_zero_point = quantization_values(output_detail)

    if input_scale <= 0 or output_scale <= 0:
        raise RuntimeError("模型输入或输出缺少有效量化参数。")

    # 使用一个代表性样本执行INT8推理，确认模型可以正常运行。
    quantized_input = np.rint(
        calibration_sample / input_scale + input_zero_point
    )
    quantized_input = np.clip(
        quantized_input,
        -128,
        127,
    ).astype(np.int8)

    interpreter.set_tensor(
        input_detail["index"],
        quantized_input,
    )
    interpreter.invoke()
    interpreter.get_tensor(output_detail["index"])

    return {
        "input": {
            "dtype": "int8",
            "shape": input_detail["shape"].tolist(),
            "scale": input_scale,
            "zero_point": input_zero_point,
        },
        "output": {
            "dtype": "int8",
            "shape": output_detail["shape"].tolist(),
            "scale": output_scale,
            "zero_point": output_zero_point,
        },
    }


def convert_model(model_path, args):
    print(f"\n正在加载模型：{model_path}")
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError("当前脚本只支持单输入、单输出模型。")

    input_shape = model.input_shape[1:]
    if any(size is None for size in input_shape):
        raise ValueError(
            f"模型输入形状不能包含动态维度：{model.input_shape}"
        )

    calibration_path = resolve_path(args.calibration_data)
    calibration_data = load_calibration_data(
        calibration_path,
        input_shape,
    )

    representative_dataset, indices = build_representative_dataset(
        calibration_data,
        args.calibration_samples,
    )

    print(f"模型输入形状：{model.input_shape}")
    print(f"代表性数据：{calibration_path}")
    print(f"校准样本数量：{len(indices)}")

    int8_model = convert_to_full_int8(
        model,
        representative_dataset,
    )

    first_sample = np.asarray(
        calibration_data[indices[0]:indices[0] + 1],
        dtype=np.float32,
    )
    quantization_info = validate_full_int8(
        int8_model,
        first_sample,
    )

    output_directory = resolve_path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    output_name = f"{model_path.stem}{args.suffix}.tflite"
    output_path = output_directory / output_name
    output_path.write_bytes(int8_model)

    info_path = output_path.with_suffix(".quantization.json")
    info_path.write_text(
        json.dumps(
            quantization_info,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"完整INT8模型已保存：{output_path}")
    print(f"量化参数已保存：{info_path}")
    print(
        "输入量化参数："
        f"scale={quantization_info['input']['scale']}，"
        f"zero_point={quantization_info['input']['zero_point']}"
    )
    print(
        "输出量化参数："
        f"scale={quantization_info['output']['scale']}，"
        f"zero_point={quantization_info['output']['zero_point']}"
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="将Keras .h5模型转换为完整INT8 TFLite模型。"
    )
    parser.add_argument(
        "--model",
        default=str(MODEL_DIRECTORY / "DS_CNN_Model.h5"),
        help="需要转换的.h5模型路径。",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="转换TinyML_model目录中的全部.h5模型。",
    )
    parser.add_argument(
        "--calibration-data",
        default=str(DEFAULT_CALIBRATION_DATA),
        help="用于INT8校准的训练特征train_data.npy路径。",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=200,
        help="用于量化校准的代表性样本数量。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(MODEL_DIRECTORY),
        help="转换后模型的输出目录。",
    )
    parser.add_argument(
        "--suffix",
        default="_int8",
        help="输出文件名后缀；默认保留原模型并生成*_int8.tflite。",
    )
    args = parser.parse_args()

    if args.calibration_samples <= 0:
        parser.error("--calibration-samples必须大于0。")

    return args


def main():
    args = parse_arguments()
    model_paths = get_model_paths(args)

    for model_path in model_paths:
        convert_model(model_path, args)


if __name__ == "__main__":
    main()
