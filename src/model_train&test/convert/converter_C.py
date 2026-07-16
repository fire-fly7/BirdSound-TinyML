import argparse
import hashlib
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIRECTORY = (
    REPOSITORY_ROOT
    / "src"
    / "model_train&test"
    / "TinyML_model"
)
DEFAULT_MODEL_PATH = MODEL_DIRECTORY / "DS_CNN_Model_int8.tflite"


def resolve_path(path_text):
    path = Path(path_text)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def make_identifier(value):
    value = re.sub(r"[^A-Za-z0-9_]", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")

    if not value:
        raise ValueError("无法生成有效的C/C++变量名。")

    if value[0].isdigit():
        value = f"model_{value}"

    return value.lower()


def validate_tflite(model_path, model_data):
    if model_path.suffix.lower() != ".tflite":
        raise ValueError(f"输入文件不是.tflite模型：{model_path}")

    if len(model_data) < 8:
        raise ValueError(f"模型文件过小或已损坏：{model_path}")

    if model_data[4:8] != b"TFL3":
        raise ValueError(
            "文件不包含有效的TFLite FlatBuffer标识TFL3："
            f"{model_path}"
        )


def format_byte_array(model_data, bytes_per_line):
    lines = []

    for start in range(0, len(model_data), bytes_per_line):
        chunk = model_data[start:start + bytes_per_line]
        formatted = ", ".join(f"0x{byte:02x}" for byte in chunk)
        lines.append(f"    {formatted},")

    return "\n".join(lines)


def build_header(
    model_path,
    model_data,
    symbol_name,
    alignment,
    bytes_per_line,
):
    length_symbol = f"{symbol_name}_len"
    header_guard = f"{symbol_name.upper()}_H_"
    sha256 = hashlib.sha256(model_data).hexdigest()
    byte_array = format_byte_array(
        model_data,
        bytes_per_line,
    )

    return f"""#ifndef {header_guard}
#define {header_guard}

#include <cstddef>
#include <cstdint>

// Source model: {model_path.name}
// Model size: {len(model_data)} bytes
// SHA-256: {sha256}
alignas({alignment}) const unsigned char {symbol_name}[] = {{
{byte_array}
}};

const std::size_t {length_symbol} = sizeof({symbol_name});

#endif  // {header_guard}
"""


def convert_model(
    model_path,
    output_directory,
    requested_symbol,
    requested_header_name,
    alignment,
    bytes_per_line,
):
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到TFLite模型：{model_path}")

    model_data = model_path.read_bytes()
    validate_tflite(model_path, model_data)

    model_identifier = make_identifier(model_path.stem)
    symbol_name = (
        make_identifier(requested_symbol)
        if requested_symbol
        else f"g_{model_identifier}_data"
    )

    if requested_header_name:
        header_name = requested_header_name
        if not header_name.lower().endswith(".h"):
            header_name += ".h"
    elif requested_symbol == "g_model_data":
        header_name = "model_data.h"
    else:
        header_name = f"{model_identifier}_data.h"

    header_path = output_directory / header_name
    header_content = build_header(
        model_path,
        model_data,
        symbol_name,
        alignment,
        bytes_per_line,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_path = header_path.with_suffix(".h.part")
    temporary_path.write_text(
        header_content,
        encoding="utf-8",
        newline="\n",
    )
    temporary_path.replace(header_path)

    print(f"输入模型：{model_path}")
    print(f"模型大小：{len(model_data)} bytes")
    print(f"数组变量：{symbol_name}")
    print(f"长度变量：{symbol_name}_len")
    print(f"C++头文件：{header_path}\n")

    return header_path


def get_model_paths(args):
    if args.all:
        model_paths = sorted(MODEL_DIRECTORY.glob("*_int8.tflite"))
        if not model_paths:
            raise FileNotFoundError(
                "TinyML_model目录中没有*_int8.tflite模型。"
                "请先运行TFlite_converter.py。"
            )
        return model_paths

    return [resolve_path(args.model)]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="将TFLite模型转换为可嵌入STM32工程的C++数组头文件。"
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="需要转换的.tflite模型路径。",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="转换TinyML_model目录中的全部*_int8.tflite模型。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(MODEL_DIRECTORY),
        help="生成的头文件输出目录。",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help=(
            "指定数组变量名；单模型部署可使用g_model_data。"
            "批量转换时不能指定。"
        ),
    )
    parser.add_argument(
        "--header-name",
        default=None,
        help="指定输出头文件名；批量转换时不能指定。",
    )
    parser.add_argument(
        "--alignment",
        type=int,
        default=16,
        help="模型数组内存对齐字节数。",
    )
    parser.add_argument(
        "--bytes-per-line",
        type=int,
        default=12,
        help="生成头文件时每行包含的字节数量。",
    )

    args = parser.parse_args()

    if args.all and (args.symbol or args.header_name):
        parser.error(
            "批量转换时不能使用--symbol或--header-name。"
        )

    if args.alignment <= 0 or (
        args.alignment & (args.alignment - 1)
    ) != 0:
        parser.error("--alignment必须为正的2次幂。")

    if args.bytes_per_line <= 0:
        parser.error("--bytes-per-line必须大于0。")

    return args


def main():
    args = parse_arguments()
    output_directory = resolve_path(args.output_dir)
    model_paths = get_model_paths(args)

    for model_path in model_paths:
        convert_model(
            model_path=model_path,
            output_directory=output_directory,
            requested_symbol=args.symbol,
            requested_header_name=args.header_name,
            alignment=args.alignment,
            bytes_per_line=args.bytes_per_line,
        )


if __name__ == "__main__":
    main()
