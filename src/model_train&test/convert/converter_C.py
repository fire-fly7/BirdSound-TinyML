import os

def convert_tflite_to_c_array(tflite_model_path, output_dir=None):

    # 1. 读取 tflite 文件
    with open(tflite_model_path, "rb") as f:
        tflite_data = f.read()

    # 2. 自动生成名称
    base_name = os.path.splitext(os.path.basename(tflite_model_path))[0]

    array_name = f"{base_name.lower()}_data"
    header_guard = array_name.upper() + "_H"

    # 3. 设置输出目录
    if output_dir is None:
        output_dir = os.getcwd()

    os.makedirs(output_dir, exist_ok=True)

    output_header_path = os.path.join(output_dir, f"{array_name}.h")

    # 4. 写入头文件
    with open(output_header_path, "w") as f:
        f.write(f'#ifndef {header_guard}\n')
        f.write(f'#define {header_guard}\n\n')

        f.write('#include <stdint.h>\n')
        f.write('#include <stddef.h>\n\n')
        f.write('alignas(16) const unsigned char model_data[] = {\n')


        for i, byte in enumerate(tflite_data):
            if i % 12 == 0:
                f.write('\n ')
            f.write(f' 0x{byte:02x},')

        f.write('\n};\n\n')
        f.write(f'const unsigned int model_data_len = {len(tflite_data)};\n')

        f.write(f'\n#endif // {header_guard}\n')

    print(f"✅ 模型已转换为 C 数组头文件：{output_header_path}")

convert_tflite_to_c_array(
    "src\model_train&test\TinyML_model\MobileNetV2.tflite",
    output_dir=r"src\model_train&test\TinyML_model"
)