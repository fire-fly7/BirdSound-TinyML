def convert_tflite_to_c_array(tflite_model_path, output_header_path, array_name="model_data"):
    with open(tflite_model_path, "rb") as f:
        tflite_data = f.read()

    with open(output_header_path, "w") as f:
        f.write(f'#ifndef {array_name.upper()}_H\n#define {array_name.upper()}_H\n\n')
        f.write(f'unsigned char {array_name}[] = {{\n')

        for i, byte in enumerate(tflite_data):
            if i % 12 == 0:
                f.write('\n ')
            f.write(f' 0x{byte:02x},')

        f.write('\n};\n')
        f.write(f'unsigned int {array_name}_len = {len(tflite_data)};\n')
        f.write(f'\n#endif // {array_name.upper()}_H\n')

    print(f"✅ 模型已保存为 C 数组头文件：{output_header_path}")


convert_tflite_to_c_array("DS_CNN_Model.tflite", "ds_cnn_model_data.h", array_name="ds_cnn_model_quant")
