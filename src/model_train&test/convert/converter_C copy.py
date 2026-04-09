import os

def convert_to_litert_header(model_path, output_dir):
    with open(model_path, "rb") as f:
        data = f.read()

    base_name = os.path.splitext(os.path.basename(model_path))[0]
    header_path = os.path.join(output_dir, f"model_data.h")
    
    os.makedirs(output_dir, exist_ok=True)

    with open(header_path, "w") as f:
        f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        # 增加 16 字节对齐，这是 TFLM/LiteRT 在嵌入式端加载模型的硬性要求
        f.write("alignas(16) const unsigned char g_model_data[] = {")
        
        for i, byte in enumerate(data):
            if i % 12 == 0: f.write("\n  ")
            f.write(f"0x{byte:02x}, ")
            
        f.write("\n};\n\n")
        f.write(f"const unsigned int g_model_data_len = {len(data)};\n\n")
        f.write("#endif\n")

    print(f"✅ 嵌入式头文件已生成: {header_path}")

convert_to_litert_header(
    r"src\model_train&test\LiteRT_model\BC_ResNet.tflite",
    output_dir=r"src\embedded_project\inc"
)