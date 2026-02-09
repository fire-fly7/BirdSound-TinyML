import os
import requests
import time
import filetype
from pydub import AudioSegment

def download_xeno_canto_audio(bird_name, save_dir, max_downloads=None, q=None):
    
    base_url = "https://xeno-canto.org/api/3/recordings"
    sp = bird_name
    downloaded = 0

    save_dir = os.path.join(save_dir, f"{bird_name.replace(' ', '_')}")
# https://xeno-canto.org/api/3/recordings?query=sp:"Columba livia"+grp:birds+q:"<C"&key=0d1823264f3d866a05df71132f35b54ae10391cf
    os.makedirs(save_dir, exist_ok=True)

    while True:
        print(f"🔍 [{bird_name}] Fetching {downloaded}...")
        response = requests.get(f'{base_url}?query=sp:"{sp}"+grp:birds+q:{q}&key=0d1823264f3d866a05df71132f35b54ae10391cf')
        if response.status_code != 200:
            print(f"❌ Failed to fetch data: status code {response.status_code}")
            break

        data = response.json()
        recordings = data.get("recordings", [])
        if not recordings:
            print("✅ No more recordings found.")
            break

        for rec in recordings:
            if max_downloads and downloaded >= max_downloads:
                print(f"✅ Reached max download limit: {max_downloads}")
                return

            file_url = f"{rec['file']}"
            audio_name = f"{bird_name.replace(' ', '_')}_{rec['file-name']}"
            audio_path = os.path.join(save_dir, audio_name)

            try:
                # 下载 .mp3
                audio_data = requests.get(file_url).content
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
                print(f"✅ Downloaded: {audio_name}")
                downloaded += 1

                type_text = filetype.guess(audio_path)
                print(type_text.extension)
                if type_text.extension == 'mp3':
                    wav_name = audio_name.replace(".mp3", ".wav")
                    wav_path = os.path.join(save_dir, wav_name)

                    # 转换为 .wav 格式（16kHz 单声道）
                    sound = AudioSegment.from_mp3(audio_path)
                    sound = sound.set_frame_rate(16000).set_channels(1)
                    sound.export(wav_path, format="wav")
                    os.remove(audio_path)  # 删除原始 .mp3
                    print(f"🎧 Converted to WAV: {wav_name}")

                time.sleep(0.5)  # 避免请求过快被封 IP

            except Exception as e:
                print(f"❗ Error downloading/converting {file_url}: {e}")

        if downloaded >= max_downloads:
            print(f"✅ Reached max download limit: {max_downloads}")
            break


    print(f"🏁 Finished: {bird_name}, total downloaded: {downloaded}")

# 🐦 要下载的鸟类列表（使用学名）
bird_list = [
    "Columba livia",        # 家鸽
    "Passer domesticus",    # 麻雀
    "Turdus merula",        # 乌鸫
]

# 批量下载
for bird in bird_list:
    max = 1  # 每种鸟类下载 200 条音频
    quality = "C"  # 查询条件, 可选参数, 默认为"A"
    dir = "row_bird_dataset_" + quality
    download_xeno_canto_audio(
        bird_name=bird,
        max_downloads=max,           # 每种下载 200 条
        q = quality,               # 查询条件
        save_dir = os.path.join(f"row_dataset",dir)
    )