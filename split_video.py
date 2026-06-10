import os
import pandas as pd
import subprocess
from huggingface_hub import snapshot_download

# 1. 強制讓 Hugging Face 下載到 kimo 的專案目錄底下
local_dir = "my_clean_dataset"
print("⏳ 正在從 Hugging Face 下載資料集到本地專案目錄...")
snapshot_download(repo_id="fenyying/advanced_datagen", repo_type="dataset", local_dir=local_dir)
print("✅ 下載完成！")

# 2. 開始讀取並精準切分影片
parquet_path = f"{local_dir}/data/chunk-000/file-000.parquet"
df = pd.read_parquet(parquet_path)
episodes = df.groupby('episode_index')['frame_index'].agg(['min', 'max', 'count']).reset_index()

for camel in ["observation.images.front", "observation.images.wrist"]:
    video_src = f"{local_dir}/videos/{camel}/chunk-000/file-000.mp4"
    if not os.path.exists(video_src):
        print(f"找不到 {camel} 的影片，跳過。")
        continue
            
    print(f"🎬 正在切分 {camel} 的 81 集影片...")
    for _, row in episodes.iterrows():
        ep_idx = int(row['episode_index'])
        start_frame = int(row['min'])
        duration_frames = int(row['count'])
        
        start_time = start_frame / 30.0
        duration_time = duration_frames / 30.0
        
        video_dst = f"{local_dir}/videos/{camel}/chunk-000/file-{ep_idx:03d}.mp4"
        
        cmd = [
            "ffmpeg", "-y", "-ss", str(start_time), "-i", video_src,
            "-t", str(duration_time), "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-an", video_dst
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("🎉 🎉 🎉 81 集影片全部切分完成！儲存在 my_clean_dataset 中！")