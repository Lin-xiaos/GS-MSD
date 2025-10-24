import os
import subprocess
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def extract_audio(video_path, audio_path):
    """
    从视频文件中提取音频并保存为 WAV 文件
    """
    try:
        # 确保路径格式兼容
        video_path = f'"{video_path}"'
        audio_path = f'"{audio_path}"'

        command = f'ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 44100 -ac 2 {audio_path} -y'
        subprocess.run(command, shell=True, check=True)
        print(f"✅ 音频提取成功：{audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ 提取音频失败: {e}")


def plot_waveform(audio_path, output_image):
    """
    读取音频文件并绘制美观的波形图
    """
    try:
        # 加载音频
        y, sr = librosa.load(audio_path, sr=None)

        # 创建画布
        plt.figure(figsize=(12, 5), dpi=300)
        plt.style.use('seaborn-v0_8-darkgrid')  # 让图表更美观
        plt.style.use('dark_background')
        librosa.display.waveshow(y, sr=sr, color='#2E4EA3')
        # c6af53
        # 颜色渐变波形
        times = librosa.times_like(y, sr=sr)  # 计算时间轴
        plt.fill_between(times, y, color='#2E4EA3', alpha=0.6)

        # 添加细节优化
        plt.xlabel('Time (s)', fontsize=12, fontweight='bold')
        plt.ylabel('Amplitude', fontsize=12, fontweight='bold')
        plt.title('Audio Waveform', fontsize=14, fontweight='bold')
        plt.grid(False)

        # 保存并显示
        plt.savefig(output_image, bbox_inches='tight', facecolor='black')
        plt.show()
        print(f"✅ 波形图已保存至 {output_image}")

    except Exception as e:
        print(f"❌ 生成波形图失败: {e}")


if __name__ == "__main__":
    video_file = "/home/zxl/MultiTask classification/MultiTask-Classfication/context_final/2_53_c.mp4"
    audio_file = "output_audio.wav"
    waveform_image = "waveform.png"

    # 1. 提取音频
    extract_audio(video_file, audio_file)

    # 2. 生成美观波形图
    plot_waveform(audio_file, waveform_image)
