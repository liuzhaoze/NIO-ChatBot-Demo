import threading
import time
import wave
from pathlib import Path

import dashscope
import pyaudio
import pygame
import webrtcvad

# 参数设置
AUDIO_RATE = 16_000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHUNK = 1024
VAD_MODE = 3  # 0-3, 越大越敏感
NO_SPEECH_THRESHOLD = 1.5  # 停止讲话检测阈值 (秒)
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# 全局变量
recording_active = True
last_activity_time = time.time()
segments_to_save = []
last_saved_time = 0
saved_intervals = []
audio_file_count = 0

# WebRTC VAD 实例
vad_instance = webrtcvad.Vad()
vad_instance.set_mode(VAD_MODE)


def audio_recording():
    global recording_active, last_activity_time, segments_to_save, last_saved_time

    p = pyaudio.PyAudio()
    stream = p.open(
        rate=AUDIO_RATE,
        channels=AUDIO_CHANNELS,
        format=AUDIO_FORMAT,
        input=True,
        frames_per_buffer=AUDIO_CHUNK,
    )
    audio_buffer = []

    print("音频录制开始")

    while recording_active:
        data = stream.read(AUDIO_CHUNK)
        audio_buffer.append(data)

        # 每 0.5 秒执行一次 VAD
        if len(audio_buffer) * AUDIO_CHUNK / AUDIO_RATE >= 0.5:
            raw_audio = b"".join(audio_buffer)
            vad_result = vad(raw_audio)

            if vad_result:
                print("🔉 检测到语音活动")
                last_activity_time = time.time()
                segments_to_save.append((raw_audio, time.time()))
            else:
                print("🔇 无语音活动")

            audio_buffer = []

        # 停止讲话后保存音频
        if time.time() - last_activity_time > NO_SPEECH_THRESHOLD:
            if segments_to_save and segments_to_save[-1][1] > last_saved_time:
                save_audio()
                last_activity_time = time.time()

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("音频录制结束")


def vad(audio_data: bytes, threshold: float = 0.4) -> bool:
    """
    对音频数据进行分块检测

    有效音频块占比超过 40% 则认为有语音活动
    """
    n_activate = 0
    n_step = int(AUDIO_RATE * 0.02)  # 20ms 为一个音频块
    n_threshold = round(threshold * len(audio_data) // n_step)

    for i in range(0, len(audio_data), n_step):
        chunk = audio_data[i : i + n_step]
        if len(chunk) != n_step:
            break
        if vad_instance.is_speech(chunk, sample_rate=AUDIO_RATE):
            n_activate += 1

    return n_activate >= n_threshold


def save_audio():
    global segments_to_save, last_saved_time, saved_intervals, audio_file_count

    pygame.mixer.init()
    audio_file_count += 1
    audio_path = CACHE_DIR / f"audio_{audio_file_count}.wav"

    if not segments_to_save:
        return

    # 接收到新的音频保存调用时，停止当前音频播放
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
        print("检测到新的语音输入，已停止当前音频播放")

    # 获取有效音频段的时间范围
    start_time = segments_to_save[0][1]
    end_time = segments_to_save[-1][1]

    # 检查是否与已保存的时间段重叠
    if saved_intervals and saved_intervals[-1][1] >= start_time:
        print("当前音频片段与已保存片段时间重叠，跳过保存")
        segments_to_save.clear()
        return

    # 保存音频文件
    audio_frames = [seg[0] for seg in segments_to_save]
    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(AUDIO_RATE)
        wf.writeframes(b"".join(audio_frames))
    print(f"音频文件保存至: {audio_path}")

    inference_thread = threading.Thread(target=inference, args=(audio_path,))
    inference_thread.start()

    # 记录已保存的时间段
    saved_intervals.append((start_time, end_time))

    # 清空缓冲区
    segments_to_save.clear()


dashscope.base_http_api_url = "https://dashscope.aliyuncs.com/api/v1"
DASHSCOPE_API_KEY = "sk-de74ecf6058e4399a663e45b9fa8c17f"


def inference(audio_file: Path):
    # Qwen3-ASR
    response = dashscope.MultiModalConversation.call(
        api_key=DASHSCOPE_API_KEY,
        model="qwen3-asr-flash",
        messages=[{"role": "user", "content": [{"audio": f"file://{audio_file}"}]}],
        result_format="message",
        asr_options={"enable_itn": True},
    )
    content = response.output.choices[0].message.content

    if content:
        user_prompt = content[0]["text"]
    else:
        return
    print(f"💬 ASR: {user_prompt}")


def main():
    global recording_active

    try:
        print("输入 Ctrl+C 退出程序")

        audio_thread = threading.Thread(target=audio_recording)
        audio_thread.start()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("正在退出...")
        recording_active = False
        audio_thread.join()
        print("程序已退出")


if __name__ == "__main__":
    main()
