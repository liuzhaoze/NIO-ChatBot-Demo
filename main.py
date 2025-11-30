import base64
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid

import dashscope
import pyaudio
from dashscope.audio.qwen_omni import (
    MultiModality,
    OmniRealtimeCallback,
    OmniRealtimeConversation,
)
from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams
from dashscope.audio.qwen_tts_realtime import (
    AudioFormat,
    QwenTtsRealtime,
    QwenTtsRealtimeCallback,
)
from dify_client import ChatClient

from B64PCMPlayer import B64PCMPlayer

logger = logging.getLogger(__name__)


def init_dashscope_api_key():
    if "DASHSCOPE_API_KEY" in os.environ:
        dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
    else:
        dashscope.api_key = "YOUR_API_KEY"


dify_api_key = None


def init_dify_api_key():
    global dify_api_key

    if "DIFY_API_KEY" in os.environ:
        dify_api_key = os.environ["DIFY_API_KEY"]
    else:
        dify_api_key = "YOUR_API_KEY"


asr_pya = None
mic_stream = None
qwen_asr_realtime = None


class ASRCallback(OmniRealtimeCallback):

    def on_open(self) -> None:
        global asr_pya
        global mic_stream
        logger.info("[ASR] connection opened, init microphone")
        asr_pya = pyaudio.PyAudio()
        mic_stream = asr_pya.open(
            format=pyaudio.paInt16, channels=1, rate=16000, input=True
        )

    def on_close(self, close_status_code, close_msg) -> None:
        logger.info(
            f"[ASR] connection closed with code: {close_status_code}, msg: {close_msg}, destroy microphone"
        )
        global asr_pya, mic_stream
        if mic_stream:
            mic_stream.stop_stream()
            mic_stream.close()
            mic_stream = None
        if asr_pya:
            asr_pya.terminate()
            asr_pya = None

    def on_event(self, response: str) -> None:
        try:
            global qwen_asr_realtime
            type = response["type"]
            if "session.created" == type:
                logger.info(f"[ASR] start session: {response['session']['id']}")
            if "input_audio_buffer.speech_started" == type:
                print("======[ASR] Speech Start======")
                # Interrupt current inference and TTS playback
                global b64_player, interrupted
                interrupted = True
                if b64_player:
                    b64_player.cancel_playing()
            if "conversation.item.input_audio_transcription.text" == type:
                text = response["stash"]
                print(f"[ASR] got stash result: {text}")
            if "input_audio_buffer.speech_stopped" == type:
                print("======[ASR] Speech Stop======")
            if "conversation.item.input_audio_transcription.completed" == type:
                print(f"[ASR] final recognized text: {response['transcript']}")
                # Start inference_and_speak in a new thread
                start_inference_and_speak_thread(response["transcript"])
                print(
                    f"[Metric] session: {qwen_asr_realtime.get_session_id()}, "
                    f"first text delay: {qwen_asr_realtime.get_last_first_text_delay()}, "
                    f"first audio delay: {qwen_asr_realtime.get_last_first_audio_delay()}"
                )
        except Exception as e:
            logger.error(f"[ASR] {e}")
            return


tts_pya = None
b64_player = None
qwen_tts_realtime = None


class TTSCallback(QwenTtsRealtimeCallback):

    def __init__(self):
        super().__init__()
        self.finish_event = threading.Event()

    def on_open(self) -> None:
        global tts_pya
        global b64_player
        logger.info("[TTS] connection opened, init player")
        tts_pya = pyaudio.PyAudio()
        b64_player = B64PCMPlayer(tts_pya)

    def on_close(self, close_status_code, close_msg) -> None:
        logger.info(
            f"[TTS] connection closed with code: {close_status_code}, msg: {close_msg}, destroy player"
        )
        global tts_pya
        global b64_player
        if b64_player:
            b64_player.wait_for_complete()
            b64_player.shutdown()
            b64_player = None
        if tts_pya:
            tts_pya.terminate()
            tts_pya = None

    def on_event(self, response: str) -> None:
        try:
            global qwen_tts_realtime
            global b64_player
            type = response["type"]
            if "session.created" == type:
                logger.info(f"[TTS] start session: {response['session']['id']}")
            if "response.created" == type:
                print(f"[TTS] response created")
            if "response.audio.delta" == type:
                recv_audio_b64 = response["delta"]
                b64_player.add_data(recv_audio_b64)
            if "response.done" == type:
                print(f"[TTS] response {qwen_tts_realtime.get_last_response_id()} done")
            if "session.finished" == type:
                logger.info("[TTS] session finished")
                print(
                    f"[Metric] session: {qwen_tts_realtime.get_session_id()}, "
                    f"first audio delay: {qwen_tts_realtime.get_first_audio_delay()}"
                )
                self.finish_event.set()
        except Exception as e:
            logger.error(f"[TTS] {e}")
            self.finish_event.set()
            return

    def wait_for_complete(self):
        self.finish_event.wait()


chat_client = None
user_id = None
conversation_id = None
timestamp = None
phone = None

# 用于标记是否中断当前的推理和TTS播放
interrupted = False

inference_and_speak_thread = None


def inference_and_speak(text: str):
    global qwen_tts_realtime, interrupted
    global chat_client, user_id, conversation_id, timestamp, phone

    # Reset the stop flag at the beginning
    interrupted = False

    chat_response = chat_client.create_chat_message(
        inputs={"timestamp": timestamp, "phone": phone},
        query=text,
        user=user_id,
        response_mode="streaming",
        conversation_id=conversation_id,
    )
    chat_response.raise_for_status()

    for line in chat_response.iter_lines():
        # Check if we should stop inference
        if interrupted:
            print("[Dify] Inference interrupted by user speech")
            break

        line = line.split("data:", 1)[-1]
        if not line.strip():
            continue

        line = json.loads(line.strip())
        if line.get("event") != "message":
            continue

        if conversation_id is None:
            conversation_id = line.get("conversation_id")

        answer = line.get("answer", "")
        if not answer:
            continue

        print(f"[Dify] send text: {answer}")

        # Check again before sending to TTS
        if interrupted:
            print("[Dify] TTS interrupted by user speech")
            break

        qwen_tts_realtime.append_text(answer)

    # Only finish if not interrupted
    if not interrupted:
        qwen_tts_realtime.finish()
    else:
        # If interrupted, we need to cancel the TTS response
        qwen_tts_realtime.cancel_response()


def start_inference_and_speak_thread(text: str):
    global inference_and_speak_thread, interrupted

    # 如果之前的线程还在运行，先设置中断标志
    if inference_and_speak_thread and inference_and_speak_thread.is_alive():
        interrupted = True
        # 等待之前的线程结束
        inference_and_speak_thread.join(timeout=0.1)

    # 创建并启动新线程
    inference_and_speak_thread = threading.Thread(
        target=inference_and_speak, args=(text,), daemon=True
    )
    inference_and_speak_thread.start()


def signal_handler(sig, frame):
    print("Ctrl+C pressed, stop conversation...")

    # Cleanup resources
    global mic_stream, asr_pya, qwen_asr_realtime
    if mic_stream:
        mic_stream.stop_stream()
        mic_stream.close()
        mic_stream = None
    if asr_pya:
        asr_pya.terminate()
        asr_pya = None
    if qwen_asr_realtime:
        qwen_asr_realtime.close()
        qwen_asr_realtime = None

    global qwen_tts_realtime
    if qwen_tts_realtime:
        qwen_tts_realtime.close()
        qwen_tts_realtime = None

    # Forcefully exit the program
    print("Conversation stopped")
    sys.exit(0)


def main():
    logging.basicConfig(level=logging.INFO)
    init_dashscope_api_key()
    init_dify_api_key()

    logger.info("Initializing Qwen3 ASR Flash Realtime...")

    global qwen_asr_realtime
    asr_callback = ASRCallback()
    qwen_asr_realtime = OmniRealtimeConversation(
        model="qwen3-asr-flash-realtime",
        url="wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
        callback=asr_callback,
    )

    qwen_asr_realtime.connect()
    qwen_asr_realtime.update_session(
        output_modalities=[MultiModality.TEXT],
        enable_input_audio_transcription=True,
        transcription_params=TranscriptionParams(
            language="zh",
            sample_rate=16000,
            input_audio_format="pcm",
            corpus_text="这是一段中文对话",
        ),
    )

    logger.info("Initializing Qwen3 TTS Flash Realtime...")

    global qwen_tts_realtime
    tts_callback = TTSCallback()
    qwen_tts_realtime = QwenTtsRealtime(
        model="qwen3-tts-flash-realtime",
        callback=tts_callback,
    )

    qwen_tts_realtime.connect()
    qwen_tts_realtime.update_session(
        voice="Cherry",
        response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
        mode="server_commit",
    )

    logger.info("Initializing Dify...")
    global dify_api_key, chat_client, user_id, timestamp, phone
    chat_client = ChatClient(
        api_key=dify_api_key,
        base_url="http://100.85.209.38/v1",
    )
    user_id = str(uuid.uuid4())
    timestamp = int(time.time())
    phone = "13800138000"

    signal.signal(signal.SIGINT, signal_handler)
    print("Press Ctrl+C to stop conversation")

    while True:
        if mic_stream:
            audio_data = mic_stream.read(3200, exception_on_overflow=False)
            audio_b64 = base64.b64encode(audio_data).decode("ascii")
            qwen_asr_realtime.append_audio(audio_b64)

        else:
            break


if __name__ == "__main__":
    main()
