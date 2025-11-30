import base64
import logging
import os
import signal
import sys

import dashscope
import pyaudio
from dashscope.audio.qwen_omni import (
    MultiModality,
    OmniRealtimeCallback,
    OmniRealtimeConversation,
)
from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams

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
asr_pcm_file = None


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
            if "conversation.item.input_audio_transcription.text" == type:
                text = response["stash"]
                print(f"[ASR] got stash result: {text}")
            if "input_audio_buffer.speech_stopped" == type:
                print("======[ASR] Speech Stop======")
            if "conversation.item.input_audio_transcription.completed" == type:
                print(f"[ASR] final recognized text: {response['transcript']}")
                print(
                    f"[Metric] session: {qwen_asr_realtime.get_session_id()}, "
                    f"first text delay: {qwen_asr_realtime.get_last_first_text_delay()}, "
                    f"first audio delay: {qwen_asr_realtime.get_last_first_audio_delay()}"
                )
        except Exception as e:
            logger.error(f"[ASR] {e}")
            return


def signal_handler(sig, frame):
    print("Ctrl+C pressed, stop conversation...")

    # Cleanup resources
    global mic_stream, asr_pya, qwen_asr_realtime, asr_pcm_file
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
    if asr_pcm_file:
        asr_pcm_file.close()
        asr_pcm_file = None

    # Forcefully exit the program
    print("Conversation stopped")
    sys.exit(0)


def main():
    logging.basicConfig(level=logging.INFO)
    init_dashscope_api_key()
    init_dify_api_key()

    logger.info("Initializing Qwen3 ASR Flash Realtime...")

    global asr_pcm_file, qwen_asr_realtime
    asr_pcm_file = open("./asr.pcm", "wb")
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

    signal.signal(signal.SIGINT, signal_handler)
    print("Press Ctrl+C to stop conversation")

    while True:
        if mic_stream:
            audio_data = mic_stream.read(3200, exception_on_overflow=False)
            asr_pcm_file.write(audio_data)
            audio_b64 = base64.b64encode(audio_data).decode("ascii")
            qwen_asr_realtime.append_audio(audio_b64)

        else:
            break


if __name__ == "__main__":
    main()
