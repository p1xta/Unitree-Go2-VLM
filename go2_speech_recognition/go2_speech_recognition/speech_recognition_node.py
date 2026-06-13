import re

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

from go2_interfaces.msg import AudioData

from go2_speech_recognition.voice_activity_detector import VADetector
from go2_speech_recognition.name_spotting import WakeWordInference
from go2_speech_recognition.speech_to_text import SpeechToText, STTConfig


SAMPLE_RATE = 16000
INT16_MAX = 32768.0


class SpeechPipelineNode(Node):
    """
    Full speech interaction pipeline.
    audio_raw -> VAD -> KWS -> STT -> /go2_vlm/user_input
    """

    def __init__(self):
        super().__init__("speech_pipeline_node")

        wake_threshold = 0.75
        stt_model_size = "base"
        audio_topic = "/audio_raw"
        output_topic = "/go2_vlm/user_input"
        self.WAKE_WORD_PATTERN = re.compile(r"\bмарвин(?:а|у|ом|е)?\b",
                                            flags=re.IGNORECASE)

        self.audio_buffer = []
        self.last_audio_time = None
        self.buffer_gap_threshold = 1.0

        self.speech_detected = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )

        self.text_pub = self.create_publisher(String, output_topic, 10)

        self.audio_sub = self.create_subscription(
            AudioData,
            audio_topic,
            self.audio_callback,
            qos,
        )

        self.get_logger().info("Loading wake word model...")
        self.kws = WakeWordInference(threshold=wake_threshold)

        self.get_logger().info("Loading STT model...")
        self.stt = SpeechToText(
            STTConfig(
                model_path="weights/vosk-model-small-ru-0.22",
                sample_rate=16000,
            )
        )

        self.get_logger().info("Initializing VAD...")
        self.vad = VADetector(
            on_speech=self.process_speech_segment,
            threshold=0.5, # TODO: find optimal threshold
            min_silence_duration_ms=1000, # TODO: find optimal duration
            speech_pad_ms=200,
        )

        self.get_logger().info("Speech pipeline node started.")


    def audio_callback(self, msg: AudioData):
        """
        Stream audio continuously into VAD-sized chunks.

        Logic:
        - Raw audio always arrives
        - Buffer accumulates short-term audio
        - VAD decides if speech exists
        - Only speech-containing segments are preserved
        - Full phrase emitted after silence > threshold
        """
        try:
            audio = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32)
            audio /= INT16_MAX

            if audio.size == 0:
                return

            current_time = self.get_clock().now().nanoseconds / 1e9

            if self.last_audio_time is None:
                self.last_audio_time = current_time

            time_gap = current_time - self.last_audio_time
            self.last_audio_time = current_time

            if time_gap > self.buffer_gap_threshold:
                if self.audio_buffer:
                    full_audio = np.concatenate(self.audio_buffer)

                    chunk_size = 512
                    for i in range(0, len(full_audio), chunk_size):
                        chunk = full_audio[i:i + chunk_size]

                        if len(chunk) < chunk_size:
                            pad = np.zeros(
                                chunk_size - len(chunk),
                                dtype=np.float32
                            )
                            chunk = np.concatenate([chunk, pad])

                        self.vad.process_chunk(chunk)

                # Reset after phrase
                self.audio_buffer = []
                self.vad.reset()
                self.speech_detected = False

            self.audio_buffer.append(audio)

            max_buffer_seconds = 10
            max_samples = SAMPLE_RATE * max_buffer_seconds

            total_samples = sum(len(x) for x in self.audio_buffer)

            while total_samples > max_samples:
                removed = self.audio_buffer.pop(0)
                total_samples -= len(removed)

        except Exception as e:
            self.get_logger().error(f"Audio callback failed: {e}")

    def process_speech_segment(self, speech_audio: np.ndarray):
        """
        Handle completed speech segment from VAD.
        - Run wake word detection
        - If wake word exists, transcribe command
        - Publish recognized text

        Args:
            speech_audio (np.ndarray): Full detected speech segment
        """
        try:
            if speech_audio is None or len(speech_audio) == 0:
                return

            kws_result = self.kws.predict(speech_audio, sample_rate=SAMPLE_RATE)

            if not kws_result["has_wake_word"]:
                self.get_logger().debug("Wake word not detected.")
                return

            confidence = kws_result["confidence"]
            wake_index = kws_result.get("wake_word_start", 0)

            self.get_logger().info(
                f"Wake word detected (confidence={confidence:.3f})"
            )

            # Cut audio from wake word onward
            command_audio = speech_audio[wake_index:]

            if len(command_audio) < SAMPLE_RATE * 0.3:
                self.get_logger().warning("Command too short after wake word.")
                self.vad.reset()
                return

            text, stt_conf = self.stt.transcribe(
                command_audio,
                sample_rate=SAMPLE_RATE,
            )

            text = text.strip()

            if not self.WAKE_WORD_PATTERN.search(text):
                self.get_logger().warning(
                    f"Wake word not found in text: '{text}'"
                )
                return

            if not text:
                self.get_logger().warning("STT returned empty text.")
                self.vad.reset()
                return

            self.get_logger().info(
                f"Recognized command: '{text}' (stt_conf={stt_conf:.3f})"
            )

            out_msg = String()
            out_msg.data = text
            self.text_pub.publish(out_msg)

            self.get_logger().info("Command published successfully.")

            self.vad.reset()

        except Exception as e:
            self.get_logger().error(f"Speech segment processing failed: {e}")
        finally:
            self.vad.reset()

    def destroy_node(self):
        """
        Cleanup before shutdown.
        """
        try:
            self.vad.reset()
        except Exception:
            pass

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = SpeechPipelineNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
