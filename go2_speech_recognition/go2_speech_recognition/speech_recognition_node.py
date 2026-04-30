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
    Full speech interaction pipeline:

    audio_raw -> VAD -> KWS -> STT -> /go2_vlm/user_input

    Workflow:
    1. Continuously receives microphone PCM chunks
    2. VAD detects speech segments
    3. Wake word model checks for dog name
    4. If wake word detected, full segment is transcribed
    5. Final recognized command is published
    """

    def __init__(self):
        super().__init__("speech_pipeline_node")

        wake_threshold = 0.65
        stt_model_size = "base"
        audio_topic = "/audio_raw"
        output_topic = "/go2_vlm/user_input"

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
                model_size=stt_model_size,
                language="ru",
                beam_size=1,
                best_of=1,
            )
        )

        self.get_logger().info("Initializing VAD...")
        self.vad = VADetector(
            on_speech=self.process_speech_segment,
            threshold=0.5,
            min_silence_duration_ms=600,
            speech_pad_ms=200,
        )

        self.get_logger().info("Speech pipeline node started.")

    def audio_callback(self, msg: AudioData):
        """
        Receive raw PCM audio chunks and feed them to VAD.

        Args:
            msg (AudioData): Incoming audio message
        """
        try:
            audio = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32)
            audio /= INT16_MAX

            if audio.size == 0:
                return

            self.vad.process_chunk(audio)

        except Exception as e:
            self.get_logger().error(f"Audio callback failed: {e}")

    def process_speech_segment(self, speech_audio: np.ndarray):
        """
        Handle completed speech segment from VAD.

        Pipeline:
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
                return

            text, stt_conf = self.stt.transcribe(
                command_audio,
                sample_rate=SAMPLE_RATE,
            )

            text = text.strip()

            if not text:
                self.get_logger().warning("STT returned empty text.")
                return

            self.get_logger().info(
                f"Recognized command: '{text}' (stt_conf={stt_conf:.3f})"
            )

            out_msg = String()
            out_msg.data = text
            self.text_pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"Speech segment processing failed: {e}")

    def destroy_node(self):
        """
        Cleanup before shutdown.
        """
        try:
            self.vad.flush()
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
    