import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from go2_speech_recognition.text_to_speech import PiperTTS, TTSConfig


class TTSNode(Node):
    def __init__(self):
        super().__init__("tts_node")

        config = TTSConfig(
            model_path="unitree-go2-vlm/go2_speech_recognition/weights/ru_RU-dmitri-medium.onnx",
            piper_bin="piper",
            lang="ru",
        )

        self.tts = PiperTTS(config)

        self.subscription = self.create_subscription(
            String,
            "/go2_vlm/vlm_response",
            self.callback,
            10,
        )

        self.audio_pub = self.create_publisher(
            String,
            "/speaker",
            10
        )

        self.get_logger().info("TTS Node started and listening on /go2_vlm/vlm_response")

    def callback(self, msg: String):
        text = msg.data.strip()

        if not text:
            return

        audio_bytes = self.tts.synthesize_to_bytes(text)

        audio_msg = String()
        audio_msg.data = audio_bytes.hex()

        self.audio_pub.publish(audio_msg)

    def play_audio(self, audio_bytes: bytes):
        """
        Temporary local playback (for debugging on laptop).
        Replace with ROS2 audio driver on robot.
        """
        import io
        import soundfile as sf
        import sounddevice as sd

        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        sd.play(data, sr)
        sd.wait()


def main(args=None):
    rclpy.init(args=args)

    node = TTSNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()