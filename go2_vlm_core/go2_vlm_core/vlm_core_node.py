import cv2
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dotenv import load_dotenv

load_dotenv()

from go2_vlm_core.vlm_client import Qwen3VLWrapper
from go2_vlm_core.command_parser import VLMParser
from go2_interfaces.msg import VlmCommand

VLM_BASE_URL = os.getenv("BASE_URL")
VLM_API_KEY = os.getenv("API_KEY")
VLM_MODEL = os.getenv("MODEL")


class VlmCore(Node):
    def __init__(self):
        super().__init__("vlm_core_node")

        self._vlm_client = Qwen3VLWrapper(
            base_url=VLM_BASE_URL,
            api_key=VLM_API_KEY,
            model=VLM_MODEL,
        )

        self._vlm_parser = VLMParser(wrapper=self._vlm_client)
        self._bridge = CvBridge()
        self._latest_image: bytes | None = None

        self._user_request_sub = self.create_subscription(String, "/go2_vlm/user_input", self._user_req_cb, 10)
        self._camera_sub = self.create_subscription(
            Image, "/camera/image_raw", self._image_cb,
            QoSPresetProfiles.SENSOR_DATA.value,
        )
        self._vlm_response_pub = self.create_publisher(String, "/go2_vlm/vlm_response", 10)
        self._vlm_action_pub = self.create_publisher(VlmCommand, "/go2_vlm/vlm_action", 10)

    def _image_cb(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            _, buffer = cv2.imencode(".jpg", frame)
            self._latest_image = buffer.tobytes()
        except Exception as e:
            self.get_logger().warning(f"Failed to convert image: {e}")

    def _user_req_cb(self, msg: String):
        user_input = msg.data.strip()

        if not user_input:
            return

        if self._latest_image is None:
            self.get_logger().warning("No camera frame yet, sending text-only request")

        parsed_vlm_response = self._vlm_parser.parse(
            user_text=user_input,
            image=self._latest_image,
        )

        if parsed_vlm_response["mode"] == "action":
            vlm_action = VlmCommand()
            vlm_action.command = parsed_vlm_response["command"]
            vlm_action.value = parsed_vlm_response["value"]
            self.get_logger().info(f"Action: {vlm_action.command} value={vlm_action.value:.2f}")
            self._vlm_action_pub.publish(vlm_action)
        elif parsed_vlm_response["mode"] == "response":
            vlm_response = String()
            vlm_response.data = parsed_vlm_response["response"]
            self._vlm_response_pub.publish(vlm_response)


def main(args=None):
    rclpy.init(args=args)
    node = VlmCore()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

