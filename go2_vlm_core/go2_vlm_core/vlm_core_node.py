import os
import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dotenv import load_dotenv

load_dotenv()

from go2_vlm_core.vlm_client import Qwen3VLWrapper
from go2_vlm_core.command_parser import VLMParser
from go2_interfaces.srv import ExecuteAction

VLM_BASE_URL = os.getenv("BASE_URL")
VLM_API_KEY = os.getenv("API_KEY")
VLM_MODEL = os.getenv("MODEL")


class VlmCore(Node):
    MAX_STEPS = 8
    FRAME_SETTLE_TIME = 0.5  # seconds after action before next VLM request

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

        self._loop_lock = threading.Lock()
        self._loop_active = False
        self._cancel_event = threading.Event()
        self._loop_thread: threading.Thread | None = None

        self._cb_group = ReentrantCallbackGroup()

        self._user_request_sub = self.create_subscription(
            String, "/go2_vlm/user_input", self._user_req_cb, 10,
            callback_group=self._cb_group,
        )
        self._camera_sub = self.create_subscription(
            Image, "/camera/image_raw", self._image_cb,
            QoSPresetProfiles.SENSOR_DATA.value,
            callback_group=self._cb_group,
        )
        self._vlm_response_pub = self.create_publisher(String, "/go2_vlm/vlm_response", 10)

        self._exec_client = self.create_client(
            ExecuteAction, "/go2_vlm/execute_action",
            callback_group=self._cb_group,
        )

        self.create_service(
            Trigger, "/go2_vlm/cancel",
            self._cancel_cb,
            callback_group=self._cb_group,
        )

    def _image_cb(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            _, buffer = cv2.imencode(".jpg", frame)
            self._latest_image = buffer.tobytes()
        except Exception as e:
            self.get_logger().warning(f"Failed to convert image: {e}")

    def _cancel_cb(self, request, response):
        with self._loop_lock:
            if not self._loop_active:
                response.success = False
                response.message = "No active agent loop"
                return response
            self._cancel_event.set()
        self.get_logger().warning("Cancel requested for agent loop")
        response.success = True
        response.message = "Cancel signal sent"
        return response

    def _user_req_cb(self, msg: String):
        user_input = msg.data.strip()
        if not user_input:
            return

        with self._loop_lock:
            if self._loop_active:
                self.get_logger().warning(
                    "Agent loop already running, ignoring new request"
                )
                return
            self._loop_active = True
            self._cancel_event.clear()

        self._loop_thread = threading.Thread(
            target=self._run_agent_loop, args=(user_input,), daemon=True,
        )
        self._loop_thread.start()

    def _run_agent_loop(self, user_input: str):
        self.get_logger().info(f"Agent loop start: {user_input!r}")
        self._vlm_parser.clear_history()

        try:
            prompt = user_input

            for step in range(self.MAX_STEPS):
                if self._cancel_event.is_set():
                    self.get_logger().info(f"Stopped")
                    return

                image = self._latest_image
                if image is None:
                    self.get_logger().warning("No camera frame yet, sending text-only")

                self.get_logger().info(
                    f"Step {step + 1}/{self.MAX_STEPS}: {prompt[:80]}"
                )
                parsed = self._vlm_parser.parse(user_text=prompt, image=image)

                obs = parsed.get("observation") or ""
                if obs:
                    self.get_logger().info(f"Observation: {obs}")

                mode = parsed.get("mode")

                if mode == "response":
                    text = parsed.get("response") or ""
                    self._publish_response(text)
                    self.get_logger().info(f"Task done: {text}")
                    return

                if mode == "action":
                    command = parsed["command"]
                    value = parsed["value"]
                    self.get_logger().info(
                        f"Action: {command} value={value:.2f}"
                    )
                    ok, message = self._execute_action_sync(command, value)
                    if self._cancel_event.is_set():
                        self.get_logger().info(f"Stopped")
                        return

                    if ok:
                        prompt = "Действие выполнено. Что дальше?"
                    else:
                        prompt = f"Действие провалилось: {message}. Что дальше?"
                    time.sleep(self.FRAME_SETTLE_TIME)
                    continue

                self.get_logger().error(f"Unknown mode from VLM: {mode}")
                return

            self.get_logger().warning("Agent loop hit MAX_STEPS")
        except Exception as e:
            self.get_logger().error(f"Agent loop crashed: {e}")
        finally:
            with self._loop_lock:
                self._loop_active = False
            self._cancel_event.clear()

    def _execute_action_sync(self, command: str, value: float):
        if not self._exec_client.wait_for_service(timeout_sec=1.0):
            return False, "execute_action service unavailable"

        req = ExecuteAction.Request()
        req.command = command
        req.value = float(value)

        future = self._exec_client.call_async(req)
        while rclpy.ok() and not future.done():
            if self._cancel_event.is_set():
                return False, "cancelled"
            time.sleep(0.05)

        if not future.done():
            return False, "interrupted"

        result = future.result()
        return result.success, result.message

    def _publish_response(self, text: str):
        if not text:
            return
        msg = String()
        msg.data = text
        self._vlm_response_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VlmCore()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
