import math
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from go2_interfaces.msg import WebRtcReq
from go2_interfaces.srv import ExecuteAction, MoveToRelativePose

WEBRTC_CMD = {
    "Damp": 1001,
    "BalanceStand": 1002,
    "StopMove": 1003,
    "StandUp": 1004,
    "StandDown": 1005,
    "RecoveryStand": 1006,
    "Euler": 1007,
    "Move": 1008,
    "Sit": 1009,
    "RiseSit": 1010,
    "SwitchGait": 1011,
    "Trigger": 1012,
    "BodyHeight": 1013,
    "FootRaiseHeight": 1014,
    "SpeedLevel": 1015,
    "Hello": 1016,
    "Stretch": 1017,
    "TrajectoryFollow": 1018,
    "ContinuousGait": 1019,
    "Content": 1020,
    "Wallow": 1021,
    "Dance1": 1022,
    "Dance2": 1023,
    "GetBodyHeight": 1024,
    "GetFootRaiseHeight": 1025,
    "GetSpeedLevel": 1026,
    "SwitchJoystick": 1027,
    "Pose": 1028,
    "Scrape": 1029,
    "FrontFlip": 1030,
    "FrontJump": 1031,
    "FrontPounce": 1032,
    "WiggleHips": 1033,
    "GetState": 1034,
    "EconomicGait": 1035,
    "FingerHeart": 1036,
    "StandOut": 1039,
    "FreeWalk": 1045,
    "Standup": 1050,
    "CrossWalk": 1051,
    "Bound": 1304,
    "MoonWalk": 1305,
    "OnesidedStep": 1303,
    "CrossStep": 1302,
    "Handstand": 1301,
}

# command -> (dx, dy, dyaw_radians) from the value argument
MOVEMENT_OFFSET = {
    "MoveForward":  lambda v: (float(v),  0.0, 0.0),
    "MoveBackward": lambda v: (-float(v), 0.0, 0.0),
    "MoveLeft":     lambda v: (0.0,  float(v), 0.0),
    "MoveRight":    lambda v: (0.0, -float(v), 0.0),
    "TurnLeft":     lambda v: (0.0, 0.0,  math.radians(float(v))),
    "TurnRight":    lambda v: (0.0, 0.0, -math.radians(float(v))),
}

# fire-and-forget commands need a delay so the action visibly completes
# before the agent loop re-prompts the VLM
WEBRTC_SETTLE_TIME = 2.0  # seconds


class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller_node")

        self.declare_parameter("webrtc_settle_time", WEBRTC_SETTLE_TIME)
        self._webrtc_settle = self.get_parameter("webrtc_settle_time").get_parameter_value().double_value

        self._cb_group = ReentrantCallbackGroup()

        self._webrtc_pub = self.create_publisher(WebRtcReq, "/webrtc_req", 10)

        self._move_client = self.create_client(
            MoveToRelativePose, "/go2_vlm/move_to_relative_pose",
            callback_group=self._cb_group,
        )

        self.create_service(
            ExecuteAction, "/go2_vlm/execute_action",
            self._execute_action_cb,
            callback_group=self._cb_group,
        )

    def _execute_action_cb(self, request, response):
        command = request.command.strip()
        value = request.value

        if command in WEBRTC_CMD:
            return self._handle_webrtc_cmd(command, response)
        if command in MOVEMENT_OFFSET:
            return self._handle_movement_cmd(command, value, response)

        response.success = False
        response.message = f"Unknown command: {command}"
        self.get_logger().warning(response.message)
        return response

    def _handle_webrtc_cmd(self, command: str, response):
        api_id = WEBRTC_CMD[command]
        req = WebRtcReq()
        req.topic = "rt/api/sport/request"
        req.api_id = api_id
        self._webrtc_pub.publish(req)
        self.get_logger().info(f"WebRTC: {command} (api_id={api_id})")

        # blocking wait so the agent loop sees a finished action
        time.sleep(self._webrtc_settle)

        response.success = True
        response.message = f"{command} dispatched"
        return response

    def _handle_movement_cmd(self, command: str, value: float, response):
        dx, dy, dyaw = MOVEMENT_OFFSET[command](value)

        if not self._move_client.wait_for_service(timeout_sec=1.0):
            response.success = False
            response.message = "move_to_relative_pose service unavailable"
            self.get_logger().error(response.message)
            return response

        move_req = MoveToRelativePose.Request()
        move_req.dx = float(dx)
        move_req.dy = float(dy)
        move_req.dyaw = float(dyaw)
        move_req.timeout = 0.0

        self.get_logger().info(
            f"{command} -> dx={dx:.2f} dy={dy:.2f} dyaw={math.degrees(dyaw):.1f}deg"
        )

        future = self._move_client.call_async(move_req)
        while rclpy.ok() and not future.done():
            time.sleep(0.05)

        if not future.done():
            response.success = False
            response.message = "move call interrupted"
            return response

        result = future.result()
        response.success = result.success
        response.message = result.message
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()

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
