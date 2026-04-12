import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger

from go2_interfaces.msg import VlmCommand, WebRtcReq
from go2_interfaces.srv import MoveForDuration

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

MOVEMENT_CMD = {
    "MoveForward", "MoveBackward",
    "MoveRight", "MoveLeft",
    "TurnRight", "TurnLeft",
}


class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller_node")

        self.declare_parameter("max_linear_vel", 0.5)
        self.declare_parameter("max_angular_vel", 0.5)

        self._max_linear_vel = self.get_parameter("max_linear_vel").get_parameter_value().double_value
        self._max_angular_vel = self.get_parameter("max_angular_vel").get_parameter_value().double_value

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel_out", 10)
        self._webrtc_pub = self.create_publisher(WebRtcReq, "/webrtc_req", 10)

        self._action_sub = self.create_subscription(VlmCommand, "/go2_vlm/vlm_action", self._action_cb, 10)

        self._move_srv_client = self.create_client(MoveForDuration, "/go2_vlm/move_for_duration")

    def _action_cb(self, msg: VlmCommand):
        command = msg.command
        value = msg.value

        if command in WEBRTC_CMD:
            self._handle_webrtc_cmd(command)
        elif command in MOVEMENT_CMD:
            self._handle_movement_cmd(command, value)

    def _handle_webrtc_cmd(self, command: str):
        command_api_id = WEBRTC_CMD[command]

        webrtc_req = WebRtcReq()
        webrtc_req.topic = "rt/api/sport/request"
        webrtc_req.api_id = command_api_id

        self._webrtc_pub.publish(webrtc_req)
        self.get_logger().info(f"WebRTC: command={command}, api_id={command_api_id}")

    def _handle_movement_cmd(self, command: str, value: float):
        vx, vy, vyaw = 0.0, 0.0, 0.0

        if command == "MoveForward":
            vx = self._max_linear_vel
            duration = value / self._max_linear_vel
        elif command == "MoveBackward":
            vx = -self._max_linear_vel
            duration = value / self._max_linear_vel
        elif command == "MoveRight":
            vy = -self._max_linear_vel 
            duration = value / self._max_linear_vel
        elif command == "MoveLeft":
            vy = self._max_linear_vel   
            duration = value / self._max_linear_vel
        elif command == "TurnRight":
            vyaw = -self._max_angular_vel  #
            duration = math.radians(value) / self._max_angular_vel
        elif command == "TurnLeft":
            vyaw = self._max_angular_vel  
            duration = math.radians(value) / self._max_angular_vel
        else:
            return

        self._call_move_service(vx, vy, vyaw, duration)

    def _call_move_service(self, vx: float, vy: float, vyaw: float, duration: float):
        if not self._move_srv_client.service_is_ready():
            self.get_logger().warning("move_for_duration service is not ready")
            return

        req = MoveForDuration.Request()
        req.linear_x = float(vx)
        req.linear_y = float(vy)
        req.angular_z = float(vyaw)
        req.duration = float(duration)

        self.get_logger().info(
            f"Command: vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f} dur={duration:.1f}s"
        )

        future = self._move_srv_client.call_async(req)
        future.add_done_callback(self._move_done_callback)

    def _move_done_callback(self, future):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(
                    f"Movement completed: {result.message} ({result.actual_duration:.1f}s)"
                )
            else:
                self.get_logger().error(f"Movement failed: {result.message}")
        except Exception as e:
            self.get_logger().error(f"MoveService exception: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
