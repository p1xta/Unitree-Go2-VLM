import threading

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger

from go2_interfaces.srv import MoveForDuration


class MoveServiceNode(Node):
    PUBLISH_RATE = 10  # Hz

    def __init__(self):
        super().__init__("move_service_node")

        self.declare_parameter("max_linear_vel", 0.3)
        self.declare_parameter("max_angular_vel", 0.5)
        self.declare_parameter("max_duration", 10.0)

        self._max_linear_vel = self.get_parameter("max_linear_vel").get_parameter_value().double_value
        self._max_angular_vel = self.get_parameter("max_angular_vel").get_parameter_value().double_value
        self._max_duration = self.get_parameter("max_duration").get_parameter_value().double_value

        self._is_moving = False
        self._stop_requested = False
        self._lock = threading.Lock()

        self._cb_group = ReentrantCallbackGroup()

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel_out", 10)
        self._rate = self.create_rate(self.PUBLISH_RATE)

        self.create_service(
            MoveForDuration,
            "/go2_vlm/move_for_duration",
            self._move_callback,
            callback_group=self._cb_group,
        )
        self.create_service(
            Trigger,
            "/go2_vlm/stop_movement",
            self._stop_callback,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f"MoveService ready. Limits: linear={self._max_linear_vel:.1f} m/s, angular={self._max_angular_vel:.1f} rad/s, max_duration={self._max_duration:.1f} s"
        )

    def _stop_callback(self, request, response):
        with self._lock:
            self._stop_requested = True
        self._cmd_pub.publish(Twist())
        self.get_logger().warning("Emergency stop requested")
        response.success = True
        response.message = "Stop requested"
        return response

    def _move_callback(self, request, response):
        with self._lock:
            if self._is_moving:
                response.success = False
                response.message = "Already moving. Call /go2_vlm/stop_movement first."
                response.actual_duration = 0.0
                return response
            self._is_moving = True
            self._stop_requested = False

        linear_x = self._clamp(request.linear_x, -self._max_linear_vel, self._max_linear_vel)
        linear_y = self._clamp(request.linear_y, -self._max_linear_vel, self._max_linear_vel)
        angular_z = self._clamp(request.angular_z, -self._max_angular_vel, self._max_angular_vel)
        duration = self._clamp(request.duration, 0.0, self._max_duration)

        for original, clamped, name in [
            (request.linear_x, linear_x, "linear_x"),
            (request.linear_y, linear_y, "linear_y"),
            (request.angular_z, angular_z, "angular_z"),
            (request.duration, duration, "duration"),
        ]:
            if original != clamped:
                self.get_logger().warning(f"{name} clamped: {original:.2f} -> {clamped:.2f}")

        if duration <= 0:
            with self._lock:
                self._is_moving = False
            response.success = False
            response.message = "Duration must be positive"
            response.actual_duration = 0.0
            return response

        self.get_logger().info(
            f"Moving: vx={linear_x:.2f} vy={linear_y:.2f} vyaw={angular_z:.2f} dur={duration:.2f}s"
        )

        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = linear_y
        twist.angular.z = angular_z

        start_time = self.get_clock().now()
        target_duration = Duration(seconds=duration)
        stopped_early = False

        try:
            while rclpy.ok():
                with self._lock:
                    if self._stop_requested:
                        stopped_early = True
                        break
                if self.get_clock().now() - start_time >= target_duration:
                    break
                self._cmd_pub.publish(twist)
                self._rate.sleep()
        except Exception as e:
            self.get_logger().error(f"Move failed: {str(e)}")
            response.success = False
            response.message = f"Error: {e}"
            response.actual_duration = (self.get_clock().now() - start_time).nanoseconds / 1e9
            return response
        finally:
            self._cmd_pub.publish(Twist())
            with self._lock:
                self._is_moving = False

        actual_duration = (self.get_clock().now() - start_time).nanoseconds / 1e9

        if stopped_early:
            response.message = f"Stopped early after {actual_duration:.2f}s"
        else:
            response.message = f"Moved for {actual_duration:.2f}s"

        self.get_logger().info(f"Movement complete: {response.message}")
        response.success = True
        response.actual_duration = actual_duration
        return response

    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, value))


def main(args=None):
    rclpy.init(args=args)
    node = MoveServiceNode()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node._cmd_pub.publish(Twist())
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
