import math
import threading

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger

from go2_interfaces.srv import MoveForDuration, MoveToRelativePose


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class MoveServiceNode(Node):
    PUBLISH_RATE = 20  # Hz, control loop frequency

    def __init__(self):
        super().__init__("move_service_node")

        # legacy MoveForDuration limits
        self.declare_parameter("max_linear_vel", 0.3)
        self.declare_parameter("max_angular_vel", 0.5)
        self.declare_parameter("max_duration", 10.0)

        # MoveToRelativePose params
        self.declare_parameter("pos_tolerance", 0.05)            # meters
        self.declare_parameter("yaw_tolerance_deg", 3.0)         # degrees
        self.declare_parameter("kp_linear", 1.5)
        self.declare_parameter("kp_angular", 1.5)
        self.declare_parameter("default_timeout", 15.0)          # seconds
        self.declare_parameter("odom_stale_threshold", 0.5)      # seconds

        self._max_linear_vel = self.get_parameter("max_linear_vel").get_parameter_value().double_value
        self._max_angular_vel = self.get_parameter("max_angular_vel").get_parameter_value().double_value
        self._max_duration = self.get_parameter("max_duration").get_parameter_value().double_value

        self._pos_tol = self.get_parameter("pos_tolerance").get_parameter_value().double_value
        self._yaw_tol = math.radians(self.get_parameter("yaw_tolerance_deg").get_parameter_value().double_value)
        self._kp_lin = self.get_parameter("kp_linear").get_parameter_value().double_value
        self._kp_ang = self.get_parameter("kp_angular").get_parameter_value().double_value
        self._default_timeout = self.get_parameter("default_timeout").get_parameter_value().double_value
        self._odom_stale_threshold = self.get_parameter("odom_stale_threshold").get_parameter_value().double_value

        self._is_moving = False
        self._stop_requested = False
        self._lock = threading.Lock()

        self._odom_lock = threading.Lock()
        self._latest_pose = None  # tuple (x, y, yaw)
        self._latest_odom_stamp = None  # rclpy.time.Time

        self._cb_group = ReentrantCallbackGroup()

        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel_out", 10)
        self._rate = self.create_rate(self.PUBLISH_RATE)

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            Odometry, "/odom", self._odom_cb, odom_qos,
            callback_group=self._cb_group,
        )

        self.create_service(
            MoveForDuration,
            "/go2_vlm/move_for_duration",
            self._move_for_duration_cb,
            callback_group=self._cb_group,
        )
        self.create_service(
            MoveToRelativePose,
            "/go2_vlm/move_to_relative_pose",
            self._move_to_relative_pose_cb,
            callback_group=self._cb_group,
        )
        self.create_service(
            Trigger,
            "/go2_vlm/stop_movement",
            self._stop_callback,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f"MoveService ready. linear<={self._max_linear_vel:.2f} m/s, angular<={self._max_angular_vel:.2f} rad/s, "
            f"pos_tol={self._pos_tol:.3f} m, yaw_tol={math.degrees(self._yaw_tol):.1f} deg"
        )

    def _odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        with self._odom_lock:
            self._latest_pose = (msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
            self._latest_odom_stamp = self.get_clock().now()

    def _get_fresh_pose(self):
        with self._odom_lock:
            pose = self._latest_pose
            stamp = self._latest_odom_stamp

        if pose is None or stamp is None:
            return None
        age = (self.get_clock().now() - stamp).nanoseconds / 1e9
        if age > self._odom_stale_threshold:
            return None
        return pose

    def _stop_callback(self, request, response):
        with self._lock:
            self._stop_requested = True
        self._cmd_pub.publish(Twist())
        self.get_logger().warning("Emergency stop requested")
        response.success = True
        response.message = "Stop requested"
        return response

    def _acquire_move_slot(self):
        with self._lock:
            if self._is_moving:
                return False
            self._is_moving = True
            self._stop_requested = False
            return True

    def _release_move_slot(self):
        with self._lock:
            self._is_moving = False

    def _move_for_duration_cb(self, request, response):
        if not self._acquire_move_slot():
            response.success = False
            response.message = "Already moving. Call /go2_vlm/stop_movement first."
            response.actual_duration = 0.0
            return response

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
            self._release_move_slot()
            response.success = False
            response.message = "Duration must be positive"
            response.actual_duration = 0.0
            return response

        self.get_logger().info(
            f"MoveForDuration: vx={linear_x:.2f} vy={linear_y:.2f} vyaw={angular_z:.2f} dur={duration:.2f}s"
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
            self.get_logger().error(f"MoveForDuration failed: {e}")
            response.success = False
            response.message = f"Error: {e}"
            response.actual_duration = (self.get_clock().now() - start_time).nanoseconds / 1e9
            self._cmd_pub.publish(Twist())
            self._release_move_slot()
            return response

        self._cmd_pub.publish(Twist())
        self._release_move_slot()

        actual_duration = (self.get_clock().now() - start_time).nanoseconds / 1e9
        response.success = True
        response.actual_duration = actual_duration
        response.message = (
            f"Stopped early after {actual_duration:.2f}s" if stopped_early
            else f"Moved for {actual_duration:.2f}s"
        )
        self.get_logger().info(f"MoveForDuration done: {response.message}")
        return response

    def _move_to_relative_pose_cb(self, request, response):
        if not self._acquire_move_slot():
            response.success = False
            response.message = "Already moving. Call /go2_vlm/stop_movement first."
            response.final_dx = 0.0
            response.final_dy = 0.0
            response.final_dyaw = 0.0
            return response

        start_pose = self._get_fresh_pose()
        if start_pose is None:
            self._release_move_slot()
            response.success = False
            response.message = "No fresh /odom available"
            response.final_dx = 0.0
            response.final_dy = 0.0
            response.final_dyaw = 0.0
            return response

        start_x, start_y, start_yaw = start_pose
        cs, sn = math.cos(start_yaw), math.sin(start_yaw)
        target_x = start_x + cs * request.dx - sn * request.dy
        target_y = start_y + sn * request.dx + cs * request.dy
        target_yaw = wrap_angle(start_yaw + request.dyaw)

        timeout = request.timeout if request.timeout > 0 else self._default_timeout

        self.get_logger().info(
            f"MoveToRelativePose: dx={request.dx:.2f} dy={request.dy:.2f} "
            f"dyaw={math.degrees(request.dyaw):.1f}deg timeout={timeout:.1f}s"
        )

        start_time = self.get_clock().now()
        timeout_dur = Duration(seconds=timeout)
        reason = "reached"
        last_pose = start_pose

        try:
            while rclpy.ok():
                with self._lock:
                    if self._stop_requested:
                        reason = "stopped"
                        break

                if self.get_clock().now() - start_time >= timeout_dur:
                    reason = "timeout"
                    break

                cur_pose = self._get_fresh_pose()
                if cur_pose is None:
                    reason = "odom_lost"
                    break
                last_pose = cur_pose
                cur_x, cur_y, cur_yaw = cur_pose

                ex_w = target_x - cur_x
                ey_w = target_y - cur_y
                ccs, csn = math.cos(cur_yaw), math.sin(cur_yaw)
                ex_r = ccs * ex_w + csn * ey_w
                ey_r = -csn * ex_w + ccs * ey_w
                eyaw = wrap_angle(target_yaw - cur_yaw)

                if math.hypot(ex_r, ey_r) < self._pos_tol and abs(eyaw) < self._yaw_tol:
                    break

                twist = Twist()
                twist.linear.x = self._clamp(self._kp_lin * ex_r, -self._max_linear_vel, self._max_linear_vel)
                twist.linear.y = self._clamp(self._kp_lin * ey_r, -self._max_linear_vel, self._max_linear_vel)
                twist.angular.z = self._clamp(self._kp_ang * eyaw, -self._max_angular_vel, self._max_angular_vel)
                self._cmd_pub.publish(twist)
                self._rate.sleep()
        except Exception as e:
            self.get_logger().error(f"MoveToRelativePose failed: {e}")
            self._cmd_pub.publish(Twist())
            self._release_move_slot()
            response.success = False
            response.message = f"Error: {e}"
            response.final_dx = 0.0
            response.final_dy = 0.0
            response.final_dyaw = 0.0
            return response

        self._cmd_pub.publish(Twist())
        self._release_move_slot()

        cur_x, cur_y, cur_yaw = last_pose
        fx_w = cur_x - start_x
        fy_w = cur_y - start_y
        response.final_dx = cs * fx_w + sn * fy_w
        response.final_dy = -sn * fx_w + cs * fy_w
        response.final_dyaw = wrap_angle(cur_yaw - start_yaw)

        response.success = (reason == "reached")
        response.message = (
            f"{reason}: dx={response.final_dx:.3f} dy={response.final_dy:.3f} "
            f"dyaw={math.degrees(response.final_dyaw):.1f}deg"
        )
        if response.success:
            self.get_logger().info(f"MoveToRelativePose done: {response.message}")
        else:
            self.get_logger().warning(f"MoveToRelativePose ended: {response.message}")
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
