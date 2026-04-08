import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import threading


class UserInputNode(Node):
    def __init__(self):
        super().__init__('user_input_node')

        self.publisher = self.create_publisher(
            String,
            '/go2_vlm/user_input',
            10
        )

        self.get_logger().info("User input node started")

        self.input_thread = threading.Thread(target=self.input_loop, daemon=True)
        self.input_thread.start()

    def input_loop(self):
        while rclpy.ok():
            try:
                user_text = input(">>> ")

                if not user_text.strip():
                    continue

                msg = String()
                msg.data = user_text

                self.publisher.publish(msg)

                self.get_logger().info(f"Published: {user_text}")

            except Exception as e:
                self.get_logger().error(f"Input error: {e}")


def main():
    rclpy.init()
    node = UserInputNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()