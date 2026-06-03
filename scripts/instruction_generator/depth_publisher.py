# depth_republish.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import numpy as np

class DepthRepublisher(Node):
    def __init__(self):
        super().__init__('depth_republisher')
        self.pub = self.create_publisher(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw/raw_depth',
            10
        )
        self.sub = self.create_subscription(
            CompressedImage,
            '/camera/camera/aligned_depth_to_color/image_raw/compressedDepth',
            self.callback,
            10
        )

    def callback(self, msg: CompressedImage):
        # compressedDepth = 12-byte header + PNG data
        raw = np.frombuffer(msg.data[12:], dtype=np.uint8)
        import cv2
        decoded = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)  # 16UC1

        out = Image()
        out.header = msg.header
        out.height = decoded.shape[0]
        out.width = decoded.shape[1]
        out.encoding = '16UC1'
        out.step = out.width * 2
        out.data = decoded.tobytes()
        self.pub.publish(out)

def main():
    rclpy.init()
    rclpy.spin(DepthRepublisher())

if __name__ == '__main__':
    main()