import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()

        self.subscription1 = self.create_subscription(
            Image,
            'robot1/camera/image_raw',
            self.image_callback1,
            10)

        self.subscription2 = self.create_subscription(
            Image,
            'robot1/camera/convolution_image',
            self.image_callback2,
            10)

        self.img1 = None
        self.img2 = None

    def image_callback1(self, msg):
        self.img1 = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        self.display_images()

    def image_callback2(self, msg):
        self.img2 = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        self.display_images()


    def display_images(self):
        if self.img1 is not None and self.img2 is not None:
            image1_resized = cv2.resize(self.img1, (600, 300))
            image2_resized = cv2.resize(self.img2, (300, 300))

            image2_resized = image2_resized[:, :, np.newaxis]
            image2_resized = cv2.cvtColor(image2_resized, cv2.COLOR_GRAY2BGR)
            image2_resized = image2_resized.astype(image1_resized.dtype)

            concatenated_image = cv2.hconcat([image1_resized, image2_resized])

            cv2.imshow('Image Window', concatenated_image)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
