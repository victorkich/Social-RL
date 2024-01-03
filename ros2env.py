import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from cv_bridge import CvBridgeError
from PIL import Image as PILImage
import io

class ROS2Env(Node):
    def __init__(self, env_name):
        super().__init__(env_name)

        # Subscrições
        self.image_subscription = self.create_subscription(Image, '/robot1/camera/image_raw', self.image_callback, 10)
        self.reward_subscription = self.create_subscription(Int32, '/global_reward', self.reward_callback, 10)
        
        # Publicações
        self.cmd_vel_publisher = self.create_publisher(Twist, '/robot1/cmd_vel', 10)

        self.reset_publisher = self.create_publisher(Int32, '/reset', 1)
        
        # Variáveis de Estado
        self.current_image = None
        self.current_reward = 0
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            # Decodifique a imagem JPEG manualmente
            pil_image = PILImage.open(io.BytesIO(msg.data))
            cv_image = np.array(pil_image)  # Converter para um array numpy
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)  # Converter de RGB para BGR, se necessário
            self.current_image = cv_image
        except Exception as e:
            raise CvBridgeError(e)

    def reward_callback(self, msg):
        self.current_reward = msg.data

    def step(self, action):
        # Envie o comando de ação para o robô
        cmd_msg = Twist()
        cmd_msg.linear.x = action[0]
        cmd_msg.angular.z = action[1]
        self.cmd_vel_publisher.publish(cmd_msg)

        # Aguarde um curto período para a ação ser executada
        rclpy.spin_once(self, timeout_sec=0.1)

        # Retorne o estado atual e a recompensa
        return self.current_image, self.current_reward

    def reset(self):
        reset = Int32()
        reset.data = 1
        self.reset_publisher.publish(reset)
        return self.current_image

    def close(self):
        # Limpeza, se necessário
        self.destroy_node()
        rclpy.shutdown()