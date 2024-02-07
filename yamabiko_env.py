import gym
from gym import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32, Int32MultiArray, Bool, Float32
import cv2
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image as PILImage
import io
import threading


class YamabikoEnv(gym.Env, Node):  # Herda de Node
    def __init__(self):
        # Inicialize o node ROS2 como parte da inicialização da classe
        super().__init__('yamabiko_gym_env')  # Nome do node

        # Define ações e espaços de observação
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1], dtype=np.float32),
        )
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

        # Inicializa a comunicação ROS2
        self.image_subscriber = self.create_subscription(Image, '/Robot/camera/image_raw', self.image_callback, 1)
        self.scan_subscriber = self.create_subscription(LaserScan, '/Robot/scan', self.scan_callback, 10)
        self.collision_subscriber = self.create_subscription(Int32, '/Robot/collision', self.collision_callback, 10)
        self.reward_subscriber = self.create_subscription(Int32, '/global_reward', self.reward_callback, 10)
        self.ball_distance_subscriber = self.create_subscription(Float32, '/Robot/ball_distance', self.ball_distance_callback, 10)
        self.ball_possession_subscriber = self.create_subscription(Bool, '/Robot/ball_possession', self.ball_possession_callback, 10)
        self.green_area_distance_subscriber = self.create_subscription(Float32, '/Robot/green_area_distance', self.green_area_distance_callback, 10)

        self.cmd_vel_publisher = self.create_publisher(Twist, '/Robot/cmd_vel', 10)
        self.claw_publisher = self.create_publisher(Bool, '/Robot/claw', 10)
        self.room_size_publisher = self.create_publisher(Int32MultiArray, '/reset', 10)

        # Variáveis para armazenar os estados
        self.current_ball_distance = None
        self.current_ball_possession = False
        self.current_green_area_distance = None
        self.current_image = None
        self.current_scan = None
        self.current_collision = None
        self.current_reward = None
        self.bridge = CvBridge()

        # Inicializa as melhores distâncias com um valor alto
        self.best_ball_distance = float('inf')
        self.best_green_area_distance = float('inf')
        self.has_ball = False

        # Inicie a thread de spin em um método separado para manter o __init__ limpo
        self.start_ros_spin_thread()

        self.previous_image = None

    def start_ros_spin_thread(self):
        # Roda rclpy.spin em uma thread separada para não bloquear o loop principal
        thread = threading.Thread(target=rclpy.spin, args=(self,), daemon=True)
        thread.start()

    def image_callback(self, msg):
        try:
            pil_image = PILImage.open(io.BytesIO(msg.data))
            cv_image = np.array(pil_image)  # Converter para um array numpy
            self.current_image = cv2.resize(cv_image, (64, 64))
        except CvBridgeError as e:
            print(e)

    def ball_distance_callback(self, msg):
        self.current_ball_distance = msg.data

    def ball_possession_callback(self, msg):
        self.current_ball_possession = msg.data

    def green_area_distance_callback(self, msg):
        self.current_green_area_distance = msg.data

    def scan_callback(self, msg):
        self.current_scan = msg.ranges

    def collision_callback(self, msg):
        if not msg.data:
            return
        self.current_collision = msg.data

    def reward_callback(self, msg):
        self.current_reward = msg.data

    def set_claw_state(self, open_claw):
        # Cria uma mensagem booleana para o estado da garra
        claw_msg = Bool()
        claw_msg.data = open_claw
        # Publica a mensagem no tópico /Robot/claw
        self.claw_publisher.publish(claw_msg)

    def step(self, action):
        # Send velocity command
        twist = Twist()
        twist.linear.x = float(action[0])
        twist.angular.z = float(action[1])
        self.cmd_vel_publisher.publish(twist)

        # Set claw state based on the action
        self.set_claw_state(open_claw=float(action[2]) > 0.5)

        rclpy.spin_once(self, timeout_sec=0.1)  # Alterado de self.node para self

        while np.array_equal(self.current_image, self.previous_image):
            rclpy.spin_once(self, timeout_sec=0.1)  # Alterado de self.node para self

        # Assemble the observation using only the image
        observation = self.current_image
        self.previous_image = observation

        reward = 0  # Inicializa a recompensa como 0
        done = False

        # Lógica de recompensa por progresso em direção à bola
        if self.current_ball_distance is not None:
            if self.current_ball_distance < self.best_ball_distance:
                # Calcula o progresso de distancia para levar como base no calculo do reward
                distance_diff = self.best_ball_distance - self.current_ball_distance

                # Atualiza a melhor distância e recompensa o agente
                self.best_ball_distance = self.current_ball_distance
                
                if distance_diff > 5:
                    reward += 0.5
                else:
                    reward += distance_diff
            else:
                # Se não houve melhora, não há recompensa
                reward += 0

        # Lógica de recompensa por pegar a bola e progresso em direção ao pressure plate
        if self.current_ball_possession and not self.has_ball:
            # Primeira vez que pega a bola
            self.has_ball = True
            reward += 1  # Recompensa por pegar a bola

            # Reseta a melhor distância para o pressure plate para incentivar o progresso
            self.best_green_area_distance = float('inf')

        # Se possui a bola e está se movendo em direção ao pressure plate
        if self.has_ball and self.current_green_area_distance is not None:
            if self.current_green_area_distance < 0.4:
                reward += 2.5
                done = True
            elif self.current_green_area_distance < self.best_green_area_distance:
                # Calcula o progresso de distancia para levar como base no calculo do reward
                distance_diff = self.best_green_area_distance - self.current_green_area_distance

                # Atualiza a melhor distância ao pressure plate e recompensa
                self.best_green_area_distance = self.current_green_area_distance
                if distance_diff > 5:
                    reward += 0.5
                else:
                    reward += distance_diff
            else:
                # Se não houve melhora, não há recompensa adicional
                reward += 0

        # Check for collision
        if self.current_collision is not None and self.current_collision > 0:
            done = True
            reward = -1  # Assign a negative reward for collision

        rclpy.spin_once(self, timeout_sec=0.1)

        # Return observation, reward, done, and info
        return observation, reward, done, {}

    def reset_environment(self):
        # Generate new random sizes for x and y that are even numbers between 8 and 18 (inclusive)
        new_size_x = 2 * np.random.randint(5, 10)  # Will generate a random even number between 10 (inclusive) and 20 (exclusive)
        new_size_y = 2 * np.random.randint(5, 10)  # Same here for y

        # Call the environment's reset function with the new size parameters
        return self.reset(new_size_x, new_size_y)

    def reset(self, new_size_x, new_size_y):
        # Create a message with the new size
        room_size_msg = Int32MultiArray()
        room_size_msg.data = [new_size_x, new_size_y]

        # Publish the message on the /reset topic
        self.room_size_publisher.publish(room_size_msg)

        # Reset internal state
        self.current_collision = 0
        self.current_reward = 0
        self.set_claw_state(open_claw=False)  # Start with the claw closed
        self.current_image = None
        self.previous_image = None

        # Inicializa as melhores distâncias com um valor alto
        self.best_ball_distance = float('inf')
        self.best_green_area_distance = float('inf')
        self.has_ball = False  # Se o agente pegou a bola

        # Wait for the environment to reset
        rclpy.spin_once(self, timeout_sec=0.1)  # Alterado de self.node para self

        # Get the initial observation by waiting for the next image message
        while self.current_image is None:
            rclpy.spin_once(self, timeout_sec=0.1)  # Alterado de self.node para self

        self.previous_image = self.current_image

        # Return the initial observation
        initial_observation = {
            "image": self.current_image,
            "scan": self.current_scan,
            "collision": self.current_collision
        }
        return initial_observation

    def close(self):
        # Sobreescrita para garantir uma parada limpa da thread
        super().destroy_node()  # Destruir o nó antes de chamar rclpy.shutdown
        rclpy.shutdown()
