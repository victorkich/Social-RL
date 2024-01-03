#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
import argparse
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from utils import ReplayMemory
from colorama import init as colorama_init
from colorama import Fore
from sensor_msgs.msg import CompressedImage, LaserScan, Image
from PIL import Image as pil_image
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
from cv_bridge import CvBridge
import os
import datetime
import time
import threading
from torchvision import transforms
import cv2


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=10000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(np.zeros(2), args)

#Tensorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), "multi_agents_env_test",
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0


class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(Image, '/robot1/camera/image_raw', self.image_callback, 1)
        self.image = None
        self.bridge = CvBridge()
        self.transform = transforms.Compose([
            transforms.Resize((32, 64), antialias=True),  # Altura, Largura
            transforms.ToTensor(),
        ])

    def image_callback(self, msg):
        # self.get_logger().info('Received image')
        image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        image = pil_image.fromarray(image)
        self.image = self.transform(image)


class ScanSubscriber(Node):
    def __init__(self):
        super().__init__('scan_subscriber')
        self.subscription = self.create_subscription(LaserScan, '/robot1/scan', self.scan_callback, 10)
        self.scan = None

    def scan_callback(self, msg):
        # self.get_logger().info('Received scan')
        msg = np.array(msg.ranges)
        scan = np.hstack([msg[:90], msg[len(msg)-90:]])
        self.scan = scan[::10]


class RewardSubscriber(Node):
    def __init__(self):
        super().__init__('reward_subscriber')
        self.subscription = self.create_subscription(Int32, '/global_reward', self.reward_callback, 10)
        self.reward = 0

    def reward_callback(self, msg):
        # self.get_logger().info('Received reward')
        self.reward = msg


class TwistPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_publisher')
        self.publisher = self.create_publisher(Twist, '/robot1/cmd_vel', 10)

    def cmd_vel_publish(self, msg):
        self.publisher.publish(msg)
        print('Publishing action:', msg.linear.x, msg.angular.z)


class ResetPublisher(Node):
    def __init__(self):
        super().__init__('cmd_vel_publisher')
        self.publisher = self.create_publisher(Int32, '/reset', 1)

    def reset_publish(self, msg):
        self.publisher.publish(msg)


class ConvolutionPublisher(Node):
    def __init__(self):
        super().__init__('convolution_publisher')
        self.publisher = self.create_publisher(Image, '/robot1/camera/convolution_image', 1)
        self.bridge = CvBridge()

    def convolutional_publish(self, msg):
        msg = msg.detach().numpy()
        selected_image = msg[0, :, :, 0]  # Selecione a primeira imagem e o primeiro canal
        resized_image = cv2.resize(selected_image, (300, 300))

        #grid_images = msg.reshape(8, 4, 8, 16)
        #grid_images = grid_images.swapaxes(1, 2).reshape(8 * 8, 4 * 16)
        # msg = self.bridge.cv2_to_compressed_imgmsg(msg.detach().numpy(), "bgr8")
        msg = self.bridge.cv2_to_imgmsg(resized_image, "32FC1")
        self.publisher.publish(msg)


os.system('clear')
colorama_init(autoreset=True)
print(Fore.RED + '------ MULTI-AGENT DEEP REINFORCEMENT LEARNING USING PYTORCH ------'.center(100))
rclpy.init()
reward_subscriber = RewardSubscriber()
scan_subscriber = ScanSubscriber()
image_subscriber = ImageSubscriber()
twist_publisher = TwistPublisher()
reset_publisher = ResetPublisher()
convolution_publisher = ConvolutionPublisher()
executor = SingleThreadedExecutor()
executor.add_node(reward_subscriber)
executor.add_node(image_subscriber)
executor.add_node(scan_subscriber)
spin_thread = threading.Thread(target=executor.spin, args=())
spin_thread.start()
time.sleep(1)

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    reset = Int32()
    reset.data = 1
    reset_publisher.reset_publish(reset)
    time.sleep(3)
    state = [image_subscriber.image, scan_subscriber.scan]

    while not done:
        if args.start_steps > total_numsteps:
            action = np.hstack([np.random.uniform(-0.8, 0.8, 1), np.random.uniform(0.0, 1.2, 1)])  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy
        if agent.policy.convolutional_image is not None:
            convolution_publisher.convolutional_publish(agent.policy.convolutional_image)

        action[1] = 0.0 if action[1] < 0.0 else action[1]

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        vel_cmd = Twist()
        vel_cmd.linear.x = float(action[1])
        vel_cmd.angular.z = float(action[0])
        twist_publisher.cmd_vel_publish(vel_cmd)

        next_state = [image_subscriber.image, scan_subscriber.scan]
        reward = reward_subscriber.reward.data * 10
        episode_steps += 1
        total_numsteps += 1

        if any(scan_subscriber.scan < 0.2):
            done = True
            reward = -10.0
        elif episode_steps >= 200:
            done = True
        else:
            done = False

        if reward > 0.0:
            done = True

        episode_reward += reward
        print('Scan:', scan_subscriber.scan)
        print('Done:', done)
        print(Fore.RED + 'Reward: ' + str(reward))

        mask = 1 if episode_steps == 100 else float(not done)
        memory.push(state[0], state[1], action, reward, next_state[0], next_state[1], mask)  # Append transition to memory
        state = next_state
        time.sleep(0.05)

    print(total_numsteps)

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 2
        for _ in range(episodes):
            reset = Int32()
            reset.data = 1
            reset_publisher.reset_publish(reset)
            time.sleep(3)
            state = [image_subscriber.image, scan_subscriber.scan]
            episode_reward = 0
            episode_steps = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)
                if agent.policy.convolutional_image is not None:
                    convolution_publisher.convolutional_publish(agent.policy.convolutional_image)

                action[1] = 0.0 if action[1] < 0.0 else action[1]
                vel_cmd = Twist()
                vel_cmd.linear.x = float(action[1])
                vel_cmd.angular.z = float(action[0])
                twist_publisher.cmd_vel_publish(vel_cmd)
                next_state = [image_subscriber.image, scan_subscriber.scan]
                reward = reward_subscriber.reward.data * 10
                if any(scan_subscriber.scan < 0.2):
                    done = True
                    reward = -10.0
                elif episode_steps >= 200:
                    done = True
                else:
                    done = False

                if reward > 0.0:
                    done = True
                episode_reward += reward
                episode_steps += 1
                state = next_state

                time.sleep(0.05)
            avg_reward += episode_reward
        avg_reward /= episodes

        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

image_subscriber.destroy_node()
reward_subscriber.destroy_node()
scan_subscriber.destroy_node()
rclpy.shutdown()
