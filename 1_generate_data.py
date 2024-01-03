import rclpy
import numpy as np
import random
import argparse
from ros2env import ROS2Env
import os
import time

DIR_NAME = './data/rollout/'

# Cria o diretório se ele não existir
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)

def main(args):
    env_name = args.env_name
    total_episodes = args.total_episodes
    time_steps = args.time_steps
    render = args.render

    # Inicialize o ROS2 aqui
    rclpy.init(args=None)

    env = ROS2Env(env_name)
    s = 0

    while s < total_episodes:
        episode_id = random.randint(0, 2**31 - 1)
        filename = DIR_NAME + str(episode_id) + ".npz"

        observation = env.reset()

        if render:
            # Render the environment if necessary
            pass

        t = 0
        obs_sequence = []
        action_sequence = []
        reward_sequence = []
        done_sequence = []

        reward = 0
        done = False

        while t < time_steps and not done:
            action = np.hstack([np.random.uniform(0.0, 1.2, 1), np.random.uniform(-0.8, 0.8, 1)])  # Sample random action
            observation, reward = env.step(action)

            obs_sequence.append(observation)
            action_sequence.append(action)
            reward_sequence.append(reward)
            done_sequence.append(done)

            t += 1

            if render:
                # Update rendering if necessary
                pass

            time.sleep(0.1)

        print(f"Episode {s} finished after {t} timesteps")
        np.savez_compressed(filename, obs=np.array(obs_sequence, dtype=object), action=np.array(action_sequence, dtype=object), reward=np.array(reward_sequence, dtype=object), done=np.array(done_sequence, dtype=object))

        s += 1
        
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Create new training data'))
    parser.add_argument('--env_name', type=str, help='name of environment')
    parser.add_argument('--total_episodes', type=int, default=242,
                        help='total number of episodes to generate')
    parser.add_argument('--time_steps', type=int, default=100,
                        help='how many timesteps at start of episode?')
    parser.add_argument('--render', default=0, type=int,
                        help='render the env as data is generated')

    args = parser.parse_args()
    main(args)
