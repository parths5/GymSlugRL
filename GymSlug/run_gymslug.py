#!/usr/bin/env python
"""
GymSlug Runner Script
Based on usage.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import matplotlib.animation as animation
import gym
import random
import os
from datetime import date

# Import expert policy
from aplysia_feeding_ub import AplysiaFeedingUB

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.env_checker import check_env
from stable_baselines import DQN
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# Utils for monitoring training
def compute_avg_return(environment, model, num_episodes=10, verbose=0, drop_out=0):
    total_return = 0.0
    record = np.zeros(num_episodes)
    for i in range(num_episodes):
        obs = environment.reset()
        episode_return = 0.0
        while True:
            action, _states = model.predict(obs)
            obs, reward, done, info = environment.step(action)
            if verbose:
                print('ep:{}.rw:{}.act:{}.o_stat:{}.n_stat;{}'.format(i, reward, action, obs, obs))
            episode_return += reward
            if done:
                break

        record[i] = episode_return
        total_return += episode_return
        print('eval-summary --> eps = {}, episode reward = {}'.format(i, episode_return))
    if drop_out:
        keep = int(num_episodes/2)
        record = np.sort(record)[keep-1:]
        avg_return = np.average(record)
        return avg_return
    else: 
        avg_return = total_return / num_episodes
        print('std:{}, max:{}, min:{}'.format(np.std(record), np.max(record), np.min(record)))
        return avg_return

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, plot_env, eval_env, num_eval_episodes, log_interval: int, eval_interval: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_reward = 0
        self.plot_env = plot_env
        self.eval_env = eval_env
        self.num_eval_episodes = num_eval_episodes
        self.eval_r_log = []
        self.eval_i_log = []

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            print('Reach step {}'.format(self.n_calls))
        if self.n_calls % self.eval_interval == 0:
            # Mean training reward over the last 100 episodes
            avg_reward = compute_avg_return(self.eval_env, self.model, self.num_eval_episodes, 0, 0)
            self.eval_r_log.append(avg_reward)
            self.eval_i_log.append(self.n_calls)
            # New best model, you could save the agent here
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                print("best policy updated, avg_reward={}".format(avg_reward))
                self.model.save(self.save_path + str(avg_reward))
                compute_avg_return(self.plot_env, self.model, 2, 0, 0)
            elif avg_reward > 125:
                print('avg_return={}>125'.format(avg_reward))
                self.model.save(self.save_path + str(avg_reward))
                compute_avg_return(self.plot_env, self.model, 2, 0, 0)

        return True
    
    def _on_training_end(self):
        print(self.eval_r_log)
        plt.plot(self.eval_i_log, self.eval_r_log)
        plt.hlines(expert_avg_reward, 0, max(self.eval_i_log), 'r')
        plt.ylabel('Average Return') 
        plt.xlabel('Iterations')
        plt.savefig(os.path.join(self.log_dir, 'training_results.png'))
        print('Training results saved to training_results.png')

if __name__ == "__main__":
    print("Starting GymSlug...")
    
    # Aplysia Feeding driver biting
    suffix = str(date.today())
    xlimits = [0,60]
    
    # Initialize simulation object
    aplysia = AplysiaFeedingUB()
    
    # Swallowing
    print('Swallowing')
    aplysia.SetSensoryStates('swallow')
    aplysia.RunSimulation()
    aplysia.GeneratePlots_WS('Swallow_'+suffix, xlimits)
    
    # Verify Environment with Stable Baselines
    # slug-v0: UnbreakableSeaweed, used in this notebook
    # slug-v1: BreakableSeaweed
    env = gym.make("gym_slug:slug-v0")
    
    # Check environment
    check_env(env)
    print('env validated. Default args: foo={}, max_steps={}, threshold={}, patience={}'.format(
        env.foo, env.max_steps, env.threshold, env.patience))
    
    # Check expert performance in gym_slug
    std_env = env
    std_env.preset_inputs = 1
    std_env.set_plotting(1)
    std_env.set_verbose(1)  # 1: print reset condition

    obs = std_env.reset()
    rewards = []
    steps = []
    reward_log = []
    num_episodes = 10

    for i in range(num_episodes):
        episode_reward = 0
        episode_steps = 0
        obs = std_env.reset()
        writer = []
        while True:       
            actionArray = np.ones((1,5))
            actionArray[0,2] = aplysia.B8[0,episode_steps]
            actionArray[0,4] = aplysia.B38[0,episode_steps]
            actionArray[0,1] = aplysia.B6B9B3[0,episode_steps]
            actionArray[0,3] = aplysia.B31B32[0,episode_steps]
            actionArray[0,0] = aplysia.B7[0,episode_steps]   
            action = actionArray[0,:]
            state, reward, done, info = std_env.step(action)
            
            episode_steps += 1
            episode_reward += reward
            writer.append(reward)
            if done:
                break

        print('summary -> @episode {}, reward = {}'.format(i, episode_reward))  
        rewards.append(episode_reward)
        steps.append(episode_steps)
        reward_log.append(writer)

    num_steps = np.sum(steps)
    avg_length = np.mean(steps)
    avg_reward = np.mean(rewards)
    total_reward = np.sum(rewards)
    print('num_episodes:', num_episodes, 'num_steps:', num_steps)
    print('avg_length', avg_length, 'avg_reward:', avg_reward)
    print('total reward: ', total_reward)
    print('std', np.std(rewards))
    print('max:{}, min:{}'.format(np.max(rewards), np.min(rewards)))
    expert_avg_reward = np.mean(rewards)
    print('expert_avg_reward = ', expert_avg_reward)
    
    # Create environments for training, evaluation, and plotting
    # initiate training env
    train_env = gym.make("gym_slug:slug-v0")
    train_env.present_inputs = 0
    train_env.set_verbose(0)

    # initiate evaluation env
    eval_env = gym.make("gym_slug:slug-v0")
    eval_env.set_verbose(0)
    eval_env.present_inputs = 0

    # initiate an env with plotting
    plot_env = gym.make("gym_slug:slug-v0")
    plot_env.set_plotting(2)
    plot_env.present_inputs = 0
    
    # Training parameters
    num_iterations = 500000  # or 1M
    log_interval = 100
    num_eval_episodes = 5
    eval_interval = 500
    
    # Setup logging
    best_reward = 0
    log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)

    train_env = Monitor(train_env, log_dir)
    
    # Create the callback
    callback = SaveOnBestTrainingRewardCallback(
        plot_env, eval_env, num_eval_episodes, 
        log_interval=log_interval, eval_interval=eval_interval, log_dir=log_dir
    )
    
    # Create and train the model
    model = DQN('MlpPolicy', train_env, learning_rate=2.5e-4, prioritized_replay=True, verbose=1)
    
    print("Starting training...")
    print("This should take approximately 30mins-1hour...")
    
    # Train the agent
    model.learn(total_timesteps=num_iterations, callback=callback)
    
    # Generate final plots
    callback._on_training_end()
    
    print("Training completed!")
