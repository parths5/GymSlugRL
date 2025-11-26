#!/usr/bin/env python
"""
SAC with Attention Policy Training Script
Based on usage_py.py but using custom AttentionPolicy instead of MlpPolicy
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
from aplysia_feeding_b import AplysiaFeedingB

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.env_checker import check_env
from stable_baselines import SAC
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import BaseCallback

# Import custom Attention Policy
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from policies.attention_policy import AttentionPolicy

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

##############################################################################################################################################
# ============================= Start by generating EXPERT model, this example is specifically for swallowing ============================== #
##############################################################################################################################################
# Set type of swallow here
unbreakable = True

if unbreakable:
    # Aplysia Feeding Driver Biting --> Unbreakable = Loaded
    suffix = str(date.today())
    xlimits = [0,60]
    aplysia = AplysiaFeedingUB() # Initialize simulation object
    print('Swallowing')
    aplysia.SetSensoryStates('swallow')
    aplysia.RunSimulation()
else:
    # Aplysia Feeding Driver Biting --> Breakable = Unloaded
    suffix = str(date.today())
    xlimits = [0,60]
    aplysia = AplysiaFeedingB() # Initialize simulation object
    print('Swallowing')
    aplysia.SetSensoryStates('swallow')
    aplysia.RunSimulation()

##==========================================
## Verify Environment with Stable Baselines
##==========================================
# slug-v0: UnbreakableSeaweed, used here
# slug-v1: BreakableSeaweed
env = gym.make("gym_slug:slug-v0") 

check_env(env)
print('env validated. Default args: foo={}, max_steps={}, threshold={}, patience={}'.format(
    env.foo, env.max_steps, env.threshold, env.patience))

##======================================
## Check expert performance in gym_slug
##======================================
std_env = env
std_env.preset_inputs = 1   # used in MuscleActivations_001 function and sets neuron properties
std_env.set_plotting(0)     # 1 is yes you want plots and 0 is no you don't want any plots
std_env.set_verbose(1)      # 1 prints reset condition and any background information about what is going on during training

obs = std_env.reset()       # Initial state by calling reset to reset environment parameters
rewards = []                # Array of reward history during training 
steps = []
reward_log = []
num_episodes = 10

for i in range(num_episodes):
    if i == 9:
        std_env.set_plotting(1)
    episode_reward = 0
    episode_steps = 0
    obs = std_env.reset() # Reset environment parameters
    writer = []
    while True:
        actionArray = np.ones((1,5)) # array of ones
        # Fill action array with expert neuron state -> 0 for off and 1 for on
        actionArray[0,2] = aplysia.B8[0,episode_steps]
        actionArray[0,4] = aplysia.B38[0,episode_steps]
        actionArray[0,1] = aplysia.B6B9B3[0,episode_steps]
        actionArray[0,3] = aplysia.B31B32[0,episode_steps]
        actionArray[0,0] = aplysia.B7[0,episode_steps]   
        action = actionArray[0,:]
        state, reward, done, info = std_env.step(action)
        
        # update steps and reward
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
print('total reward: ',total_reward)
print('std', np.std(rewards))
print('max:{}, min:{}'.format(np.max(rewards), np.min(rewards)))

plt.figure()
plt.plot(rewards)
plt.show()
plt.savefig("fig1.png")

plt.figure()
for i in reward_log:
    plt.plot(i)
plt.show()
plt.savefig("fig2.png")

expert_avg_reward = np.mean(rewards)
print('expert_avg_reward = ', expert_avg_reward)

##==================================================
## Utility function for training process monitoring
##==================================================
def compute_avg_return(environment, model, num_episodes=10, verbose = 0, drop_out = 0):
    """
    Compute average return over multiple episodes
    """
    total_return = 0.0
    record = np.zeros(num_episodes)
    for i in range(num_episodes):
        obs = environment.reset()  # for every episode reset environment
        episode_return = 0.0       # initialize episode reward to 0 
        while True:
            action, _states = model.predict(obs)                # returns observed action and state from SAC model
            obs, reward, done, info = environment.step(action)  # take step to get reward from action
            if verbose:
                print('ep:{}.rw:{}.act:{}.o_stat:{}.n_stat;{}'.format(i, reward, action, obs, obs))
            # keep looping and adding up reward until episode is done
            episode_return += reward
            if done:
                break

        record[i] = episode_return      # store episode reward
        total_return += episode_return  # add episode reward to total to average out later
        print('eval-summary --> eps = {}, episode reward = {}'.format(i, episode_return))
    if drop_out:
        keep = int(num_episodes/2)          # usually 5
        record = np.sort(record)[keep-1:]   # sorts rewards from least to most and only keeps the first 5 values
        avg_return = np.average(record)     # record new average
        return avg_return
    else: 
        avg_return = total_return / num_episodes # no dropout so will return total reward average
        print('std:{}, max:{}, min:{}'.format(np.std(record), np.max(record), np.min(record)))
        return avg_return

##============================================================
## Create environments for training, evaluation, and plotting
##============================================================
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

## Train SAC with Attention Policy
# training parameters
num_iterations = 500000 # @param {type:"integer"} or 1M #360000
log_interval = 100      # @param {type:"integer"}
num_eval_episodes = 5   # @param {type:"integer"}
eval_interval = 500     # @param {type:"integer"}

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
            # Mean training reward over the last episodes
            avg_reward = compute_avg_return(self.eval_env, self.model, self.num_eval_episodes, 0, 0)
            self.eval_r_log.append(avg_reward)
            self.eval_i_log.append(self.n_calls)
            # New best model, you could save the agent here
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                print("best policy updated, avg_reward={}".format(avg_reward))
                avr_reward_str = self._printable_reward_title(avg_reward)
                self.model.save(self.save_path + avr_reward_str)        # save new model with best average output
                compute_avg_return(self.plot_env, self.model, 2, 0, 0)  # generate plots for best model
        return True
    

    def _on_training_end(self):
        # for plotting
        print(self.eval_r_log)
        plt.plot(self.eval_i_log, self.eval_r_log)
        plt.hlines(expert_avg_reward, 0, max(self.eval_i_log), 'r', label='Expert')
        plt.ylabel('Average Return') 
        plt.xlabel('Iterations')
        plt.title('SAC with Attention - Training Progress')
        plt.legend()
        plt.show()
        plt.savefig("fig3_attention.png")
        print('Training results saved to fig3_attention.png')

    def _printable_reward_title(self, avg_reward) -> str:
        # for printing reward
        lhs = int(avg_reward)
        rhs = int((avg_reward - lhs) * 10000000000)
        return str(lhs)+"_"+str(rhs)

best_reward = 0
log_dir = './logs_sac_attention/'
os.makedirs(log_dir, exist_ok=True)

train_env = Monitor(train_env, log_dir)

# Create the callback class obj
callback = SaveOnBestTrainingRewardCallback(
    plot_env, eval_env, num_eval_episodes, 
    log_interval=log_interval, eval_interval=eval_interval, log_dir=log_dir
)

# Create SAC model with Attention Policy
# Note: We pass the AttentionPolicy class and policy_kwargs for attention parameters
print("Creating SAC model with Attention Policy...")
print("Attention parameters: heads=4, attention_dim=64")

model = SAC(
    AttentionPolicy, 
    train_env, 
    learning_rate=2.5e-4, 
    verbose=1,
    policy_kwargs={
        'attention_heads': 4,
        'attention_dim': 64
    }
)

print("Starting training with SAC + Attention...")
print("This should take approximately 30mins-1hour...")

# Train the agent SAC model with callback class as one of the parameters
model.learn(total_timesteps=num_iterations, callback=callback)

# plot rewards over time
callback._on_training_end()

print("Training completed!")

