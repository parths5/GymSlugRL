'''




import numpy as np
# matplotlib qt
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import matplotlib.animation as animation
import gym
import random
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


## Import expert policy
# The goal is to learn a policy that 
# 1. generates similar motor neuron control/action with the expert
# 2. shows similar or higher average episode reward with the expert (~132.8)


from aplysia_feeding_ub import AplysiaFeedingUB      #import unbreakable class
from datetime import date

print("\n\n                 STAMP 1\n\n")
#Aplysia Feeding driver biting
suffix = str(date.today())      #set suffix as today's date
xlimits = [0,60]                #creates a list with two values 0 and 60 --> maybe cause expert was run for 60s?
## Initialize simulation object
aplysia = AplysiaFeedingUB()    #create ubreakable seaweed feeding model object
print("\n\n                 STAMP 2\n\n")
## Swallowing
print('Swallowing')
aplysia.SetSensoryStates('swallow')
''''''
# Passing this function with single parameter 'swallow' will do this below

elif (behavior=='swallow'):
    # below are sensory state vectors
    self.sens_chemical_lips = np.ones((1,nt))
    self.sens_mechanical_lips = np.ones((1,nt))
    self.sens_mechanical_grasper = np.ones((1,nt))
    self.fixation_type = np.ones((1,nt))
''''''
print("\n\n                 STAMP 3\n\n")
aplysia.RunSimulation()
'''
# I will explain path of things that happen when RunSimulation is called
    #
'''
aplysia.GeneratePlots_WS('Swallow_'+suffix,xlimits)
print("\n\n                 STAMP 4\n\n")

## uncomment to visualize expert motion (saved to test.mp4)
#aplysia.VisualizeMotion()

## Verify Environment with Stable Baselines

# slug-v0: UnbreakableSeaweed, used in this notebook
# slug-v1: BreakableSeaweed
env = gym.make("gym_slug:slug-v1")
print("\n\n                 STAMP 5\n\n")
    # from stable_baselines3.sac.policies import MlpPolicy
    # from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

from stable_baselines.common.env_checker import check_env
check_env(env)

print('env validated. Default args: foo={}, max_steps={}, threshold={}, patience={}'.format(env.foo, env.max_steps,env.threshold, env.patience))
print("\n\n                 STAMP 6\n\n")

## Check expert performance in gym_slug
std_env = env
std_env.preset_inputs = 1
std_env.set_plotting(1)
std_env.set_verbose(1) # 1: print reset condition
print("\n\n                 STAMP 7\n\n")
obs = std_env.reset()
rewards = []
steps = []
reward_log = []
num_episodes = 10
print("\n\n                 STAMP 8\n\n")
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
    print("\n\n                 STAMP 9\n\n")
    print('summary -> @episode {}, reward = {}'.format(i, episode_reward))  
    rewards.append(episode_reward)
    steps.append(episode_steps)
    reward_log.append(writer)
print("\n\n                 STAMP 10\n\n")

num_steps = np.sum(steps)
avg_length = np.mean(steps)
avg_reward = np.mean(rewards)
total_reward = np.sum(rewards)
print('num_episodes:', num_episodes, 'num_steps:', num_steps)
print('avg_length', avg_length, 'avg_reward:', avg_reward)
print('total reward: ',total_reward)
print('std', np.std(rewards))
print('max:{}, min:{}'.format(np.max(rewards), np.min(rewards)))
expert_avg_reward = np.mean(rewards)
print('expert_avg_reward = ', expert_avg_reward)
print("\n\n                 STAMP 11\n\n")
 #
## Utility function for training process monitoring


def compute_avg_return(environment, model, num_episodes=10, verbose = 0, drop_out = 0):
    # === Computer Average Return Reward ============================================================================================================================================================================================================
    total_return = 0.0
    record = np.zeros(num_episodes)
    for i in range(num_episodes):
        obs = environment.reset()
        episode_return = 0.0
        while True:
            action, _states = model.predict(obs)
            obs, reward, done, info = environment.step(action) # todo Wen: action shape!
            if verbose:
                print('ep:{}.rw:{}.act:{}.o_stat:{}.n_stat;{}'.format(i,time_step.reward.numpy(),action_step.action[0],old_step.observation.numpy()[0,:],time_step.observation.numpy()[0,:]))
            episode_return += reward # = or +=
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

 #
## Create environments for training, evaluation, and plotting
print("\n\n                 STAMP 12\n\n")

# initiate training env
train_env = gym.make("gym_slug:slug-v1")
train_env.present_inputs = 0
train_env.set_verbose(0)
print("\n\n                 STAMP 13\n\n")
# initiate evalation env
eval_env = gym.make("gym_slug:slug-v1")
eval_env.set_verbose(0)
eval_env.present_inputs = 0
print("\n\n                 STAMP 14\n\n")
# initiate an env with plotting
plot_env = gym.make("gym_slug:slug-v1")
plot_env.set_plotting(2)
plot_env.present_inputs = 0
print("\n\n                 STAMP 15\n\n")

## Train vanilla DQN. Periodically evalulate using callback

# training parameters
num_iterations = 50000 # @param {type:"integer"} or 1M
log_interval = 100  # @param {type:"integer"}
num_eval_episodes = 5  # @param {type:"integer"}
eval_interval = 500  # @param {type:"integer"}
print("\n\n                 STAMP 16\n\n")
from stable_baselines import DQN
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import matplotlib.animation as animation

from stable_baselines import DDPG
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.callbacks import BaseCallback


print("\n\n                 STAMP 17\n\n")
class SaveOnBestTrainingRewardCallback(BaseCallback):
    
    def __init__(self, plot_env, eval_env, num_eval_episodes, log_interval: int, eval_interval: int, log_dir: str, verbose=1):
        # === Preallocate arrays ============================================================================================================================================================================================================
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
        # === Create folder if needed ============================================================================================================================================================================================================
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)


    def _on_step(self) -> bool:
        # === On Step ============================================================================================================================================================================================================
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
                self.model.save(self.save_path + str(avg_reward(int)))   ####### MAKE ALL REWARDS INTO WHOLE NUMBERS WITH NO DECIMAL POINTS TO NOT MESS UP TYPE
                compute_avg_return(self.plot_env, self.model, 2, 0, 0)
        return True
    

    def _on_training_end(self):
        # === Create folder if needed ============================================================================================================================================================================================================
        print(self.eval_r_log)
        plt.plot(self.eval_i_log, self.eval_r_log)
        plt.hlines(expert_avg_reward, 0, max(self.eval_i_log), 'r')
        plt.ylabel('Average Return') 
        plt.xlabel('Iterations')
        plt.show()
print("\n\n                 STAMP 18\n\n")

best_reward = 0
log_dir = '/home/camila/GymSlug-main/GymSlug'
os.makedirs(log_dir, exist_ok=True)

train_env = Monitor(train_env, log_dir)
print("\n\n                 STAMP 19\n\n")
# Create the callback
callback = SaveOnBestTrainingRewardCallback(plot_env, eval_env, num_eval_episodes, log_interval=log_interval, eval_interval=eval_interval, log_dir=log_dir)
model = DQN('MlpPolicy', train_env, prioritized_replay=True, learning_rate=2.5e-4,  verbose=1)
print("\n\n                 STAMP 20\n\n")
# Train the agent
model.learn(total_timesteps=num_iterations, callback=callback)
callback._on_training_end()
print("\n\n                 STAMP 21\n\n")
#!zip -r usage.zip /content

from google.colab import files
files.download('usage.zip')





print("\n\n                 STAMP 22\n\n")
'''