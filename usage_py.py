# USAGE_PY.PY
# this file was made for running and plotting the expert model, then running and plotting the trained model --> set up to only run origional breakable/unbreakable environments
#
##===========================================
## Verify Environment with Stable Baselines |
##===========================================
#
##=======================================
## Check expert performance in gym_slug |
##=======================================
#
##===================================================
## Utility function for training process monitoring |
##===================================================
#
##=============================================================
## Create environments for training, evaluation, and plotting |
##=============================================================

'''
List of all imports made in code
=================================
# gym creates RL environment and Stabel Baseline implements the chosen RL Algorithm

import numpy as np                                  # for numpy arrays
import matplotlib.pyplot as plt                     # for plotting
import math                                         # for math functions
import matplotlib.patches as patches                # unused
import matplotlib.animation as animation            # unused
import gym                                          # for gym environment
import random                                       # for random number generator

from aplysia_feeding_ub import AplysiaFeedingUB     # for 2020 Full Swallowing Model
from datetime import date                           # for date and time stamp

from stable_baselines.common.policies import MlpPolicy          # for multi-layer perceptron policy
from stable_baselines.common.vec_env import DummyVecEnv         # unused
from stable_baselines.common.env_checker import check_env       # for checking enviroment parameters

from stable_baselines import DQN            # for DQN (discrete action space)
import os                                   # for terminal comunication
import gym                                  # repeat
import numpy as np                          # repeat
import matplotlib.pyplot as plt             # repeat
import math                                 # repeat
import matplotlib.patches as patches        # repeat
import matplotlib.animation as animation    # repeat

from stable_baselines import DDPG                                   # for DDPG (continuous action space) --> unused
from stable_baselines.ddpg.policies import LnMlpPolicy              # unused

from stable_baselines import results_plotter                        # unused
from stable_baselines.bench import Monitor                          # for monitoring trained data
from stable_baselines.results_plotter import load_results, ts2xy    # unused
from stable_baselines.common.noise import AdaptiveParamNoiseSpec    # unused
from stable_baselines.common.callbacks import BaseCallback          # for class parent
'''

##############################################################################################################################################
##############################################################################################################################################
# ================= Running all code took 23 min and after I closed all the plots the python file terminated (1 min later) ================= #
##############################################################################################################################################
##############################################################################################################################################
import numpy as np
#%matplotlib qt
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import matplotlib.animation as animation
import gym
import random
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})


##############################################################################################################################################
# ============================= Start by generating EXPERT model, this example is specifically for swallowing ============================== #
##############################################################################################################################################
## Import expert policy
# The goal is to learn a policy that 
# 1. generates similar motor neuron control/action with the expert
# 2. shows similar or higher average episode reward with the expert (~132.8)

from aplysia_feeding_ub import AplysiaFeedingUB
from aplysia_feeding_b import AplysiaFeedingB
from datetime import date

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
    #aplysia.GeneratePlots_WS('Swallow_'+suffix, xlimits)
    #aplysia.GenerateFiles_CJF('Swallow_'+suffix, xlimits, '__expert')

else:
    # Aplysia Feeding Driver Biting --> Breakable = Unloaded
    suffix = str(date.today())
    xlimits = [0,60]
    aplysia = AplysiaFeedingB() # Initialize simulation object
    print('Swallowing')
    aplysia.SetSensoryStates('swallow')
    aplysia.RunSimulation()
    #aplysia.GeneratePlots_WS('Swallow_'+suffix, xlimits)
    #aplysia.GenerateFiles_CJF('Swallow_'+suffix, xlimits, '__expert') #Expert_BreakableSwallowing/


##==========================================
## Verify Environment with Stable Baselines
##==========================================
# slug-v0: UnbreakableSeaweed, used here
# slug-v1: BreakableSeaweed
env = gym.make("gym_slug:slug-v0") 
# this environment supposibly is created by using class that inherits gym so when gym is called, 
# the custom class is called instead


from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
# Vectorized Environments are a method for stacking multiple independent environments into a single environment. 
# Instead of training an RL agent on 1 environment per step, it allows us to train it on n environments per step
from stable_baselines.common.env_checker import check_env

check_env(env)
print('env validated. Default args: foo={}, max_steps={}, threshold={}, patience={}'.format(env.foo, env.max_steps, env.threshold, env.patience))


##======================================
## Check expert performance in gym_slug
##======================================
# When environment is created, it is created using the slug_unbreakable environment so all calls bellow 
# are functions attributed to custom class
std_env = env
std_env.preset_inputs = 1   # used in MuscleActivations_001 function and sets neuron properties
std_env.set_plotting(0)     # 1 is yes you want plots and 0 is no you don't want any plots
std_env.set_verbose(1)      # 1 prints reset condition and any background information about what is going on during training

obs = std_env.reset()       # Initial state by calling reset to reset environment parameters
rewards = []                # Array of reward istory during training 
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
        # Will fill action array with expert neuron state -> 0 for off and 1 for on
        # Simulating through a test if RL environment functions work
        #       Will create an action array list called actionArray
        #       Fill up the actionArray with the neural model states during specified episode 
        #       Call step(action) function to learn rest of swallowing sequence
        #       Will {state, reward, done, info}
        #       Will continue until done returns true --> means sequence is completed
        #   Will repeat this num_episode times (but will always get the same reward output)
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

plt.figure()
for i in reward_log:
    plt.plot(i)
plt.show()

expert_avg_reward = np.mean(rewards)
print('expert_avg_reward = ', expert_avg_reward)





##==================================================
## Utility function for training process monitoring
##==================================================
def compute_avg_return(environment, model, num_episodes=10, verbose = 0, drop_out = 0):
    # compute_avg_return: function that returns the reward after training for specified amount of episodes in DQN model and specified environment
    total_return = 0.0
    record = np.zeros(num_episodes)
    for i in range(num_episodes):
        obs = environment.reset()  # for every episode reset environment
        episode_return = 0.0       # initialize episode reward to 0 
        while True:
            action, _states = model.predict(obs)                # returns observed action and state from DQN model
            obs, reward, done, info = environment.step(action)  # todo Wen: action shape! --> idk what this means but will take first step to get reward from action
            if verbose:
                print('ep:{}.rw:{}.act:{}.o_stat:{}.n_stat;{}'.format(i,time_step.reward.numpy(),action_step.action[0],old_step.observation.numpy()[0,:],time_step.observation.numpy()[0,:]))
            # keep looping and adding up reward until episode is done
            episode_return += reward # = or +=
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

# initiate evalation env
eval_env = gym.make("gym_slug:slug-v0")
eval_env.set_verbose(0)
eval_env.present_inputs = 0

# initiate an env with plotting
plot_env = gym.make("gym_slug:slug-v0")
plot_env.set_plotting(2)
plot_env.present_inputs = 0

## Train vanilla DQN. Periodically evalulate using callback
# training parameters
num_iterations = 500000 # @param {type:"integer"} or 1M #360000
log_interval = 100      # @param {type:"integer"}
num_eval_episodes = 5   # @param {type:"integer"}
eval_interval = 500     # @param {type:"integer"}



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
        # DQN model constructor that mostly records parameter info rather than model architecture
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
                avr_reward_str = self._printable_reward_title(avg_reward)
                self.model.save(self.save_path + avr_reward_str)        # save new model with best average output
                compute_avg_return(self.plot_env, self.model, 2, 0, 0)  # idk what this does???
            #elif avg_reward > 125: # what's the importance of 125???
            #    print('avg_return={}>125'.format(avg_reward))
            #    avr_reward_str = self._printable_reward_title(avg_reward)
            #    self.model.save(self.save_path + avr_reward_str)
            #    compute_avg_return(self.plot_env, self.model, 2, 0, 0)
        return True
    

    def _on_training_end(self):
        # for plotting
        print(self.eval_r_log)
        plt.plot(self.eval_i_log, self.eval_r_log)
        plt.hlines(expert_avg_reward, 0, max(self.eval_i_log), 'r')
        plt.ylabel('Average Return') 
        plt.xlabel('Iterations')
        plt.show()


    def _printable_reward_title(self, avg_reward) -> str:
        # for printting reward
        lhs = int(avg_reward)
        rhs = int((avg_reward - lhs) * 10000000000)
        return str(lhs)+"_"+str(rhs)
        
    




best_reward = 0
log_dir = '/home/camila/GymSlug-main/GymSlug/bestModel/bestModelBreak'
os.makedirs(log_dir, exist_ok=True)

train_env = Monitor(train_env, log_dir)

# Create the callback class obj
callback = SaveOnBestTrainingRewardCallback(plot_env, eval_env, num_eval_episodes, log_interval=log_interval, eval_interval=eval_interval, log_dir=log_dir)
# Create DQN model
model = DQN('MlpPolicy', train_env, prioritized_replay=True, learning_rate=2.5e-4,  verbose=1)
# Train the agent DQN model with callback class as one of the parameters
model.learn(total_timesteps=num_iterations, callback=callback)
# plot rewards over time
callback._on_training_end()

# classstable_baselines.deepq.DQN(policy, env, gamma=0.99, learning_rate=0.0005, buffer_size=50000, exploration_fraction=0.1, exploration_final_eps=0.02, 
# exploration_initial_eps=1.0, train_freq=1, batch_size=32, double_q=True, learning_starts=1000, target_network_update_freq=500, prioritized_replay=False, 
# prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-06, param_noise=False, n_cpu_tf_sess=None, 
# verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False, seed=None)
# !zip -r usage.zip /content

# from google.colab import files
# files.download('usage.zip')
