# USAGE_PY.PY
# Updated with Multi-Head Attention (MHA) for TD3
# Fixed for Stable Baselines 2 (TF 1.15)
# FIXES:
# 1. Inherits from TD3MlpPolicy to fix Variable Scoping ("No variables to optimize").
# 2. Explicitly sets self.policy in make_actor to fix the "NoneType" crash.
# 3. ADDED: Full logging (CSV, NPY) and Plot Saving (.png) functionality.

import numpy as np
import matplotlib.pyplot as plt
import math
import gym
import random
import tensorflow as tf  
import os
import time
from datetime import date

# Stable Baselines Imports
from stable_baselines.common.env_checker import check_env
from stable_baselines import TD3
from stable_baselines.common.noise import NormalActionNoise
from stable_baselines.bench import Monitor
from stable_baselines.common.callbacks import BaseCallback

# IMPORTANT: Inherit from the concrete implementation
from stable_baselines.td3.policies import MlpPolicy as TD3MlpPolicy

# Custom Simulation Imports
from aplysia_feeding_ub import AplysiaFeedingUB
from aplysia_feeding_b import AplysiaFeedingB

float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

##############################################################################################################################################
# ============================= Start by generating EXPERT model ============================== #
##############################################################################################################################################

unbreakable = True

if unbreakable:
    suffix = str(date.today())
    xlimits = [0,60]
    aplysia = AplysiaFeedingUB() 
    print('Swallowing (Unbreakable)')
    aplysia.SetSensoryStates('swallow')
    aplysia.RunSimulation()
else:
    suffix = str(date.today())
    xlimits = [0,60]
    aplysia = AplysiaFeedingB() 
    print('Swallowing (Breakable)')
    aplysia.SetSensoryStates('swallow')
    aplysia.RunSimulation()


##==========================================
## Verify Environment
##==========================================
env = gym.make("gym_slug:slug-v6") 
check_env(env)
print('env validated. Default args: foo={}, max_steps={}, threshold={}, patience={}'.format(env.foo, env.max_steps, env.threshold, env.patience))


##======================================
## Check expert performance
##======================================
std_env = env
std_env.preset_inputs = 1   
std_env.set_plotting(0)     
std_env.set_verbose(1)      

obs = std_env.reset()       
rewards = []                
steps = []
reward_log = []
num_episodes = 10

for i in range(num_episodes):
    if i == 9:
        std_env.set_plotting(1)
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

expert_avg_reward = np.mean(rewards)
print('expert_avg_reward = ', expert_avg_reward)


##==================================================
## Utility function
##==================================================
def compute_avg_return(environment, model, num_episodes=10, verbose = 0, drop_out = 0):
    total_return = 0.0
    record = np.zeros(num_episodes)
    for i in range(num_episodes):
        obs = environment.reset()  
        episode_return = 0.0        
        while True:
            action, _states = model.predict(obs)                
            obs, reward, done, info = environment.step(action)  
            episode_return += reward 
            if done:
                break
        record[i] = episode_return       
        total_return += episode_return  
        if verbose:
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


##============================================================
## Create environments
##============================================================
train_env = gym.make("gym_slug:slug-v6")
train_env.present_inputs = 0
train_env.set_verbose(0)
train_env.set_training(1)

eval_env = gym.make("gym_slug:slug-v6")
eval_env.set_verbose(0)
eval_env.present_inputs = 0
eval_env.set_training(1)

plot_env = gym.make("gym_slug:slug-v6")
plot_env.set_plotting(2)
plot_env.present_inputs = 0
plot_env.set_training(1)

##============================================================
## Define Custom Callbacks (Updated with Full Logging & Saving)
##============================================================

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, plot_env, eval_env, num_eval_episodes, log_interval: int, eval_interval: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.start_time = None
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_reward = -float('inf') 
        self.plot_env = plot_env
        self.eval_env = eval_env
        self.num_eval_episodes = num_eval_episodes
        self.eval_r_log = []
        self.eval_i_log = []
        self.eval_t_log = []  # Time log

    def _on_training_start(self):
        self.start_time = time.time()

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            print('Reach step {}'.format(self.n_calls))
        if self.n_calls % self.eval_interval == 0:
            avg_reward = compute_avg_return(self.eval_env, self.model, self.num_eval_episodes, 0, 0)
            self.eval_r_log.append(avg_reward)
            self.eval_i_log.append(self.n_calls)
            self.eval_t_log.append(time.time() - self.start_time)
            
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                print("best policy updated, avg_reward={}".format(avg_reward))
                avr_reward_str = self._printable_reward_title(avg_reward)
                self.model.save(self.save_path + avr_reward_str)        
                compute_avg_return(self.plot_env, self.model, 2, 0, 0)  
        return True
    
    def _on_training_end(self):
        # Calculate total time
        total = time.time() - self.start_time
        print(self.eval_r_log)
        
        # Use the configured log directory for saving
        # This ensures files go to 'Trial_IRL_TD3_Attention'
        B = self.log_dir
        if not B.endswith('/'): B += '/'
        T = '__Trained'

        try:
            print(f"Training finished! Total time: {total:.2f} seconds")
            if len(self.eval_r_log) > 0:
                print(f"Max reward: {max(self.eval_r_log):.2f} vs Expert: {expert_avg_reward:.2f}")

            # Save Reward Logs
            np.save(B+'reward'+T+'.npy', self.eval_r_log)
            np.savetxt(B+'reward'+T+'.csv', self.eval_r_log)
            
            # Save Time Logs
            np.save(B+'time'+T+'.npy', self.eval_t_log)
            np.savetxt(B+'time'+T+'.csv', self.eval_t_log)

            # Generate and Save Plot
            plt.figure()
            plt.plot(self.eval_i_log, self.eval_r_log)
            if 'expert_avg_reward' in globals():
                plt.hlines(expert_avg_reward, 0, max(self.eval_i_log), 'r', label='Expert')
            plt.ylabel('Average Return') 
            plt.xlabel('Iterations')
            plt.legend()
            plt.savefig(B+"reward.png")
            print(f"Results saved to {self.log_dir}")
            plt.show()
            
        except Exception as e:
            print(f"Error saving results: {e}")

    def _printable_reward_title(self, avg_reward) -> str:
        lhs = int(avg_reward)
        rhs = int((avg_reward - lhs) * 10000000000)
        return str(lhs)+"_"+str(rhs)


##============================================================
## Define Custom MHA Policy
##============================================================

class MhaTD3Policy(TD3MlpPolicy):
    """
    Custom TD3 Policy with Multi-Head Attention (MHA).
    INHERITANCE: Inherits from TD3MlpPolicy to handle make_critics logic automatically.
    OVERRIDE: Overrides make_actor and make_critic with MHA logic.
    """
    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, **kwargs):
        super(MhaTD3Policy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, **kwargs)

    def _dense(self, x, units, name, activation=None):
        """
        Standard dense layer creation using get_variable.
        """
        with tf.compat.v1.variable_scope(name):
            input_dim = int(x.get_shape()[-1])
            w = tf.compat.v1.get_variable("w", shape=[input_dim, units], 
                                initializer=tf.glorot_uniform_initializer())
            b = tf.compat.v1.get_variable("b", shape=[units], 
                                initializer=tf.zeros_initializer())
            
            output = tf.matmul(x, w) + b
            if activation:
                output = activation(output)
            return output

    def _layer_norm(self, x, name="layer_norm"):
        with tf.compat.v1.variable_scope(name):
            params_shape = [int(x.get_shape()[-1])]
            beta = tf.compat.v1.get_variable('beta', params_shape, initializer=tf.zeros_initializer())
            gamma = tf.compat.v1.get_variable('gamma', params_shape, initializer=tf.ones_initializer())
            mean, variance = tf.nn.moments(x, axes=[-1], keep_dims=True)
            normalized = (x - mean) / tf.sqrt(variance + 1e-6)
            return gamma * normalized + beta

    def attention_block(self, x, hidden_dim, num_heads=4):
        head_dim = hidden_dim // num_heads
        Q = self._dense(x, hidden_dim, name='Q')
        K = self._dense(x, hidden_dim, name='K')
        V = self._dense(x, hidden_dim, name='V')
        
        Q_ = tf.reshape(Q, [-1, num_heads, head_dim]) 
        K_ = tf.reshape(K, [-1, num_heads, head_dim])
        V_ = tf.reshape(V, [-1, num_heads, head_dim])
        
        scores = tf.matmul(Q_, K_, transpose_b=True)
        scale = tf.cast(head_dim, tf.float32) ** 0.5
        scores = scores / scale
        attn_weights = tf.nn.softmax(scores)
        
        output_heads = tf.matmul(attn_weights, V_)
        output_flat = tf.reshape(output_heads, [-1, hidden_dim])
        output_proj = self._dense(output_flat, hidden_dim, name='Out')
        output_norm = self._layer_norm(x + output_proj, name="attn_ln")
        return output_norm

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None: 
            obs = self.processed_obs

        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            # Network Architecture
            h1 = self._dense(obs, 256, name='fc1', activation=tf.nn.relu)
            h1_attn = self.attention_block(h1, hidden_dim=256, num_heads=4)
            h2 = self._dense(h1_attn, 256, name='fc2', activation=tf.nn.relu)
            pi_h = self._dense(h2, self.ac_space.shape[0], name='pi', activation=tf.nn.tanh)
            
            # CRITICAL FIX: We must assign self.policy to the actor's output tensor.
            # The base class relies on this attribute to perform inference (step()).
            self.policy = pi_h
            
            return pi_h

    def make_critic(self, obs=None, action=None, reuse=False, scope="qf1"):
        if obs is None: 
            obs = self.processed_obs

        # We assume the parent class has handled the outer "model/qf" scope.
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            q_in = tf.concat([obs, action], axis=-1)
            h1 = self._dense(q_in, 256, name='fc1', activation=tf.nn.relu)
            h1_attn = self.attention_block(h1, hidden_dim=256, num_heads=4)
            h2 = self._dense(h1_attn, 256, name='fc2', activation=tf.nn.relu)
            q_out = self._dense(h2, 1, name='q_out')
            return q_out

    # DO NOT OVERRIDE make_critics. Parent handles scoping.


##============================================================
## Train Setup
##============================================================

best_reward = 0
log_dir = '/home/camila/GymSlug-main/GymSlug/Trial_IRL_TD3_Attention/'
os.makedirs(log_dir, exist_ok=True)
train_env = Monitor(train_env, log_dir)

num_iterations = 1000000 
log_interval = 100       
num_eval_episodes = 5   
eval_interval = 500      

callback = SaveOnBestTrainingRewardCallback(
    plot_env, 
    eval_env, 
    num_eval_episodes, 
    log_interval=log_interval, 
    eval_interval=eval_interval, 
    log_dir=log_dir
)

n_actions = train_env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=0.2 * np.ones(n_actions)
)

model = TD3(
    MhaTD3Policy,          
    train_env,
    action_noise=action_noise,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    train_freq=100,        
    gradient_steps=100,    
    policy_delay=2,
    target_policy_noise=0.2,
    target_noise_clip=0.5,
    verbose=1,
    seed=12
)

print("Starting Training with Multi-Head Attention TD3...")
model.learn(total_timesteps=num_iterations, callback=callback)

callback._on_training_end()