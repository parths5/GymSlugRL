## slug_unbreakable.py
## Holds class for unbreakable seaweed
#
#       def __init__(self, foo=0, max_steps=1000, threshold=-1000.0, delay=1, patience = 20, cr_threshold = -1000):
#       def set_plotting(self,toggle)
#       def set_verbose(self, inp)
#
#       def step(self,action) -> return self._state, reward, done, info
#       def reset(self) -> return self._state
#
#       def to_binary(self, num)
#
#       def MuscleActivations_001(self,action)
#       def Biomechanics_001(self) -> return reward[0]
#
#       def GeneratePlots(self,label)
#       def GeneratePlots_training(self,label)
#
#       def render(self, mode='human')
#       def close(self)
#       def RotationMatrixZ(self,theta)
#       def unitVec(self,origin,endpoint)
#       def Animate(self,j)
#       def VisualizeMotion(self)

      

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import matplotlib.animation as animation
import gym
import random
from datetime import date

from matplotlib.collections import PatchCollection
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
float_formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# Create class with inherited gym.Env (parent class)
class UnbreakableSeaweed(gym.Env):
    """
    Description: 
        A 1-D model of Aplysia californica feeding. Origional Biomechanical model (2020) and RL Neural model (focus on I2/B32, B38, B8a/b, B6/9/3, Force)
        The goal is to ingest the most edible food --> reward function
        
    Source:
        This enviornment cooresponds to the model of Aplysia feeding presented in 
        Control for Multifunctionality: Bioinspired Control Based on Feeding in Aplysia californica 2020
    
    Observation (7-element):
        Type: 
        Num      Observation          Min     Max
        0        x_h                  0       1
        1        x_g                  0       1
        2        force_on_object      -Inf    Inf
        3        pressure_grasper     -Inf    Inf
        4        pressure_jaws        -Inf    Inf
        5        edible               -1      1
        6        grasper_friction_state 0     1
        
    Actions (32-element):
        Type: 5 element array
        element 0 - B7 state
        element 1 - B6/B9/B3 state
        element 2 - B 8a/b state
        element 3 - B31/B32 state
        element 4 - B38 state
        Control frequency: 20 Hz
        
    Reward:
        Reward is proportional to the amount of seaweed ingested
        
    Episode Termination:
        Episode is greater than max_steps_per_iteration. Default: 1000
            
    """
    
    #################################
    # properties for visualization  #
    #################################
    
    # define the location of the ground plane
    x_ground = np.array([[0],[0]])
    len_ground_line = 5

    # define the location of the force transducer
    x_transducer = x_ground + np.array([[8],[0]])
    len_transducer_line = 5

    # define location and shape of head
    x_H = x_ground + np.array([[0],[0]])
    x_H_width = 1
    x_H_height = 4

    # define the extents of the grasper protraction/retraction path
    grasper_origin = x_H + np.array([[0],[0]])
    grasper_full = grasper_origin + np.array([[1],[0]])

    # define the starting position for the bottom of the grasper along this track
    x_R = grasper_origin + np.array([[0],[0]])

    # specify vectors based on the grasper in the upright position
    theta_grasper_initial = math.pi/2

    # specify the grasper radius
    r_grasper = 1
    grasper_offset = 1

    # define the positions of the I2 muscle origins
    x_I2_Borigin = grasper_origin + np.array([[0],[0]])
    x_I2_Aorigin = grasper_origin + np.array([[0],[2*r_grasper]])

    # define the position of the hinge origin
    x_hinge_origin = grasper_origin + np.array([[0],[0]])

    # specify the angle relative to horizontal for each of the attachment points fixed on the grasper surface
    theta_s = 0
    theta_I2_A = math.pi/6
    theta_I2_B = math.pi/6
    
    # plot line representing ground
    line_ground =[]
    # plot a dot at the origin
    dot_ground =[]
    # plot line representing force transducer
    line_transducer =[]
    # plot line representing track
    line_grapser_track =[]
    # plot line from R to G
    line_RG =[]
    # plot dot at point R
    dot_R =[]
    # plot dot at point G
    dot_G =[]
    # plot dot at point S
    dot_S =[]
    # plot dot at I2 attachment point A
    dot_I2_A =[]
    # plot dot at I2 attachment point B
    dot_I2_B =[]
    # plot dot at I2 attachment point A
    dot_I2_Aorigin =[]
    # plot dot at I2 attachment point B
    dot_I2_Borigin =[]
    # draw grasper
    draw_circle =[]

    # draw head
    head =[]
    dot_H_spring =[]
    # draw head spring as dashed line
    line_H_spring =[]
    # draw grasper to head spring as dashed line
    line_G_spring =[]
    
    # other variables --> not used in env code below??? NOT TRUE CAUSE USED IN MUSCLE ACTIVATION FUNCTION BUT IDK WHAT IT DOES???
    preset_inputs = 0
    
    generat_plots_toggle = 0
    init_reward = 0.0
    init_force_level = 'low'
    high_threshold = 4
    low_threshold = 40

    # saved from expert code, only for visualization
    output_expert_mean = np.load('output_expert_mean.npy')
    output_expert_std = np.load('output_expert_std.npy')


    def __init__(self, foo=0, max_steps=1000, threshold=-1000.0, delay=1, patience=20, cr_threshold=-1000):
        # === Preallocate arrays ============================================================================================================================================================================================================
        #   Parameters of Constructor
        #       foo : unkown value used when calculating reward
        #       max_steps : epochs
        #       threshold : used to check when to reset
        #       delay : unknown use
        #       patience : used to penalize reward if swallowing is not cyclical
        #       cr_threshold : used to check when to reset

        # saved from expert code, only for visualization
        self.output_expert_mean = np.load('output_expert_mean.npy') # load expert mean
        self.output_expert_std = np.load('output_expert_std.npy')   # load expert standard deviation 


        ####################################
        # Initial Biomechanical Variables  #
        ####################################
        self.biomechanicsModel = 1
        self.generat_plots_toggle = 0
        self.verbose = 0
        self.cr_threshold = cr_threshold
        self.patience = patience
        self.delta_gm = 0
        self.idle_count = 0
        self.gfs = 0
        self.foo = foo
        self.threshold = threshold
        self.total_reward = 0
        self.total_reward_log = [self.total_reward]
        self.reward_range = (-1e6, 1e6)

        # In OG model, these variables are all state variables --> arrays instantiated in global space and defined in init function
        # BUT these are the specific initial conditions (starting in line 292 in "apysia_feeding_ub.py")
        # muscle initial state variables
        self.P_I4 = 0
        self.A_I4 = 0.05
        self.P_I3_anterior = 0
        self.A_I3_anterior = 0.05
        self.T_I3 = 0.05
        self.A_I3 = 0.05
        self.T_I2 = 0.05
        self.A_I2 = 0.05
        self.T_hinge = 0
        self.A_hinge = 0.05

        # body initial state variables
        self.x_h = 0.0
        self.x_g = 0.0
        self.force_on_object = 0
        
        # Friction coefficients (gloabl variables in OG model)
        self.mu_s_g = 0.4               # mu_s coefficient of static friction at grasper
        self.mu_k_g = 0.3               # mu_k coefficient of kinetic friction at grasper
        self.mu_s_h = 0.3               # mu_s coefficient of static friction at jaws
        self.mu_k_h = 0.3               # mu_k coefficient of kinetic friction at jaws
        
        # Maximum muscle forces (gloabl variables in OG model)
        self.max_I4 = 1.75              # Maximum pressure grasper can exert on food
        self.max_I3ant = 0.6            # Maximum I3 anterior force
        self.max_I3 = 1                 # Maximum I3 force
        self.max_I2 = 1.5               # Maximum I2 force
        self.max_hinge = 0.2            # Maximum hinge force

        # Muscle time constants (gloabl variables in OG model)
        self.tau_I4 = 1.0/np.sqrt(2)              # time constant (in seconds) for I4 activation
        self.tau_I3anterior = 2.0/np.sqrt(2)      # time constant (in seconds) for I3anterior activation
        self.tau_I2_ingestion = 0.5*1/np.sqrt(2)  # time constant (in seconds) for I2 activation during ingestion
        self.tau_I2_egestion = 1.4*1/np.sqrt(2)   # time constant (in seconds) for I2 activation during egestion
        self.tau_I3 = 1.0/np.sqrt(2)              # time constant (in seconds) for I3 activation
        self.tau_hinge  = 1.0/np.sqrt(2)          # time constant (in seconds) for hinge activation
        self.TimeStep_h = 0.05

        # body time constants (gloabl variables in OG model)
        self.c_g = 1.0                  # time constant (in seconds) for grapser motion
        self.c_h = 1.0                  # time constant (in seconds) for body motion

        # Spring constants (gloabl variables in OG model)
        self.K_sp_h = 2.0       # spring constant representing neck and body between head and ground
        self.K_sp_g = 0.1       # spring constant representing attachment between buccal mass and head

        # Reference points for springs (gloabl variables in OG model)
        self.x_h_ref = 0.0      # head spring reference position
        self.x_gh_ref = 0.4     # grasper spring reference position

        # environment variable (OG is 10)
        self.seaweed_strength = 100 

        #########################
        # Setting Up DQN Space  #
        #########################
        
        # action space
        # self.action_space = gym.spaces.Discrete(32) # for all 2^5 possible discrete choices of actions at each step
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        # observation space
        self.x_g_threshold = 1
        self.x_h_threshold = 1
        self.sens_mechanical_grasper = 1
        
        high = np.array([self.x_h_threshold*2,      # max x_h
                         self.x_g_threshold*2,      # max x_g
                         np.finfo(np.float32).max,  # max force_on_object
                         np.finfo(np.float32).max,  # max pressure_grasper
                         np.finfo(np.float32).max,  # max Pressure_jaws
                         1,                         # max edible
                         1],                        # max grasper_friction_state
                        dtype=np.float32)

        self.observation_space = gym.spaces.Box(low = -high,
                                                high = high,
                                                dtype = np.float32)
        self._state = np.array([0,0,0,0,0,1,0]) 
        
        self._episode_ended = False
        self.steps_beyond_done = None   
        self.max_steps = max_steps
        self.current_step = 1

        # history vectors are used in step, reset, plotting, and visualization functions
        # Sensory State History --> all nan
        self.sens_mechanical_grasper_history = np.zeros((1,self.max_steps+1))
        self.sens_chemical_lips_history = np.zeros((1,self.max_steps+1))
        self.sens_mechanical_lips_history = np.zeros((1,self.max_steps+1))

        # Cerebral Interneuron History --> all nan
        self.CBI2_history = np.zeros((1,self.max_steps+1))
        self.CBI3_history = np.zeros((1,self.max_steps+1))
        self.CBI4_history = np.zeros((1,self.max_steps+1))

        # Buccal Interneuron History --> all nan
        self.B64_history = np.zeros((1,self.max_steps+1))
        self.B20_history = np.zeros((1,self.max_steps+1))
        self.B40B30_history = np.zeros((1,self.max_steps+1))
        self.B4B5_history = np.zeros((1,self.max_steps+1))
        
        # Body State History
        self.x_g_history=np.zeros((1,self.max_steps+1))
        self.x_h_history=np.zeros((1,self.max_steps+1))
        self.force_history=np.zeros((1,self.max_steps+1))
        self.grasper_friction_state_history=np.zeros((1,self.max_steps+1))

        # Buccal Motor Neuron History
        self.B8_history=np.zeros((1,self.max_steps+1))
        self.B38_history=np.zeros((1,self.max_steps+1))
        self.B7_history=np.zeros((1,self.max_steps+1))
        self.B31B32_history=np.zeros((1,self.max_steps+1))
        self.B6B9B3_history=np.zeros((1,self.max_steps+1))

        # Muscle State Variables
        self.P_I4_history=np.zeros((1,self.max_steps+1))
        self.P_I3_anterior_history=np.zeros((1,self.max_steps+1))
        self.T_I3_history=np.zeros((1,self.max_steps+1))
        self.T_I2_history=np.zeros((1,self.max_steps+1))
        self.T_hinge_history=np.zeros((1,self.max_steps+1))
        self.theta_g=np.zeros((1,self.max_steps+1)) # ok

        # Initialize body state history
        self.x_g_history[0,0] = self.x_g
        self.x_h_history[0,0] = self.x_h
        self.force_history[0,0]=0
        self.grasper_friction_state_history[0,0] = 0
        self.theta_g[0,0] = 0
        
        # Timing variables
        self.StartingTime = 0
        self.TimeStep = 0.05
        self.EndTime = self.max_steps*0.05


    def set_plotting(self,toggle):
        # === Plot Setter ============================================================================================================================================================================================================
        self.generat_plots_toggle = toggle
        print("plotting set to", toggle)
    

    def set_verbose(self, inp):
        # === Verbose Setter ============================================================================================================================================================================================================
        #   option to print out text during training process
        self.verbose = inp
        print("verbose set to", inp)
    

    def step(self,action): # return self._state, reward, done, info
        # === Step ============================================================================================================================================================================================================
        #   one of the three required functions for DQN environment
        term_stat = -1 # (0: reach max_steps)  (1: out of bound)
        if self._episode_ended:
            # this is for the cycle AFTER ending the episode --> all if statements below will make _episodes_ended = true to reset, return current state, and begin a new episode
            return self.reset()
        
        if self.current_step == self.max_steps:
            # this will end the episode if the defined max_steps for an episode is reached --> max_steps = 1000 and usually this is the case
            term_stat = 0
            if self.verbose == 1: print('reset - current_step == self.max_steps')
            self._episode_ended = True # set true so the next step calls the reset function
        elif (self.total_reward < self.cr_threshold) or (self.total_reward < self.threshold and self.current_step > 6*20):
            # this will end the episode if the episode's reward < -1000 (meaning something is wrong since reward should be positive)
            # or the episode's reward < -1000 AND the amount of steps in this episode > 120 (meaning again something went wrong is a little amount of steps, so early)
            if self.verbose == 1: print('reset early stop- total_reward={}@step {}'.format(self.total_reward, self.current_step))
            self._episode_ended = True # set true so the next step calls the reset function
        elif self.idle_count >= 100: 
            # 5 s, 20 steps/s --> if idle for >= 90 steps, early stop
            # don't want optimal feeding cycle to be just holding the seaweed in the mouth without swallowing --> if shoreter than count of 100, penalized below
            if self.verbose == 1: print('reset early stop- idle too long:{} steps'.format(self.idle_count))
            self._episode_ended = True # set true so the next step calls the reset function
        elif self.x_h < -0.2 or self.x_h > 1.2 or self.x_g < -0.2 or self.x_g > 1.2:
            # this will end the episode if head positions < -.2 OR > 1.2 OR grasper postion < -.2 OR > 1.2 
            # positions for both grasper and head should be in the interval [0,1] so don't want it to be out of bounds
            term_stat = 1
            if self.verbose == 1: print('reset - x_h or x_g out!: x_h: {} x_g: {}'.format(self.x_h, self.x_g))
            self._episode_ended = True # set true so the next step calls the reset function
        else:
            # if no need to end episode, update next states and actions --> SUPER IMPORTANT
            # obtain current x_h, x_g, force_on_object
            [x_h, x_g, force_on_object, pressure_grasper, pressure_jaws, edible, grasper_friction_state] = self._state # DEFINE PREVIOUS STATE VARIABLES
            
            if edible == 1:
                self.fixation_type = 0 # if 0 = unloaded, if 1 = loaded
            else:
                self.fixation_type = 0

            tmp = self.Biomechanics_001() # GET REWARD FROM BIOMECHANICS FUNCTION
            reward = tmp * 100 
         
            self.MuscleActivations_001(action) # DETERMINE NEW STATE SPACE USING ACTION 

            self._state = np.array([self.x_h, self.x_g, self.force_on_object,self.P_I4, self.P_I3_anterior,edible, self.grasper_friction_state],dtype=np.float32) # DEFINE NEW STATE VARIABLES
            
            self.current_step += 1 # INCREMENT AMOUNT OF STEPS TAKEN IN EPISODE

            #print("====== Current Step: ", self.current_step)
            #print(self.force_history)

            # Record new state variables in history arrays
            self.x_g_history[0,self.current_step] = self.x_g
            self.x_h_history[0,self.current_step] = self.x_h
            self.force_history[0,self.current_step] = self.force_on_object
            self.grasper_friction_state_history[0,self.current_step] = self.grasper_friction_state

            # PENALIZE REWARD IF IDLE --> WILL BE USED IN ABOVE IF STATEMENTS TO TERMINATE RUN
            if self.grasper_friction_state == self.gfs and self.delta_gm == 0:
                self.idle_count += 1
            else:
                self.idle_count = 0
                self.gfs = self.grasper_friction_state
            if self.idle_count >= self.patience:
                reward -= 1

            # Record current action variables as well as other useful variables in history arrays
            self.B8_history[0,self.current_step] = self.B8
            self.B38_history[0,self.current_step] = self.B38
            self.B6B9B3_history[0,self.current_step] = self.B6B9B3
            self.B31B32_history[0,self.current_step] = self.B31B32
            self.B7_history[0,self.current_step] = self.B7
            
            # more history
            self.P_I4_history[0,self.current_step] = self.P_I4
            self.P_I3_anterior_history[0,self.current_step] = self.P_I3_anterior
            self.T_I3_history[0,self.current_step] =self.T_I3
            self.T_I2_history[0,self.current_step] = self.T_I2
            self.T_hinge_history[0,self.current_step] =self.T_hinge

            self.sens_mechanical_grasper_history[0,self.current_step] = math.nan
            self.sens_chemical_lips_history[0,self.current_step] = math.nan
            self.sens_mechanical_lips_history[0,self.current_step] = math.nan
            self.CBI2_history[0,self.current_step] = math.nan
            self.CBI3_history[0,self.current_step] = math.nan
            self.CBI4_history[0,self.current_step] = math.nan
            self.B64_history[0,self.current_step] = math.nan
            self.B20_history[0,self.current_step] = math.nan
            self.B40B30_history[0,self.current_step] = math.nan
            self.B4B5_history[0,self.current_step] = math.nan 

        if self._episode_ended:
            reward = 0.0 # reset reward --> technically this would be called when no reward was calculated from the biomechanics function so need to define "reward" to make code work?
            # can define here what you want to output when the episode has been terminated
            if self.generat_plots_toggle == 1:
                # CAMILA UPDATES TO CHECK IF SAVING FILES WORK
                self.GeneratePlots('Plot_'+str(date.today()))
                # save force trajectory
                self.EndTime = (self.current_step - 1) * self.TimeStep
                t=np.atleast_2d(np.arange(self.StartingTime,self.EndTime+self.TimeStep,self.TimeStep))
                self.EndTime = self.max_steps*0.05
                end_ind = t.shape[1]
                arr = np.concatenate([t.transpose(),self.force_history[0,:end_ind].reshape(-1,1)], axis = 1)
                #used to determine what optim force would look like using optim motor neurons
                #np.save('Optimized_Swallows/Force_single__optim.npy', arr)
                #np.savetxt('Optimized_Swallows/Force_single__optim.csv', arr)
                #np.save('Trial_1B/GrasperMotion__trained.npy', arr)
                #np.savetxt('Trial_1B/GrasperMotion__trained.csv', arr)
                #self.GeneratePlotsFiles_CJF('Plot_'+str(date.today()), '__Expert')
            elif self.generat_plots_toggle == 2:
                self.GeneratePlots_training('Plot_'+str(date.today()))
                self.GeneratePlotsFiles_training_CJF('Plot_'+str(date.today()), 'Trial_20B/', '__Trained')
            self.total_reward += reward # no more reward to update total reward --> idk why this line of code is here honestly
            self.total_reward_log.append(self.total_reward) # record terminated episode's total reward to log array 
            # will return state space, reward calculated during step 
            return self._state, reward, True, {}
        else:
            # episode has NOT been terminated so update reward as it should be
            self.total_reward += reward
            self.total_reward_log.append(self.total_reward)
            return self._state, reward, False, {}


    def reset(self): # return self._state
        # === Reset ============================================================================================================================================================================================================
        # print(" ====> Reset")
        # Purpose of this function is to reset all variables back to their initial state, this includes state variables, helper variables, and all history arrays --> ACTION SPACE IS NOT RESET BECAUSE IT WILL BE CALCULATED

        # RESET STATE VARTIABLES TO RANDOM VALUES BETWEEN 0 AND 0.05
        lb, ub = 0.0, 0.05
        self.x_h = random.uniform(lb, ub)
        self.x_g = random.uniform(lb, ub)
        self.force_on_object =random.uniform(lb, ub)
        pressure_grasper = random.uniform(lb, ub)
        pressure_jaws = random.uniform(lb, ub)
        edible = 1 # if unbreakable, why is it edible???? --> Because it still needs to be swallowed, not rejected
        
        self._state = np.array([self.x_h, self.x_g, self.force_on_object,pressure_grasper, pressure_jaws,edible,0],dtype=np.float32)
        self._episode_ended = False # reset for next episode
        self.current_step = 1
        self.total_reward = 0

        self.total_reward_log = [self.total_reward]
        self.gfs = 0 # CHECK WHAT THIS VARIABLE IS??? --> Grasper_Friction_State
        self.idle_count = 0
        self.delta_gm = 0
        
        # reset neuron history to empty arrays (length 1000)
        self.sens_mechanical_grasper_history = np.zeros((1,self.max_steps+1))
        self.sens_chemical_lips_history = np.zeros((1,self.max_steps+1))
        self.sens_mechanical_lips_history = np.zeros((1,self.max_steps+1))
        self.CBI2_history = np.zeros((1,self.max_steps+1))
        self.CBI3_history = np.zeros((1,self.max_steps+1))
        self.CBI4_history = np.zeros((1,self.max_steps+1))
        self.B64_history = np.zeros((1,self.max_steps+1))
        self.B20_history = np.zeros((1,self.max_steps+1))
        self.B40B30_history = np.zeros((1,self.max_steps+1))
        self.B4B5_history = np.zeros((1,self.max_steps+1))

        # reset state and action space history (as well as muscle history) to empty arrays
        self.x_g_history=np.zeros((1,self.max_steps+1))
        self.x_h_history=np.zeros((1,self.max_steps+1))
        self.force_history=np.zeros((1,self.max_steps+1))
        self.grasper_friction_state_history=np.zeros((1,self.max_steps+1))
        self.B8_history=np.zeros((1,self.max_steps+1))
        self.B38_history=np.zeros((1,self.max_steps+1))
        self.B7_history=np.zeros((1,self.max_steps+1))
        self.B31B32_history=np.zeros((1,self.max_steps+1))
        self.B6B9B3_history=np.zeros((1,self.max_steps+1))
        self.P_I4_history=np.zeros((1,self.max_steps+1))
        self.P_I3_anterior_history=np.zeros((1,self.max_steps+1))
        self.T_I3_history=np.zeros((1,self.max_steps+1))
        self.T_I2_history=np.zeros((1,self.max_steps+1))
        self.T_hinge_history=np.zeros((1,self.max_steps+1))

        # Set first state history varibles as previous random numbers selected above 
        self.x_g_history[0,0] = self.x_g
        self.x_h_history[0,0] = self.x_h
        self.force_history[0,0]=0
        self.grasper_friction_state_history[0,0] = 0
        
        self.StartingTime = 0
        self.TimeStep = 0.05
        self.EndTime = self.max_steps*0.05
        
        # return reset state space (with random numbers calculated above)
        return self._state


    # def to_binary(self, num): # return [int(i) for i in tmp]
    #     # === Convert to binary value ============================================================================================================================================================================================================
    #     tmp = np.binary_repr(num) # e.g., '11001'
    #     if len(tmp) < 5:
    #         tmp = '0'* (5-len(tmp)) + tmp
    #     return [int(i) for i in tmp]


    def MuscleActivations_001(self,action):
        # === Set Muscle Activation Curves  ==========================================================================================================================================================================
        if isinstance(action, (int, np.integer)):
            # make sure to convert all variables in action space to discrete binary variables
            action = self.to_binary(action)
        if self.preset_inputs == 1: # TECHNICALLY THIS VALUE WILL NEVER BE 1 SINCE ONLY DEFINED/CALLED ONCE BEFORE CONSTRUCTOR AND SET AS 0
            # Name action state variables to make calcualtions easier
            self.B8 = action[2]
            self.B38 = action[4]
            self.B6B9B3 = action[1]
            self.B31B32 = action[3]
            self.B7 = action[0]
        
        # the following code works with the python environment without a tf wrapper --> WHAT DOES THIS MEAN??? --> tf = tensor flow wrapper
        # takes into consideration all different format types of action variable
        elif isinstance(action, list):
            self.B8 = action[2]
            self.B38 = action[4]
            self.B6B9B3 = action[1]
            self.B31B32 = action[3]
            self.B7 = action[0]
        else:    # assumed continuous action array from SAC model.predict(obs)
            self.B8 = action[2] # CL
            self.B38 = action[4] # CL
            self.B6B9B3 = action[1] # CL
            self.B31B32 = action[3] # CL
            self.B7 = action[0] # CL
        
        edible = self._state[5]
        
        ## Update I4: If food present, and grasper closed, then approaches
        # pmax pressure as dp/dt=(B8*pmax-p)/tau_p.  Use a quasi-backward-Euler
        self.P_I4=((self.tau_I4*self.P_I4+self.A_I4*self.TimeStep_h)/(self.tau_I4+self.TimeStep_h)) # old -- keep this version
        self.A_I4=((self.tau_I4*self.A_I4+self.B8*self.TimeStep_h)/(self.tau_I4+self.TimeStep_h))

        ## Update pinch force: If food present, and grasper closed, then approaches
        # pmax pressure as dp/dt=(B8*pmax-p)/tau_p.  Use a quasi-backward-Euler
        self.P_I3_anterior=(self.tau_I3anterior*self.P_I3_anterior+self.A_I3_anterior*self.TimeStep_h)/(self.tau_I3anterior+self.TimeStep_h)
        self.A_I3_anterior=(self.tau_I3anterior*self.A_I3_anterior+(self.B38+self.B6B9B3)*self.TimeStep_h)/(self.tau_I3anterior+self.TimeStep_h)

        ## Update I3 (retractor) activation: dm/dt=(B6-m)/tau_m
        self.T_I3=(self.tau_I3*self.T_I3+self.TimeStep_h*self.A_I3)/(self.tau_I3+self.TimeStep_h)
        self.A_I3=(self.tau_I3*self.A_I3+self.TimeStep_h*self.B6B9B3)/(self.tau_I3+self.TimeStep_h)

        ## Update I2 (protractor) activation: dm/dt=(B31-m)/tau_m.  quasi-B-Eul.
        self.T_I2=((self.tau_I2_ingestion*edible+self.tau_I2_egestion*(1-edible))*self.T_I2+self.TimeStep_h*self.A_I2)/((self.tau_I2_ingestion*edible+self.tau_I2_egestion*(1-edible))+self.TimeStep_h)
        self.A_I2=((self.tau_I2_ingestion*edible+self.tau_I2_egestion*(1-edible))*self.A_I2+self.TimeStep_h*self.B31B32)/((self.tau_I2_ingestion*edible+self.tau_I2_egestion*(1-edible))+self.TimeStep_h)

        ## Update Hinge activation: dm/dt=(B7-m)/tau_m.  quasi-B-Eul.
        # bvec(12,j+1)=(tau_m*hinge_last+dt*B7_last)/(tau_m+dt) # old
        self.T_hinge=(self.tau_hinge*self.T_hinge+self.TimeStep_h*self.A_hinge)/(self.tau_hinge+self.TimeStep_h)#new
        self.A_hinge=(self.tau_hinge*self.A_hinge+self.TimeStep_h*self.B7)/(self.tau_hinge+self.TimeStep_h)
    

    def Biomechanics_001(self): # return reward[0]
        # === Set Biomechanics  ==========================================================================================================================================================================
        edible = self._state[5] # should always be 1 for yes edible
        old_gm = np.array([self.x_g - self.x_h]) # gm = grasper motion AND gh = grasper/head
        
        unbroken = 1 # tracking variable to keep track of seaweed being broken off during feeding
        x_gh = self.x_g-self.x_h

        ## Grasper Forces
        # all forces in form F = Ax+b
        x_vec = np.array([[self.x_h],[self.x_g]])

        F_I2 = self.max_I2*self.T_I2*np.dot(np.array([1,-1]),x_vec) + self.max_I2*self.T_I2*1 #FI2 = FI2_max*T_I2*(1-(xg-xh))
        F_I3 = self.max_I3*self.T_I3*np.dot(np.array([-1,1]),x_vec)-self.max_I3*self.T_I3*0 #FI3 = FI3_max*T_I3*((xg-xh)-0)
        F_hinge = (x_gh>0.5)*self.max_hinge*self.T_hinge*np.dot(np.array([-1,1]),x_vec)-(x_gh>0.5)*self.max_hinge*self.T_hinge*0.5 #F_hinge = [hinge stretched]*F_hinge_Max*T_hinge*((xg-xh)-0.5)
        F_sp_g = self.K_sp_g*np.dot(np.array([1,-1]),x_vec)+self.K_sp_g*self.x_gh_ref #F_sp,g = K_g((xghref-(xg-xh))

        F_I4 = self.max_I4*self.P_I4
        F_I3_ant = (self.max_I3ant*self.P_I3_anterior*np.dot(np.array([1,-1]),x_vec)+self.max_I3ant*
                    self.P_I3_anterior*1)#: pinch force

        # calculate F_f for grasper
        if(self.fixation_type == 0): # object is not fixed to a contrained surface
            # F_g = F_I2+F_sp_g-F_I3-F_hinge # if the object is unconstrained it does not apply a resistive force back on the grasper. Therefore the force is just due to the muscles

            A2 = (1/self.c_g*(self.max_I2*self.T_I2*np.array([1,-1])+self.K_sp_g*np.array([1,-1])
                              -self.max_I3*self.T_I3*np.array([-1,1])-self.max_hinge*self.T_hinge*
                              (x_gh>0.5)*np.array([-1,1])))
            B2 = (1/self.c_g*(self.max_I2*self.T_I2*1+self.K_sp_g*self.x_gh_ref+self.max_I3*self.T_I3*
                              0+(x_gh>0.5)*self.max_hinge*self.T_hinge*0.5))
            A21 = A2[0]
            A22 = A2[1]

            # the force on the object is approximated based on the friction
            if(abs(F_I2+F_sp_g-F_I3-F_hinge) <= abs(self.mu_s_g*F_I4)): # static friction is true
                F_f_g = -self.sens_mechanical_grasper*(F_I2+F_sp_g-F_I3-F_hinge)
                self.grasper_friction_state = 1
            else:
                F_f_g = self.sens_mechanical_grasper*self.mu_k_g*F_I4
                # specify sign of friction force
                F_f_g = -(F_I2+F_sp_g-F_I3-F_hinge)/abs(F_I2+F_sp_g-F_I3-F_hinge)*F_f_g
                self.grasper_friction_state = 0

        elif (self.fixation_type == 1): # object is fixed to a contrained surface
            if unbroken:
                if(abs(F_I2+F_sp_g-F_I3-F_hinge) <= abs(self.mu_s_g*F_I4)): # static friction is true
                    F_f_g = -self.sens_mechanical_grasper*(F_I2+F_sp_g-F_I3-F_hinge)

                    # F_g = F_I2+F_sp_g-F_I3-F_hinge + F_f_g
                    self.grasper_friction_state = 1

                    # identify matrix components for semi-implicit integration
                    A21 = 0
                    A22 = 0
                    B2 = 0
                else:
                    F_f_g = -np.sign(F_I2+F_sp_g-F_I3-F_hinge)[0]*self.sens_mechanical_grasper*self.mu_k_g*F_I4

                    # specify sign of friction force
                    # F_g = F_I2+F_sp_g-F_I3-F_hinge + F_f_g
                    self.grasper_friction_state = 0

                    # identify matrix components for semi-implicit integration
                    A2 = (1/self.c_g*(self.max_I2*self.T_I2*np.array([1,-1])+self.K_sp_g*np.array([1,-1])
                                      -self.max_I3*self.T_I3*np.array([-1,1])-self.max_hinge*self.T_hinge*
                                      (x_gh>0.5)*np.array([-1,1])))
                    B2 = (1/self.c_g*(self.max_I2*self.T_I2*1+self.K_sp_g*self.x_gh_ref+self.max_I3*self.T_I3
                                      *0+(x_gh>0.5)*self.max_hinge*self.T_hinge*0.5+F_f_g))

                    A21 = A2[0]
                    A22 = A2[1]

            else:
                # F_g = F_I2+F_sp_g-F_I3-F_hinge #if the object is unconstrained it does not apply a resistive force back on the grasper. Therefore the force is just due to the muscles

                A2 = (1/self.c_g*(self.max_I2*self.T_I2*np.array([1,-1])+self.K_sp_g*np.array([1,-1])-self.max_I3
                                  *self.T_I3*np.array([-1,1])-self.max_hinge*self.T_hinge*(x_gh>0.5)
                                  *np.array([-1,1])))
                B2 = (1/self.c_g*(self.max_I2*self.T_I2*1+self.K_sp_g*self.x_gh_ref+self.max_I3*self.T_I3*
                                  0+(x_gh>0.5)*self.max_hinge*self.T_hinge*0.5))

                A21 = A2[0]
                A22 = A2[1]

                # the force on the object is approximated based on the friction
                if(abs(F_I2+F_sp_g-F_I3-F_hinge) <= abs(self.mu_s_g*F_I4)): # static friction is true
                    F_f_g = -self.sens_mechanical_grasper*(F_I2+F_sp_g-F_I3-F_hinge)
                    self.grasper_friction_state = 1
                else:
                    F_f_g = self.sens_mechanical_grasper*self.mu_k_g*F_I4
                    # specify sign of friction force
                    F_f_g = -(F_I2+F_sp_g-F_I3-F_hinge)/abs(F_I2+F_sp_g-F_I3-F_hinge)*F_f_g
                    self.grasper_friction_state = 0

        # [j*dt position_grasper_relative I2 F_sp I3 hinge GrapserPressure_last F_g]

        ## Body Forces
        # all forces in the form F = Ax+b
        F_sp_h = self.K_sp_h*np.dot(np.array([-1,0]),x_vec)+self.x_h_ref*self.K_sp_h
        # all muscle forces are equal and opposite
        if(self.fixation_type == 0):     # object is not constrained
            # F_h = F_sp_h #If the object is unconstrained it does not apply a force back on the head. Therefore the force is just due to the head spring.

            A1 = 1/self.c_h*self.K_sp_h*np.array([-1,0])
            B1 = 1/self.c_h*self.x_h_ref*self.K_sp_h

            A11 = A1[0]
            A12 = A1[1]

            if(abs(F_sp_h+F_f_g) <= abs(self.mu_s_h*F_I3_ant)): # static friction is true
                F_f_h = -self.sens_mechanical_grasper*(F_sp_h+F_f_g) # only calculate the force if an object is actually present
                self.jaw_friction_state = 1
            else:
                F_f_h = -np.sign(F_sp_h+F_f_g)[0]*self.sens_mechanical_grasper*self.mu_k_h*F_I3_ant # only calculate the force if an object is actually present
                self.jaw_friction_state = 0

        elif (self.fixation_type == 1):
            # calcuate friction due to jaws
            if unbroken: #if the seaweed is intact
                if(abs(F_sp_h+F_f_g) <= abs(self.mu_s_h*F_I3_ant)): # static friction is true
                    F_f_h = -self.sens_mechanical_grasper*(F_sp_h+F_f_g) # only calculate the force if an object is actually present
                    # F_h = F_sp_h+F_f_g + F_f_h
                    self.jaw_friction_state = 1

                    A11 = 0
                    A12 = 0
                    B1 = 0

                else:
                    F_f_h = -np.sign(F_sp_h+F_f_g)[0]*self.sens_mechanical_grasper*self.mu_k_h*F_I3_ant # only calculate the force if an object is actually present
                    # F_h = F_sp_h+F_f_g + F_f_h

                    self.jaw_friction_state = 0

                    if (self.grasper_friction_state == 1): # object is fixed and grasper is static  
                    # F_f_g = -mechanical_in_grasper*(F_I2+F_sp_g-F_I3-F_Hi)
                        A1 = (1/self.c_h*(self.K_sp_h*np.array([-1,0])+(-self.sens_mechanical_grasper*
                                                                       (self.max_I2*self.T_I2*np.array([1,-1])
                                                                        +self.K_sp_g*np.array([1,-1])-self.max_I3*
                                                                        self.T_I3*np.array([-1,1])-self.max_hinge*
                                                                        self.T_hinge*(x_gh>0.5)*np.array([-1,1]))
                                          -np.sign(F_sp_h+F_f_g)[0]*self.sens_mechanical_grasper*self.mu_k_h*
                                                                       self.max_I3ant*self.P_I3_anterior
                                                                       *np.array([1,-1]))))
                        B1 = (1/self.c_h*(self.x_h_ref*self.K_sp_h+(-self.sens_mechanical_grasper*(self.max_I2*self.T_I2*1+self.K_sp_g*self.x_gh_ref+self.max_I3*self.T_I3*0+(x_gh>0.5)*self.max_hinge*self.T_hinge*0.5))
                                          -np.sign(F_sp_h+F_f_g)[0]*self.sens_mechanical_grasper*self.mu_k_h*self.max_I3ant*self.P_I3_anterior*1))

                    else: # both are kinetic
                    # F_f_g = -np.sign(F_I2+F_sp_g-F_I3-F_Hi)*mechanical_in_grasper*mu_k_g*F_I4
                        A1 = (1/self.c_h*(self.K_sp_h*np.array([-1,0])-np.sign(F_sp_h+F_f_g)[0]
                                         *self.sens_mechanical_grasper*self.mu_k_h*self.max_I3ant*
                                         self.P_I3_anterior*np.array([1,-1])))
                        B1 = (1/self.c_h*(self.x_h_ref*self.K_sp_h-np.sign(F_I2+F_sp_g-F_I3-F_hinge)[0]*
                                          self.sens_mechanical_grasper*self.mu_k_g*F_I4
                                          -np.sign(F_sp_h+F_f_g)[0]*self.sens_mechanical_grasper*
                                          self.mu_k_h*self.max_I3ant*self.P_I3_anterior*1))               

                    A11 = A1[0]
                    A12 = A1[1]

            else: # if the seaweed is broken the jaws act as if unconstrained
                if(abs(F_sp_h+F_f_g) <= abs(self.mu_s_h*F_I3_ant)): # static friction is true
                    F_f_h = -self.sens_mechanical_grasper*(F_sp_h+F_f_g) # only calculate the force if an object is actually present
                    self.jaw_friction_state = 1
                else:
                    F_f_h = -np.sign(F_sp_h+F_f_g)[0]*self.sens_mechanical_grasper*self.mu_k_h*F_I3_ant # only calculate the force if an object is actually present
                    self.jaw_friction_state = 0


                A1 = 1/self.c_h*self.K_sp_h*[-1,0]
                B1 = 1/self.c_h*self.x_h_ref*self.K_sp_h

                A11 = A1[0]
                A12 = A1[1]
                self.jaw_friction_state = 0

        A = np.array([[A11,A12],[A21,A22]])
        B = np.array([[B1],[B2]])

        x_last = np.array(x_vec)

        x_new = (1/(1-self.TimeStep_h*A.trace()))*(np.dot((np.identity(2)+self.TimeStep_h*
                                                         np.array([[-A22,A12],[A21,-A11]])),x_last)+
                                                         self.TimeStep_h*B)
        
        self.x_g = x_new[1,0]
        self.x_h = x_new[0,0]

        ## calculate force on object
        self.force_on_object = F_f_g+F_f_h

        # check if seaweed is broken
        if (self.fixation_type ==1):
            if (self.force_on_object>self.seaweed_strength):
                print('seaweed broke')
                unbroken = 0

            # check to see if a new cycle has started
            x_gh_next = self.x_g-self.x_h

            if (not unbroken and x_gh <0.3 and x_gh_next>x_gh):
                print('start a new cycle')
                unbroken = 1 

            self.force_on_object= unbroken*self.force_on_object
        
        # Calculate Reward
        new_gm = np.array([self.x_g - self.x_h]) # grasper motion = self.x_g - self.x_h
        delta_gm = new_gm - old_gm # current - previous grasper motion
        self.delta_gm = delta_gm
        if self.grasper_friction_state: # static friction

            if delta_gm < 0:
                reward = -delta_gm * (self.force_on_object ** self.foo)
            else:
                reward = -delta_gm

            if not edible:
                reward *= -1
        else: # kinetic friction
            # penalizes the agent for signiﬁcant compressive load that tends to push edible seaweed out of the feeding apparatus.
            if self.force_on_object < 0 and delta_gm > 0:
              reward = self.force_on_object / 100
            else:
              reward = [0.0]                       
        return reward[0]


    def GeneratePlots(self,label):
        # === Plot Generator ==========================================================================================================================================================================
        import math, copy

        self.EndTime = (self.current_step - 1) * self.TimeStep
        t=np.atleast_2d(np.arange(self.StartingTime,self.EndTime+self.TimeStep,self.TimeStep))
        self.EndTime = self.max_steps*0.05
        end_ind = t.shape[1]
        
        tmp =  [1] * 15
        tmp.extend([2,2])
        axs = plt.figure(figsize=(10,15), constrained_layout=True).subplots(17,1, sharex=True, gridspec_kw={'height_ratios': tmp})
        lineW =2
        i= 0

        #External Stimuli
        ax0 = axs[0]
        ax0.plot(t.transpose(),self.sens_mechanical_grasper_history[0,:end_ind].transpose(), color=[56/255, 232/255, 123/255],linewidth=2) #========== mechanical in grasper

        ax0.set_ylabel('Mech. in Grasper')

        i=1
        ax = axs[i]
        ax.plot(t.transpose(),self.sens_chemical_lips_history[0,:end_ind].transpose(), color=[70/255, 84/255, 218/255],linewidth=2) #================= chemical at lips
        ax.set_ylabel('Chem. at Lips')
        i=i+1

        ax = axs[i]
        ax.plot(t.transpose(),self.sens_mechanical_lips_history[0,:end_ind].transpose(), color=[47/255, 195/255, 241/255],linewidth=2) #============== mechanical at lips
        ax.set_ylabel('Mech. at Lips')
        i=i+1
        
        ax = axs[i]
        ax.plot(t.transpose(),self.CBI2_history[0,:end_ind].transpose(),'k',linewidth=lineW) #======================================================== CBI2
        ax.set_ylabel('CBI-2')
        i=i+1

        ax = axs[i]
        ax.plot(t.transpose(),self.CBI3_history[0,:end_ind].transpose(),'k',linewidth=lineW) #======================================================== CBI3
        ax.set_ylabel('CBI-3')
        i=i+1
       
        ax = axs[i]
        ax.plot(t.transpose(),self.CBI4_history[0,:end_ind].transpose(),'k',linewidth=lineW) #======================================================== CBI4
        ax.set_ylabel('CBI-4')
        i=i+1
        
        #Interneurons
        ax = axs[i]
        ax.plot(t.transpose(),self.B64_history[0,:end_ind].transpose(),linewidth=lineW, color=[90/255, 131/255, 198/255]) #=========================== B64
        ax.set_ylabel('B64', color=[90/255, 131/255, 198/255])
        i=i+1

        ax = axs[i]
        ax.plot(t.transpose(),self.B20_history[0,:end_ind].transpose(),linewidth=lineW, color=[44/255, 166/255, 90/255]) #============================ B20
        i=i+1
        ax.set_ylabel('B20', color=[44/255, 166/255, 90/255])

        ax = axs[i]
        ax.plot(t.transpose(),self.B40B30_history[0,:end_ind].transpose(),linewidth=lineW, color=[192/255, 92/255, 185/255]) #======================== B40/B30
        i=i+1
        ax.set_ylabel('B40/B30', color=[192/255, 92/255, 185/255])
        
        ax = axs[i]
        ax.plot(t.transpose(),self.B4B5_history[0,:end_ind].transpose(),linewidth=lineW, color=[51/255, 185/255, 135/255]) #========================== B4/5
        i=i+1
        ax.set_ylabel('B4/B5', color=[51/255, 185/255, 135/255])
        
        #motor neurons
        ax = axs[i]
        ax.plot(t.transpose(),self.B31B32_history[0,:end_ind].transpose(),linewidth=lineW, color=[220/255, 81/255, 81/255]) #========================= I2 input
        i=i+1
        ax.set_ylabel('B31/B32',color=[220/255, 81/255, 81/255])
        
        ax = axs[i]
        ax.plot(t.transpose(),self.B8_history[0,:end_ind].transpose(),linewidth=lineW, color=[213/255, 155/255, 196/255]) #=========================== B8a/b
        i=i+1
        ax.set_ylabel('B8a/b', color=[213/255, 155/255, 196/255])

        ax = axs[i]
        ax.plot(t.transpose(),self.B38_history[0,:end_ind].transpose(),linewidth=lineW, color=[238/255, 191/255, 70/255]) #=========================== B38
        i=i+1
        ax.set_ylabel('B38', color=[238/255, 191/255, 70/255])
        
        ax = axs[i]
        ax.plot(t.transpose(),self.B6B9B3_history[0,:end_ind].transpose(),linewidth=lineW, color=[90/255, 155/255, 197/255]) #======================== B6/9/3
        i=i+1
        ax.set_ylabel('B6/B9/B3', color=[90/255, 155/255, 197/255])
        
        ax = axs[i]
        ax.plot(t.transpose(),self.B7_history[0,:end_ind].transpose(),linewidth=lineW, color=[56/255, 167/255, 182/255]) #============================= B7
        i=i+1
        ax.set_ylabel('B7', color=[56/255, 167/255, 182/255])
        
        #Grasper Motion plot
        grasper_motion = self.x_g_history - self.x_h_history
        
        ax = axs[i]
        ax.plot(t.transpose(),grasper_motion[0,:end_ind].transpose(),'b',linewidth=lineW)

        # overlay the grasper friction state as thick blue dots
        grasper_motion_gfs = copy.deepcopy(grasper_motion) # long
        t_gfs = copy.deepcopy(t) # short
        grasper_motion_gfs[self.grasper_friction_state_history != 1] = math.nan # long
        t_gfs[0, self.grasper_friction_state_history[0, :end_ind] != 1] = math.nan # short
        ax.plot(t_gfs.transpose(), grasper_motion_gfs[0,:end_ind].transpose(),'b', linewidth = lineW * 2)
        
        # overlay b&w bars
        gm_delta = np.zeros_like(grasper_motion) # long
        t_delta = copy.deepcopy(t) # short
        gm_delta[:,1:] = grasper_motion[:,1:] - grasper_motion[:,:-1]
        
        t_delta[0,gm_delta[0, :end_ind] <= 0] = math.nan

        gm_delta[gm_delta > 0] = 1.25
        gm_delta[gm_delta != 1.25 ] = math.nan

        ax.plot(t.transpose(), 1.25 * np.ones_like(t)[0,:end_ind].transpose(), 'k', linewidth = lineW * 3)
        ax.plot(t_delta.transpose(), gm_delta[0,:end_ind].transpose(),'w', linewidth = lineW * 2.8)

        i=i+1
        ax.set_ylabel('Grasper Motion', color=[0/255, 0/255, 255/255])
       
        #subplot(15,1,15)
        ax = axs[i]
        ax.plot(t.transpose(),self.force_history[0, :end_ind].transpose(),'k',linewidth=lineW)
        ax.set_ylabel('Force', color=[0/255, 0/255, 0/255])    
        # i=i+1

        for i, a in enumerate(axs):
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            if i < 15:
                a.set_ylim([0, 1])
            elif i == 15:
                a.set_ylim([0, 1.5])
            else:
                a.set_ylim([-0.5, 1])
        plt.xticks([])
        plt.yticks([])
        plt.setp(axs, xlim=(np.min(t),np.max(t)))
        plt.show()

    '''
    def GeneratePlotsFiles_CJF(self,label,T):
        # === Plot and File Generator ==========================================================================================================================================================================
        import math, copy

        self.EndTime = (self.current_step - 1) * self.TimeStep
        t=np.atleast_2d(np.arange(self.StartingTime,self.EndTime+self.TimeStep,self.TimeStep))
        self.EndTime = self.max_steps*0.05
        end_ind = t.shape[1]
        
        tmp =  [1] * 15
        tmp.extend([2,2])
        axs = plt.figure(figsize=(10,15), constrained_layout=True).subplots(17,1, sharex=True, gridspec_kw={'height_ratios': tmp})
        lineW =2
        i= 0

        #External Stimuli
        ax0 = axs[0]
        ax0.plot(t.transpose(),self.sens_mechanical_grasper_history[0,:end_ind].transpose(), color=[56/255, 232/255, 123/255],linewidth=2) #========== mechanical in grasper
        arr = np.concatenate([t.transpose(),self.sens_mechanical_grasper_history[0,:end_ind].transpose()], axis = 1)
        np.save('MechanicalInGrasper'+T+'.npy', arr)
        np.savetxt('MechanicalInGrasper'+T+'.csv', arr)
        print(1)
        ax0.set_ylabel('Mech. in Grasper')

        i=1
        ax = axs[i]
        ax.plot(t.transpose(),self.sens_chemical_lips_history[0,:end_ind].transpose(), color=[70/255, 84/255, 218/255],linewidth=2) #================= chemical at lips
        arr = np.concatenate([t.transpose(),self.sens_chemical_lips_history[0,:end_ind].transpose()], axis = 1)
        np.save('ChemicalAtLips'+T+'.npy', arr)
        np.savetxt('ChemicalAtLips'+T+'.csv', arr)
        print(2)
        ax.set_ylabel('Chem. at Lips')
        i=i+1

        ax = axs[i]
        ax.plot(t.transpose(),self.sens_mechanical_lips_history[0,:end_ind].transpose(), color=[47/255, 195/255, 241/255],linewidth=2) #============== mechanical at lips
        arr = np.concatenate([t.transpose(),self.sens_mechanical_lips_history[0,:end_ind].transpose()], axis = 1)
        np.save('MechanicalAtLips'+T+'.npy', arr)
        np.savetxt('MechanicalAtLips'+T+'.csv', arr)
        print(3)
        ax.set_ylabel('Mech. at Lips')
        i=i+1
        
        ax = axs[i]
        ax.plot(t.transpose(),self.CBI2_history[0,:end_ind].transpose(),'k',linewidth=lineW) #======================================================== CBI2
        arr = np.concatenate([t.transpose(),self.CBI2_history[0,:end_ind].transpose()], axis = 1)
        np.save('CBI2'+T+'.npy', arr)
        np.savetxt('CBI2'+T+'.csv', arr)
        print(4)
        ax.set_ylabel('CBI-2')
        i=i+1

        ax = axs[i]
        ax.plot(t.transpose(),self.CBI3_history[0,:end_ind].transpose(),'k',linewidth=lineW) #======================================================== CBI3
        arr = np.concatenate([t.transpose(),self.CBI3_history[0,:end_ind].transpose()], axis = 1)
        np.save('CBI3'+T+'.npy', arr)
        np.savetxt('CBI3'+T+'.csv', arr)
        print(5)
        ax.set_ylabel('CBI-3')
        i=i+1
       
        ax = axs[i]
        ax.plot(t.transpose(),self.CBI4_history[0,:end_ind].transpose(),'k',linewidth=lineW) #======================================================== CBI4
        arr = np.concatenate([t.transpose(),self.CBI4_history[0,:end_ind].transpose()], axis = 1)
        np.save('CBI4'+T+'.npy', arr)
        np.savetxt('CBI4'+T+'.csv', arr)
        print(6)
        ax.set_ylabel('CBI-4')
        i=i+1
        
        #Interneurons
        ax = axs[i]
        ax.plot(t.transpose(),self.B64_history[0,:end_ind].transpose(),linewidth=lineW, color=[90/255, 131/255, 198/255]) #=========================== B64
        arr = np.concatenate([t.transpose(),self.B64_history[0,:end_ind].transpose()], axis = 1)
        np.save('B64'+T+'.npy', arr)
        np.savetxt('B64'+T+'.csv', arr)
        print(7)
        ax.set_ylabel('B64', color=[90/255, 131/255, 198/255])
        i=i+1

        ax = axs[i]
        ax.plot(t.transpose(),self.B20_history[0,:end_ind].transpose(),linewidth=lineW, color=[44/255, 166/255, 90/255]) #============================ B20
        arr = np.concatenate([t.transpose(),self.B20_history[0,:end_ind].transpose()], axis = 1)
        np.save('B20'+T+'.npy', arr)
        np.savetxt('B20'+T+'.csv', arr)
        print(8)
        i=i+1
        ax.set_ylabel('B20', color=[44/255, 166/255, 90/255])

        ax = axs[i]
        ax.plot(t.transpose(),self.B40B30_history[0,:end_ind].transpose(),linewidth=lineW, color=[192/255, 92/255, 185/255]) #======================== B40/B30
        arr = np.concatenate([t.transpose(),self.B40B30_history[0,:end_ind].transpose()], axis = 1)
        np.save('B40_B30'+T+'.npy', arr)
        np.savetxt('B40_B30'+T+'.csv', arr)
        print(9)
        i=i+1
        ax.set_ylabel('B40/B30', color=[192/255, 92/255, 185/255])
        
        ax = axs[i]
        ax.plot(t.transpose(),self.B4B5_history[0,:end_ind].transpose(),linewidth=lineW, color=[51/255, 185/255, 135/255]) #========================== B4/5
        arr = np.concatenate([t.transpose(),self.B4B5_history[0,:end_ind].transpose()], axis = 1)
        np.save('B4_5'+T+'.npy', arr)
        np.savetxt('B4_5'+T+'.csv', arr)
        print(10)
        i=i+1
        ax.set_ylabel('B4/B5', color=[51/255, 185/255, 135/255])
        
        #motor neurons
        ax = axs[i]
        ax.plot(t.transpose(),self.B31B32_history[0,:end_ind].transpose(),linewidth=lineW, color=[220/255, 81/255, 81/255]) #========================= I2 input
        arr = np.concatenate([t.transpose(),self.B31B32_history[0,:end_ind].transpose()], axis = 1)
        np.save('I2_input'+T+'.npy', arr)
        np.savetxt('I2_input'+T+'.csv', arr)
        print(11)
        i=i+1
        ax.set_ylabel('B31/B32',color=[220/255, 81/255, 81/255])
        
        ax = axs[i]
        ax.plot(t.transpose(),self.B8_history[0,:end_ind].transpose(),linewidth=lineW, color=[213/255, 155/255, 196/255]) #=========================== B8a/b
        arr = np.concatenate([t.transpose(),self.B8_history[0,:end_ind].transpose()], axis = 1)
        np.save('B8a_b'+T+'.npy', arr)
        np.savetxt('B8a_b'+T+'.csv', arr)
        print(12)
        i=i+1
        ax.set_ylabel('B8a/b', color=[213/255, 155/255, 196/255])

        ax = axs[i]
        ax.plot(t.transpose(),self.B38_history[0,:end_ind].transpose(),linewidth=lineW, color=[238/255, 191/255, 70/255]) #=========================== B38
        arr = np.concatenate([t.transpose(),self.B38_history[0,:end_ind].transpose()], axis = 1)
        np.save('B38'+T+'.npy', arr)
        np.savetxt('B38'+T+'.csv', arr)
        print(13)
        i=i+1
        ax.set_ylabel('B38', color=[238/255, 191/255, 70/255])
        
        ax = axs[i]
        ax.plot(t.transpose(),self.B6B9B3_history[0,:end_ind].transpose(),linewidth=lineW, color=[90/255, 155/255, 197/255]) #======================== B6/9/3
        arr = np.concatenate([t.transpose(),self.B6B9B3_history[0,:end_ind].transpose()], axis = 1)
        np.save('B6_9_3'+T+'.npy', arr)
        np.savetxt('B6_9_3'+T+'.csv', arr)
        print(14)
        i=i+1
        ax.set_ylabel('B6/B9/B3', color=[90/255, 155/255, 197/255])
        
        ax = axs[i]
        ax.plot(t.transpose(),self.B7_history[0,:end_ind].transpose(),linewidth=lineW, color=[56/255, 167/255, 182/255]) #============================= B7
        arr = np.concatenate([t.transpose(),self.B7_history[0,:end_ind].transpose()], axis = 1)
        np.save('B7'+T+'.npy', arr)
        np.savetxt('B7'+T+'.csv', arr)
        print(15)
        i=i+1
        ax.set_ylabel('B7', color=[56/255, 167/255, 182/255])
        
        #Grasper Motion plot
        grasper_motion = self.x_g_history - self.x_h_history
        
        ax = axs[i]
        ax.plot(t.transpose(),grasper_motion[0,:end_ind].transpose(),'b',linewidth=lineW)

        # overlay the grasper friction state as thick blue dots
        grasper_motion_gfs = copy.deepcopy(grasper_motion) # long
        t_gfs = copy.deepcopy(t) # short
        grasper_motion_gfs[self.grasper_friction_state_history != 1] = math.nan # long
        t_gfs[0, self.grasper_friction_state_history[0, :end_ind] != 1] = math.nan # short
        ax.plot(t_gfs.transpose(), grasper_motion_gfs[0,:end_ind].transpose(),'b', linewidth = lineW * 2)
        
        # overlay b&w bars
        gm_delta = np.zeros_like(grasper_motion) # long
        t_delta = copy.deepcopy(t) # short
        gm_delta[:,1:] = grasper_motion[:,1:] - grasper_motion[:,:-1]
        
        t_delta[0,gm_delta[0, :end_ind] <= 0] = math.nan

        gm_delta[gm_delta > 0] = 1.25
        gm_delta[gm_delta != 1.25 ] = math.nan

        ax.plot(t.transpose(), 1.25 * np.ones_like(t)[0,:end_ind].transpose(), 'k', linewidth = lineW * 3)
        ax.plot(t_delta.transpose(), gm_delta[0,:end_ind].transpose(),'w', linewidth = lineW * 2.8)

        i=i+1
        ax.set_ylabel('Grasper Motion', color=[0/255, 0/255, 255/255])
       
        #subplot(15,1,15)
        ax = axs[i]
        ax.plot(t.transpose(),self.force_history[0, :end_ind].transpose(),'k',linewidth=lineW)
        ax.set_ylabel('Force', color=[0/255, 0/255, 0/255])    
        # i=i+1

        for i, a in enumerate(axs):
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            if i < 15:
                a.set_ylim([0, 1])
            elif i == 15:
                a.set_ylim([0, 1.5])
        plt.xticks([])
        plt.yticks([])
        plt.setp(axs, xlim=(np.min(t),np.max(t)))
        plt.show(block=False)
    '''

    def GeneratePlots_training(self,label):
        # === Training Plot Generator ==========================================================================================================================================================================
        self.EndTime = (self.current_step - 1) * self.TimeStep
        t=np.atleast_2d(np.arange(self.StartingTime,self.EndTime+self.TimeStep,self.TimeStep))
        self.EndTime = self.max_steps*0.05
        end_ind = t.shape[1]

        axs = plt.figure(figsize=(20,20), constrained_layout=True).subplots(6, 2)
        lineW =2
        i=0

        #motor neurons
        ax = axs[int(i/2),i%2]
        ax.plot(t.transpose(),self.B31B32_history[0,:end_ind].transpose(),linewidth=lineW, color=[220/255, 81/255, 81/255]) # I2 input
        i=i+1
        ax.set_ylabel('B31/B32_activity',color=[220/255, 81/255, 81/255])
        
        ax = axs[int(i/2),i%2]
        ax.plot(t.transpose(),self.B8_history[0,:end_ind].transpose(),linewidth=lineW, color=[213/255, 155/255, 196/255]) # B8a/b
        i=i+1
        ax.set_ylabel('B8a/b_activity', color=[213/255, 155/255, 196/255])

        ax = axs[int(i/2),i%2]
        ax.plot(t.transpose(),self.B38_history[0,:end_ind].transpose(),linewidth=lineW, color=[238/255, 191/255, 70/255]) # B38
        i=i+1
        ax.set_ylabel('B38_activity', color=[238/255, 191/255, 70/255])

        ax = axs[int(i/2),i%2]
        ax.plot(t.transpose(),self.B6B9B3_history[0,:end_ind].transpose(),linewidth=lineW, color=[90/255, 155/255, 197/255]) # B6/9/3
        i=i+1
        ax.set_ylabel('B6/B9/B3_activity', color=[90/255, 155/255, 197/255])
        
        ax = axs[int(i/2),i%2]
        ax.plot(t.transpose(),self.B7_history[0,:end_ind].transpose(),linewidth=lineW, color=[56/255, 167/255, 182/255]) # B7
        i=i+1
        ax.set_ylabel('B7_activity', color=[56/255, 167/255, 182/255])
        
        #muscles
        ax = axs[int(i/2),i%2]
        ax.plot(t.transpose(),self.T_I2_history[0,:end_ind].transpose(),linewidth=lineW, color=[220/255, 81/255, 81/255]) # I2 input
        i=i+1
        ax.set_ylabel('I2:protraction.B31/B32_muscle',color=[220/255, 81/255, 81/255])
        
        ax = axs[int(i/2),i%2]
        ax.plot(t.transpose(),self.P_I4_history[0,:end_ind].transpose(),linewidth=lineW, color=[213/255, 155/255, 196/255]) # B8a/b
        i=i+1
        ax.set_ylabel('I4:grasperclose.B8a/b_muscle', color=[213/255, 155/255, 196/255])

        ax = axs[int(i/2),i%2]
        ax.plot(t.transpose(),self.P_I3_anterior_history[0,:end_ind].transpose(),linewidth=lineW, color=[238/255, 191/255, 70/255]) # B38
        i=i+1
        ax.set_ylabel('I3:pinch.B38_muscle', color=[238/255, 191/255, 70/255])
        
        ax = axs[int(i/2),i%2]
        ax.plot(t.transpose(),self.T_I3_history[0,:end_ind].transpose(),linewidth=lineW, color=[90/255, 155/255, 197/255]) # B6/9/3
        i=i+1
        ax.set_ylabel('I3:retraction.B6/B9/B3_muscle', color=[90/255, 155/255, 197/255])
        
        ax = axs[int(i/2),i%2]
        ax.plot(t.transpose(),self.T_hinge_history[0,:end_ind].transpose(),linewidth=lineW, color=[56/255, 167/255, 182/255]) # B7
        i=i+1
        ax.set_ylabel('hinge:retraction.B7_muscle', color=[56/255, 167/255, 182/255])
        
        #Grasper Motion
        grasper_motion = self.x_g_history - self.x_h_history
        ax = axs[int(i/2),i%2]
        ax.plot(t.transpose(),grasper_motion[0,:end_ind].transpose(),'b',linewidth=lineW)
        ax.set_ylabel('Grasper Motion', color=[0/255, 0/255, 255/255])
       
        #subplot(15,1,15)
        ax = axs[int(i/2),i%2]
        ax.plot(t.transpose(),self.force_history[0,:end_ind].transpose(),'k',linewidth=lineW)
        ax.set_ylabel('Force', color=[0/255, 0/255, 0/255])

        ax = axs[int(i/2),i%2]
        ax.plot(t.transpose(),self.grasper_friction_state_history[0,:end_ind].transpose(),linewidth=lineW)
        ax.set_ylabel('friction state')
        
        plt.show(block=False)


    def GeneratePlotsFiles_training_CJF(self,label,B,T):
        
        # === Training Plot Generator ==========================================================================================================================================================================
        self.EndTime = (self.current_step - 1) * self.TimeStep
        t=np.atleast_2d(np.arange(self.StartingTime,self.EndTime+self.TimeStep,self.TimeStep))
        self.EndTime = self.max_steps*0.05
        end_ind = t.shape[1]
        import os
        os.makedirs(B, exist_ok=True)
        
        # motor neurons
        # ================== I2 input
        arr = np.concatenate([t.transpose(),self.B31B32_history[0,:end_ind].reshape(-1,1)], axis = 1)
        
        np.save(B+'I2_input'+T+'.npy', arr)
        np.savetxt(B+'I2_input'+T+'.csv', arr)
        
        # ================== B8a/b
        arr = np.concatenate([t.transpose(),self.B8_history[0,:end_ind].reshape(-1,1)], axis = 1)
        np.save(B+'B8a_b'+T+'.npy', arr)
        np.savetxt(B+'B8a_b'+T+'.csv', arr)

        # ================== B38
        arr = np.concatenate([t.transpose(),self.B38_history[0,:end_ind].reshape(-1,1)], axis = 1)
        np.save(B+'B38'+T+'.npy', arr)
        np.savetxt(B+'B38'+T+'.csv', arr)

        # ================== B6/9/3
        arr = np.concatenate([t.transpose(),self.B6B9B3_history[0,:end_ind].reshape(-1,1)], axis = 1)
        np.save(B+'B6_9_3'+T+'.npy', arr)
        np.savetxt(B+'B6_9_3'+T+'.csv', arr)
        
        # ================== B7
        arr = np.concatenate([t.transpose(),self.B7_history[0,:end_ind].reshape(-1,1)], axis = 1)
        np.save(B+'B7'+T+'.npy', arr)
        np.savetxt(B+'B7'+T+'.csv', arr)
        '''
        # muscles
        # ================== I2 input --- T_I2
        arr = np.concatenate([t.transpose(),self.T_I2_history[0,:end_ind].reshape(-1,1)], axis = 1)
        np.save(B+'T_I2'+T+'.npy', arr)
        np.savetxt(B+'T_I2'+T+'.csv', arr)
        
        # ================== B8a/b --- P_I4
        arr = np.concatenate([t.transpose(),self.P_I4_history[0,:end_ind].reshape(-1,1)], axis = 1)
        np.save(B+'P_I4'+T+'.npy', arr)
        np.savetxt(B+'P_I4'+T+'.csv', arr)

        # ================== B38 --- P_I3_anterior
        arr = np.concatenate([t.transpose(),self.P_I3_anterior_history[0,:end_ind].reshape(-1,1)], axis = 1)
        np.save(B+'P_I3_anterior'+T+'.npy', arr)
        np.savetxt(B+'P_I3_anterior'+T+'.csv', arr)
        
        # ================== B6/9/3 --- T_I3
        arr = np.concatenate([t.transpose(),self.T_I3_history[0,:end_ind].reshape(-1,1)], axis = 1)
        np.save(B+'T_I3'+T+'.npy', arr)
        np.savetxt(B+'T_I3'+T+'.csv', arr)
        
        # ================== B7 --- T_hinge
        arr = np.concatenate([t.transpose(),self.T_hinge_history[0,:end_ind].reshape(-1,1)], axis = 1)
        np.save(B+'T_hinge'+T+'.npy', arr)
        np.savetxt(B+'T_hinge'+T+'.csv', arr)
        '''

        # Grasper Motion
        grasper_motion = self.x_g_history - self.x_h_history
        # ================== Grasper Motion
        arr = np.concatenate([t.transpose(),grasper_motion[0,:end_ind].reshape(-1,1)], axis = 1)
        np.save(B+'RelativeGrasperMotion'+T+'.npy', arr)
        np.savetxt(B+'RelativeGrasperMotion'+T+'.csv', arr)

        
        #subplot(15,1,15)
        # ================== Force
        arr = np.concatenate([t.transpose(),self.force_history[0,:end_ind].reshape(-1,1)], axis = 1)
        np.save(B+'Force'+T+'.npy', arr)
        np.savetxt(B+'Force'+T+'.csv', arr)
        
    
    
    def render(self, mode='human'):
        # === Render ==========================================================================================================================================================================
        pass


    def close(self):
        # === Close ==========================================================================================================================================================================
        pass


    def RotationMatrixZ(self,theta):
        # === Rotate Matrix ==========================================================================================================================================================================
        return np.matrix([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


    def unitVec(self,origin,endpoint):
        # === Create Unit Vector ==========================================================================================================================================================================
        vec = origin-endpoint
        if (vec.item(0)==0 and vec.item(1)==0):
            return np.array([[0],[0]])
        else:
            return (origin - endpoint)/np.linalg.norm(origin - endpoint)


    def Animate(self,j):
        # === Animate Feeding Mechanism ==========================================================================================================================================================================
        if j % 100 == 0:
            print(j)
        lines =[] # ok
        
        M_grasper_rot = self.RotationMatrixZ(self.theta_g[0,j]) # ok
        self.x_R = np.array([[self.x_g_history[0,j]],[0]]) # ok
        self.x_H = np.array([[self.x_h_history[0,j]],[0]]) # ok

        #define vectors to various points on grasper
        x_G_def = M_grasper_rot*np.array([[self.grasper_offset*math.cos(self.theta_grasper_initial)],
                                          [self.grasper_offset*math.sin(self.theta_grasper_initial)]]) + self.x_R # ok
        x_S_def = x_G_def + M_grasper_rot*np.array([[self.r_grasper*math.cos(self.theta_s)],
                                                    [self.r_grasper*math.sin(self.theta_s)]])  # ok
        x_I2_A_def = x_G_def + M_grasper_rot*np.array([[-self.r_grasper*math.cos(self.theta_I2_A)],
                                                       [self.r_grasper*math.sin(self.theta_I2_A)]]) # ok
        x_I2_B_def = x_G_def + M_grasper_rot*np.array([[-self.r_grasper*math.cos(self.theta_I2_B)],
                                                       [-self.r_grasper*math.sin(self.theta_I2_B)]]) # ok

        #rotate all the vectors
        #x_G = M_grasper_rot*x_G_def
        #x_S = M_grasper_rot*x_S_def
        #x_I2_A = M_grasper_rot*x_I2_A_def
        #x_I2_B = M_grasper_rot*x_I2_B_def
        x_G = x_G_def
        x_S = x_S_def
        x_I2_A = x_I2_A_def
        x_I2_B = x_I2_B_def

        #Calculate vectors and lengths for forces/tensions

        #I2
        vec_I2_A = self.unitVec(self.x_I2_Aorigin,x_I2_A) # ok
        vec_I2_B = self.unitVec(self.x_I2_Borigin,x_I2_B) # ok
        length_I2 = np.linalg.norm(vec_I2_A)+np.linalg.norm(vec_I2_B)+np.linalg.norm(x_I2_A-x_I2_B) # ok

        #I3
        vec_I3 = np.array([[-1],[0]])
        length_I3 = (x_G[0,0] -(-self.r_grasper))

        #hinge
        vec_hinge = self.unitVec(self.x_hinge_origin,self.x_R)
        length_hinge = np.linalg.norm(self.x_R - self.x_hinge_origin)

        #seaweed or tube
        vec_object = np.array([[9],[0]]) # prev: 1,0
        
        #plot slug head

        #plot line representing track
        self.line_grapser_track.set_data([self.grasper_origin[0],self.grasper_full[0]],[self.grasper_origin[1],self.grasper_full[1]])

        #plot line from R to G
        self.line_RG.set_data([self.x_R[0],x_G[0]],[self.x_R[1],x_G[1]])
        #plot dot at point R
        # self.dot_R.set_data(self.x_R[0],self.x_R[1])
        #plot dot at point G
        # self.dot_G.set_data(x_G[0],x_G[1])
        #plot dot at point S
        # self.dot_S.set_data(x_S[0],x_S[1])
        #plot dot at I2 attachment point A
        self.dot_I2_A.set_data(x_I2_A[0],x_I2_A[1])
        #plot dot at I2 attachment point B
        self.dot_I2_B.set_data(x_I2_B[0],x_I2_B[1])
        #plot dot at I2 attachment point A
        self.dot_I2_Aorigin.set_data(self.x_I2_Aorigin[0],self.x_I2_Aorigin[1])
        #plot dot at I2 attachment point B
        self.dot_I2_Borigin.set_data(self.x_I2_Borigin[0],self.x_I2_Borigin[1])

        #draw grasper
        self.draw_circle.center = x_G[0],x_G[1]
        self.wedge.set_center((x_G[0][0,0]  , x_G[1][0,0]))
        self.wedge.set_theta1(30*(1-self.P_I4_history[0,j]))
        self.wedge.set_theta2(360 - 30*(1-self.P_I4_history[0,j]) )

        #draw head
        self.head.set_xy((self.x_H[0],self.x_H[1]-self.x_H_height/2))
        
        #plot a dot at the head spring attachment
        self.dot_H_spring.set_data(self.x_H[0],self.x_H[1])

        #draw head spring as dashed line
        self.line_H_spring.set_data([self.x_H[0],self.x_ground[0]],[self.x_H[1],self.x_ground[1]])

        #draw grasper to head spring as dashed line
        self.line_G_spring.set_data([self.grasper_origin[0],self.x_H[0]],[self.grasper_origin[1],self.x_H[1]])

        lines.append(self.line_ground)
        lines.append(self.dot_ground)
        lines.append(self.line_transducer)
        lines.append(self.line_grapser_track)
        lines.append(self.line_RG)
        # lines.append(self.dot_R)
        # lines.append(self.dot_G)
        # lines.append(self.dot_S)
        lines.append(self.dot_I2_A)
        lines.append(self.dot_I2_B)
        lines.append(self.dot_I2_Aorigin)
        lines.append(self.dot_I2_Borigin)
        lines.append(self.draw_circle)
        lines.append(self.food)
        lines.append(self.wedge)

        self.ab.remove()
        sh = mpimg.imread('slug.png')
        imagebox = OffsetImage(sh, zoom=0.15)
        self.ab = AnnotationBbox(imagebox, (self.x_H[0]+0.5, 0.5), frameon=False)
        axes.add_artist(self.ab)

        lines.append(self.ab) 
        
        lines.append(self.head)
        lines.append(self.dot_H_spring)
        lines.append(self.line_H_spring)
        lines.append(self.line_G_spring)
        if j > 0:
            self.line2.set_data(self.time_axis[:j], self.total_reward_log[:j])
        else:
            self.line2.set_data(np.array([0]), np.array([0]))
        
        return lines, self.line2

    
    def VisualizeMotion(self):
        # === Visualize Motion ==========================================================================================================================================================================
        if (self.biomechanicsModel==1):
            self.x_H = self.x_ground + np.array([[0],[0]])
            self.grasper_origin = self.x_H + np.array([[0],[0]])
            self.grasper_offset=0            
            #show the current pose of the model
            global axes, ax2
            figure, (axes, ax2) = plt.subplots(1, 2)
            self.time_axis = np.arange(len(self.output_expert_mean)) * 0.05
            
            self.line2, = ax2.plot([], [], 'yo', label = 'Agent')

            ax2.plot(self.time_axis, self.output_expert_mean, label = 'Expert')
            ax2.fill_between(self.time_axis, self.output_expert_mean - self.output_expert_std, self.output_expert_mean + self.output_expert_std, alpha = 0.2)

            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Cumulative Reward')
            ax2.legend()

            figure.set_size_inches(9, 3, True)
            resultShape = np.shape(self.x_h_history) # ok
            print('*resultShape =', resultShape)
            
            M_grasper_rot = self.RotationMatrixZ(self.theta_g[0,0])
            self.x_R = np.array([[self.x_g_history[0,0]],[0]])
            self.x_H = np.array([[self.x_h_history[0,0]],[0]])

            #define vectors to various points on grasper
            x_G_def = M_grasper_rot*np.array([[self.grasper_offset*math.cos(self.theta_grasper_initial)],
                                              [self.grasper_offset*math.sin(self.theta_grasper_initial)]]) + self.x_R
            x_S_def = x_G_def + M_grasper_rot*np.array([[self.r_grasper*math.cos(self.theta_s)],
                                                        [self.r_grasper*math.sin(self.theta_s)]])
            x_I2_A_def = x_G_def + M_grasper_rot*np.array([[-self.r_grasper*math.cos(self.theta_I2_A)],
                                                           [self.r_grasper*math.sin(self.theta_I2_A)]])
            x_I2_B_def = x_G_def + M_grasper_rot*np.array([[-self.r_grasper*math.cos(self.theta_I2_B)],
                                                           [-self.r_grasper*math.sin(self.theta_I2_B)]])

            #rotate all the vectors
            #x_G = M_grasper_rot*x_G_def
            #x_S = M_grasper_rot*x_S_def
            #x_I2_A = M_grasper_rot*x_I2_A_def
            #x_I2_B = M_grasper_rot*x_I2_B_def
            x_G = x_G_def
            x_S = x_S_def
            x_I2_A = x_I2_A_def
            x_I2_B = x_I2_B_def

            #Calculate vectors and lengths for forces/tensions

            #I2
            vec_I2_A = self.unitVec(self.x_I2_Aorigin,x_I2_A)
            vec_I2_B = self.unitVec(self.x_I2_Borigin,x_I2_B)
            length_I2 = np.linalg.norm(vec_I2_A)+np.linalg.norm(vec_I2_B)+np.linalg.norm(x_I2_A-x_I2_B)

            #I3
            vec_I3 = np.array([[-1],[0]])
            length_I3 = (x_G[0,0] -(-self.r_grasper))

            #hinge
            vec_hinge = self.unitVec(self.x_hinge_origin,self.x_R)
            length_hinge = np.linalg.norm(self.x_R - self.x_hinge_origin)

            #seaweed or tube
            vec_object = np.array([[1],[0]])

            lines = []
            
            #plot line representing ground
            self.line_ground, = axes.plot([self.x_ground[0],self.x_ground[0]],[self.x_ground[1]+self.len_ground_line/2,
                                                           self.x_ground[1]-self.len_ground_line/2])
            lines.append(self.line_ground)
            #plot a dot at the origin
            self.dot_ground, = axes.plot(self.x_ground[0],self.x_ground[1],'ko', alpha = 0.05)
            lines.append(self.dot_ground)

            #plot line representing force transducer
            self.line_transducer, = axes.plot([self.x_transducer[0],self.x_transducer[0]],[self.x_transducer[1]+self.len_transducer_line/2,
                                                                   self.x_transducer[1]-self.len_transducer_line/2],'y')
            lines.append(self.line_transducer)

            #plot line representing track
            self.line_grapser_track, = axes.plot([self.grasper_origin[0],self.grasper_full[0]],[self.grasper_origin[1],self.grasper_full[1]],'y--')
            lines.append(self.line_grapser_track)
            #plot line from R to G
            self.line_RG, = axes.plot([self.x_R[0],x_G[0]],[self.x_R[1],x_G[1]])
            lines.append(self.line_RG)
            #plot dot at point R
            # self.dot_R, = axes.plot(self.x_R[0],self.x_R[1],'ko')
            # lines.append(self.dot_R)

            #plot dot at point G
            # self.dot_G, = axes.plot(x_G[0],x_G[1],'mo')
            # lines.append(self.dot_G)
            self.wedge = patches.Wedge((0,0), 0.5, 30, 300, ec="none")
            axes.add_patch(self.wedge)
            lines.append(self.wedge)

            #plot 'food'
            self.food, = axes.plot([0.5, 8],[0, 0],'g', linewidth=6, alpha = 0.75)
            lines.append(self.food)

            #plot dot at point S
            # self.dot_S, = axes.plot(x_S[0],x_S[1],'go')
            # lines.append(self.dot_S)
            #plot dot at I2 attachment point A
            self.dot_I2_A, = axes.plot(x_I2_A[0],x_I2_A[1],'ro', alpha = 0.05)
            lines.append(self.dot_I2_A)
            #plot dot at I2 attachment point B
            self.dot_I2_B, = axes.plot(x_I2_B[0],x_I2_B[1],'ro', alpha = 0.05)
            lines.append(self.dot_I2_B)
            #plot dot at I2 attachment point A
            self.dot_I2_Aorigin, = axes.plot(self.x_I2_Aorigin[0],self.x_I2_Aorigin[1],'ro', alpha = 0.05)
            lines.append(self.dot_I2_Aorigin)
            #plot dot at I2 attachment point B
            self.dot_I2_Borigin, = axes.plot(self.x_I2_Borigin[0],self.x_I2_Borigin[1],'ro', alpha = 0.05)
            lines.append(self.dot_I2_Borigin)
            #draw grasper
            self.draw_circle = plt.Circle(x_G, self.r_grasper,fill=False,color='blue', alpha = 0.05)
            axes.add_artist(self.draw_circle)
            axes.set_xlim([-2,6])
            axes.set_ylim([-3,3])
            axes.set_aspect(aspect = 1)
            lines.append(self.draw_circle)
            #draw head
            self.head = patches.Rectangle((self.x_H[0],self.x_H[1]-self.x_H_height/2),self.x_H_width,self.x_H_height, alpha = 0.05)
            self.head.set_fill(0)
            axes.add_patch(self.head)
            lines.append(self.head)
       
            sh = mpimg.imread('slug.png')
            imagebox = OffsetImage(sh, zoom=0.02)
            self.ab = AnnotationBbox(imagebox, (2, 0), frameon=False)
            global tmp
            tmp = axes.add_artist(self.ab)
            lines.append(self.ab) 

            # artists.append(ax.add_artist(ab))

            #plot a dot at the head spring attachment
            self.dot_H_spring, = axes.plot(self.x_H[0],self.x_H[1],'ko', alpha = 0.05)
            lines.append(self.dot_H_spring)

            #draw head spring as dashed line
            self.line_H_spring, = axes.plot([self.x_H[0],self.x_ground[0]],[self.x_H[1],self.x_ground[1]],'k--', alpha = 0.05)

            lines.append(self.line_H_spring)
            #draw grasper to head spring as dashed line
            self.line_G_spring, = axes.plot([self.grasper_origin[0],self.x_H[0]],[self.grasper_origin[1],self.x_H[1]],'b--', alpha = 0.05)
            lines.append(self.line_G_spring)
            dpi = 600
            ani = animation.FuncAnimation(figure, self.Animate, resultShape[1], interval=50, blit=False)
            # writer = animation.PillowWriter(fps=20)  
            # ani.save("dummy_blitoff.gif", writer=writer, dpi = dpi) 

            writer = animation.writers['ffmpeg'](fps=20)
            ani.save('animation.mp4',writer=writer,dpi=dpi)
            # plt.show()