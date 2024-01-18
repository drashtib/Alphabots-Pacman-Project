# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 01:52:37 2024

@author: Drashti Bhatt
"""
#importing liabraries
import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback,StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.dqn.policies import CnnPolicy
from gymnasium.utils.save_video import save_video
from gymnasium.wrappers import FrameStack,  ResizeObservation
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

#%%DQN model
class DQNModel1: 
    def __init__(self, name=None, environment_name=None,eval_freq=20000, buffer_size=1000):
        self.name = name # name of the game
        self.environment_name = environment_name# environment name
        self.eval_freq = eval_freq # evaluation frequency
        self.buffer_size = buffer_size # buffer size for the replay buffer
        self.log_path = os.path.join('C:/Users/Drashti Bhatt/PacmanDQN/Pacman_alphabots' + self.name + '_Log') # path for loging data
        self.save_path = os.path.join('C:/Users/Drashti Bhatt/PacmanDQN/Pacman_alphabots' + self.name +'_Model') # path for saving model
        self.env = self.make_environment() 
        self.model = self.build_model() 
    
    #environment function
    def make_environment(self): 
        env = gym.make(self.environment_name, render_mode="rgb_array") 
        return env 

    #DQN Model Function
    def build_model(self): # A call to the function that builds the DQN model
        model = DQN(CnnPolicy, self.env, verbose=0, 
                    tensorboard_log=self.log_path, buffer_size=self.buffer_size) 
        return model 
    
    #Single Episode function
    def execute_single_episode(self): 
        obs, _ = self.env.reset() # resets the environment
        done = False # sets the done flag
        score = 0 

        while not done: # loops until the done flag is 
            action= self.env.action_space.sample()  # choose an action from a sample randomly
            obs, reward, done, *info = self.env.step(action) 
            score += reward 

        return score 

    #Function to execute the episodes
    def execute_episodes(self, num_episodes=10, game_mode ="random"): 
        if game_mode == "random": # if the game mode is random
          print(f"Playing the  random{self.name} game for {num_episodes} episodes") # prints the message
          scores = [self.execute_single_episode() for _ in range(num_episodes)] 
          for episode, score in enumerate(scores, 1):
            print(f"Episode {episode}: {score}") 

        if game_mode == "predict": # if the game mode is predict
          episode_rewards = [] #list of episode rewards
          image_frames = [] #a list of image_frames for the images

          for episode in range(num_episodes): # loops through the number of episodes
              obs, _ = self.env.reset() 
              done = False 
              score = 0 
              
              while not done: 
                  action, _ = self.model.predict(obs) # predict actions
                  obs, reward, done, *info= self.env.step(action) 
                  score += reward 
                  frame = Image.fromarray(self.env.render()) #Capture image  from the environment
                  frame = np.array(frame) # converts to numpy
                  image_frames.append(frame)# adds the frame to the list

              episode_rewards.append(score) 

              print(f"Episode {episode+1}: {score}")

          video_path =  os.path.join(self.save_path, self.name + "_MS-Pacman") # video path


          save_video(image_frames, video_path, fps=30, name_prefix =f"{self.name}-MS-Pacman") #to save the video

    #Function to train agent
    def train_agent(self, time_steps=None, stop_value=None): 
        stop_callback = StopTrainingOnRewardThreshold(reward_threshold=stop_value, verbose=0) 
        eval_callback = EvalCallback(self.env, callback_on_new_best=stop_callback,
                                     eval_freq=self.eval_freq, best_model_save_path=self.save_path) 
        self.model.learn(total_timesteps=time_steps, callback=eval_callback) 

    #Policy function
    def policy(self, episodes=None): 
        mean_reward, reward_std = evaluate_policy(self.model, self.env, n_eval_episodes=episodes) # evaluates the policy
        print(f"Average reward over {episodes} episodes is {mean_reward} with a standard deviation {reward_std}") 

    def best_model(self):
        best_model = DQN.load(self.save_path + "/best_model")
        return best_model

    def save_model(self):
        return self.model.save(self.save_path)

    def close_env(self): 
        self.env.close() 
     
#%%
#initialize the agent and the environment
Pacman_agent = DQNModel1(name="Pacman", environment_name="MsPacman-v0")

#Play the pacman game randomly for 20 episodes
Pacman_agent.execute_episodes(num_episodes=20)
     
#train_agent 
Pacman_agent.train_agent(time_steps=20000, stop_value=670)   

#evaluate the policy used by the agent
Pacman_agent.policy(episodes=10)
     
Pacman_agent.save_model()
     
# test the agent 
Pacman_agent.execute_episodes(num_episodes=10, game_mode="predict")
     

#Close the environment
Pacman_agent.close_env()
