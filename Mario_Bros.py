# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 18:54:50 2022

@author: Rufus Sam A
"""

# Import the game
import gym_super_mario_bros
# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
# Import the SIMPLIFIED controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Setup game
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Create a flag - restart or not
done = True
# Loop through each frame in the game
for step in range(100000000): 
    # Start the game to begin with 
    if done: 
        # Start the gamee
        env.reset()
    # Do random actions
    state, reward, done, info = env.step(env.action_space.sample())
    # Show the game on the screen
    env.render()
# Close the game
env.close()

# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Import Matplotlib to show the impact of frame stacking
from matplotlib import pyplot as plt

# 1. Create the base environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
# 2. Simplify the controls 
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order='last')

state = env.reset()

state, reward, done, info = env.step([5])

plt.figure(figsize=(20,16))
for idx in range(state.shape[3]):
    plt.subplot(1,4,idx+1)
    plt.imshow(state[0][:,:,idx])
plt.show()

#3. Train the RL Model

# Import os for file path management
import os 
# Import PPO for algos
from stable_baselines3 import PPO
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
    
CHECKPOINT_DIR = r'C:\Users\Rufus Sam A\Downloads\Lincoln\Project\Super_mario\train'
LOG_DIR = r'C:\Users\Rufus Sam A\Downloads\Lincoln\Project\Super_mario\logs'
#CHECKPOINT_DIR = './train'
#LOG_DIR = './log'

# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# This is the AI model started
model = PPO('MlpPolicy', env, learning_rate=0.001, n_steps=1024, batch_size=32, n_epochs=15, 
            gamma=0.45, gae_lambda=0.40, clip_range=0.2, clip_range_vf=None, normalize_advantage=True, 
            ent_coef=0.0, vf_coef=0.9, max_grad_norm=0.3, use_sde=False, sde_sample_freq=- 9, 
            target_kl=None, tensorboard_log=LOG_DIR, create_eval_env=False, policy_kwargs=None, 
            verbose=1, seed=100, device='auto', _init_setup_model=True)


# Train the AI model, this is where the AI model starts to learn
model.learn(total_timesteps=1000000, callback=callback)

model.save('thisisatestmodel')

# Load model
model = PPO.load('./train/best_model_5000000')

state = env.reset()

# Start the game 
state = env.reset()
# Loop through the game
while True: 
    
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()