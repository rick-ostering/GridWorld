

import gymnasium as gym
import numpy as np

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from grid_world_env import GridWorldEnv
from utils import check_make_dir, run_test
import os


# Get the directory path of the Python file
file_dir = os.path.dirname( __file__ )
log_dir = check_make_dir( file_dir, "log" )
model_file = os.path.join( log_dir , "best_model" )

grid_size = 20

eval_env = GridWorldEnv( render_mode = "human" , grid_size = grid_size )
eval_env.metadata[ "render_fps" ] = 1000

model = PPO.load( model_file )


for i in range( 2 ):
	print( "\nepisode: " , i+1 )
	run_test( eval_env , model , max_steps_per_episode = 600, sleep_time=0.03 )

eval_env.close()

