
from os import path
from os import mkdir
from time import sleep

def check_make_dir( dir_name, file_name ):
    full_path = path.join( dir_name , file_name )
    if not path.isdir( full_path ):
        mkdir( full_path )
    return full_path


def run_test( env, model, max_steps_per_episode = 300, sleep_time = 0 ):
    if env.run == True:
        obs, _ = env.reset()
        acc_reward = 0
        for i in range( max_steps_per_episode ):
            action, _ = model.predict( obs, deterministic = False )
            obs, reward, term, trunc, info = env.step( action )
            acc_reward += reward
            sleep( sleep_time )
            if term:
                break
            if env.run == False:
                break
        
        print( "steps: " , i , "  reward: ", acc_reward, "score: ", env.food_eaten )
