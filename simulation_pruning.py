import numpy as np
import random
import pickle

from modules import *


# random seed
seed = 15
np.random.seed(seed)
random.seed(seed)


# define parameters
n_world = 5000
n_test = 2000
n_branch = 2
sig_r = 0.3

noise_set = np.linspace(0.0, 0.5, 20)
n_depth_set = np.linspace(2, 5, 4, dtype = int)
n_sample = 4


# initialize recording
choices_pruning = np.zeros((2, len(n_depth_set), len(noise_set), n_world)) # infinite samples vs. finite (5) samples

# simulation
# loop through depths
for i_depth, n_depth in enumerate(n_depth_set):

    # get parameters
    prms = get_prms(n_branch = n_branch, n_depth = n_depth, sig_r = sig_r)
    n_leaves = prms['n_leaves']
    n_half = int(n_leaves / prms['n_branch'])

    # loop throught noises
    for i_noise, noise in enumerate(noise_set):

        # set transition probability
        p = [1 - noise, noise]

        # loop thought environments
        for i_world in range(n_world):

            # pruning part
            mus = assign_mus(prms)

            # set large penalty and large reward
            mus[:n_half] -= 10 # big penalty at the start for one depth-1 node
            mus[0] += 15 # big reward at the end for one leaf node

            # get true rewards
            true_action_rewards = get_true_action_rewards(mus, prms, p = p) # (n_branch,)

            # create agent
            agent = agent_random(prms, n_test)

            # reset buffer
            agent.reset_buffer()

            # loop through samples
            for step in range(n_sample):
                actions = agent.rollout(step) # (n_test,)
                rewards = sample_rewards(actions, mus, prms, p = p) # (n_test,)
                agent.update_buffer(actions, rewards)
                
            # make choice
            choices = agent.make_choices()

            # record
            choices_pruning[0, i_depth, i_noise, i_world] = np.argmax(true_action_rewards) # get choices from real values
            choices_pruning[1, i_depth, i_noise, i_world] = np.mean(choices) # get choices from finite samples

# save data
data = {
    'n_depth_set': n_depth_set,
    'n_branch': n_branch,
    'n_sample': n_sample,
    'noise_set': noise_set,
    'n_world': n_world,
    'n_test': n_test,
    'sig_r': sig_r,
    'choices_pruning': choices_pruning,
}

with open('data/data_simulation/data_pruning.p', 'wb') as f:
    pickle.dump(data, f)