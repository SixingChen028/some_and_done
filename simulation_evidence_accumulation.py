import numpy as np
import random
import pickle

from modules import *


# random seed
seed = 15
np.random.seed(seed)
random.seed(seed)


# define parameters
n_world = 2000
n_test = 2000
n_branch = 2
sig_r = 0.3

n_depth_set = np.linspace(1, 7, 7, dtype = int)
n_sample = 20


# initialize recording
max_rewards = np.zeros((len(n_depth_set), n_world))
expected_rewards = np.zeros((len(n_depth_set), n_world))
true_rewards = np.zeros((len(n_depth_set), n_world, n_branch))
raw_rewards = np.zeros((len(n_depth_set), n_world, n_branch, n_sample, n_test))


# simulation
# loop through depths
for i_depth, n_depth in enumerate(n_depth_set):

    # get parameters
    prms = get_prms(n_branch = n_branch, n_depth = n_depth, sig_r = sig_r)

    # loop thought environments
    for i_world in range(n_world):

        # draw a world
        mus = assign_mus(prms)

        # get true rewards
        true_action_rewards = get_true_action_rewards(mus, prms) # (n_branch,)
        max_rewards[i_depth, i_world] = np.amax(true_action_rewards)

        # get rollout actions
        # here we rollout each arm one time
        actions = np.arange(prms['n_branch'])[:, None, None] * np.ones((1, n_sample, n_test)) # (n_branch, n_sample, n_test)

        # get rewards
        rewards = sample_rewards(actions, mus, prms) # (2, n_sample, n_test)

        # make choices
        choices = np.argmax(np.mean(rewards, axis = 1), axis = 0) # (n_test)

        # compute reward
        exrew = np.mean(true_action_rewards[choices]) # (n_test,) -> (1,), average over n_test
        expected_rewards[i_depth, i_world] = exrew # (1,)
        true_rewards[i_depth, i_world, :] = true_action_rewards # (n_branch)
        raw_rewards[i_depth, i_world, ...] = rewards # (n_branch, n_sample. n_test)


# save data
data = {
    'n_depth_set': n_depth_set,
    'n_branch': n_branch,
    'n_sample': n_sample,
    'n_world': n_world,
    'n_test': n_test,
    'sig_r': sig_r,
    'max_rewards': max_rewards,
    'expected_rewards': expected_rewards,
    'true_rewards': true_rewards,
    'raw_rewards': raw_rewards,
}

with open('data/data_simulation/data_evidence_accumulation.p', 'wb') as f:
    pickle.dump(data, f)




# comparison simulation
n_sample_set = np.linspace(0, 20, 20, dtype = int)

max_rewards = np.zeros((len(n_depth_set), n_world))
expected_rewards = np.zeros((len(n_depth_set), len(n_sample_set), n_world))

# loop through depths
for i_depth, n_depth in enumerate(n_depth_set):

    # get parameters
    prms = get_prms(n_branch = n_branch, n_depth = n_depth, sig_r = sig_r)

    # loop thought environments
    for i_world in range(n_world):

        # draw a world
        mus = assign_mus(prms)

        # get true rewards
        true_action_rewards = get_true_action_rewards(mus, prms) # (n_branch,)
        max_rewards[i_depth, i_world] = np.amax(true_action_rewards)

        # loop through samples
        for i_sample, n_sample in enumerate(n_sample_set):

            # randomly choose an action if no samples
            if n_sample == 0:
                choices = np.random.choice(np.arange(prms['n_branch']), n_test)

            else:
                actions = np.arange(prms['n_branch'])[:, None, None] * np.ones((1, n_sample, n_test))
                rewards = sample_rewards(actions, mus, prms)
                choices = np.argmax(np.mean(rewards, axis = 1), axis = 0)

            # compute reward
            exrew = np.mean(true_action_rewards[choices]) # (n_test,) -> (1,), average over n_test
            expected_rewards[i_depth, i_sample, i_world] = exrew 


# save data
data = {
    'n_sample_set': n_sample_set,
    'max_rewards': max_rewards,
    'expected_rewards': expected_rewards,
}

with open('data/data_simulation/data_evidence_accumulation_comparison.p', 'wb') as f:
    pickle.dump(data, f)
