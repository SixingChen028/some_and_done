import numpy as np
import random
import pickle

from modules import *


# random seed
seed = 15
np.random.seed(seed)
random.seed(seed)


# define parameters
n_world = 500
n_test = 500
sig_r = 0.3

correlation_set = np.array([True, False])
n_prob = 4
n_branch_set = np.linspace(2, 5, 4, dtype = int)
n_depth_set = np.linspace(1, 6, 6, dtype = int)
n_sample_set = np.concatenate([np.linspace(0, 20, 20, dtype = int), [25, 30, 45, 50, 70, 90]])


# initialize recording
max_rewards = np.zeros((len(correlation_set), len(n_branch_set), len(n_depth_set), n_prob, n_world))
expected_rewards = np.zeros((len(correlation_set), len(n_branch_set), len(n_depth_set), n_prob, len(n_sample_set), n_world))

best_actions = np.zeros((len(correlation_set), len(n_branch_set), len(n_depth_set), n_prob, n_world))
expected_accuracies = np.zeros((len(correlation_set), len(n_branch_set), len(n_depth_set), n_prob, len(n_sample_set), n_world))


# simulation
# loop though correlation
for i_correlation, correlation in enumerate(correlation_set):

    # loop through branches
    for i_branch, n_branch in enumerate(n_branch_set):

        # loop through depths
        for i_depth, n_depth in enumerate(n_depth_set):

            # get probability set
            prob_set = np.linspace(1 / n_branch, 1, n_prob)

            # loop through probabilities
            for i_prob, prob in enumerate(prob_set):

                # get parameters
                prms = get_prms(n_branch = n_branch, n_depth = n_depth, sig_r = sig_r)

                # loop thought environments
                for i_world in range(n_world):

                    # draw a world
                    # structured task with correlations
                    if correlation:
                        mus = assign_mus(prms)
                    
                    # no correlations within branches
                    else:
                        mus = np.random.normal(0, prms['sig_b'] * np.sqrt(prms['n_depth']), prms['n_leaves'])

                    # get true rewards
                    true_action_rewards = get_true_action_rewards(mus, prms) # (n_branch,)
                    max_rewards[i_correlation, i_branch, i_depth, i_prob, i_world] = np.amax(true_action_rewards)

                    # compute best action
                    best_action = np.argmax(true_action_rewards)
                    best_actions[i_correlation, i_branch, i_depth, i_prob, i_world] = best_action

                    # create agent
                    agent = agent_random(prms, n_test)

                    # loop though sample number
                    for i_sample, n_sample in enumerate(n_sample_set):
                        
                        # reset buffer
                        agent.reset_buffer()

                        # randomly choose an action if no samples
                        if n_sample == 0:
                            choices = np.random.choice(np.arange(prms['n_branch']), n_test)
                        
                        else:
                            # loop through samples
                            for step in range(n_sample):
                                actions = agent.rollout(step) # (n_test,)
                                rewards = sample_rewards(actions, mus, prms) # (n_test,)
                                agent.update_buffer(actions, rewards)
                            
                            # make choice
                            choices = agent.make_choices()

                        # compute reward
                        exrew = np.mean(true_action_rewards[choices]) # (n_test,) -> (1,), average over n_test
                        expected_rewards[i_correlation, i_branch, i_depth, i_prob, i_sample, i_world] = exrew

                        # compute accuracy
                        exacc = np.mean(choices == best_action)
                        expected_accuracies[i_correlation, i_branch, i_depth, i_prob, i_sample, i_world] = exacc
                        

# save data
data = {
    'correlation_set': correlation_set,
    'n_depth_set': n_depth_set,
    'n_branch_set': n_branch_set,
    'n_sample_set': n_sample_set,
    'n_prob': n_prob,
    'n_world': n_world,
    'n_test': n_test,
    'sig_r': sig_r,
    'expected_rewards': expected_rewards,
    'max_rewards': max_rewards,
    'expected_accuracies': expected_accuracies,
}

with open('data/data_simulation/data_factor.p', 'wb') as f:
    pickle.dump(data, f)