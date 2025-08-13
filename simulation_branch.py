import numpy as np
import random
import pickle

from modules import *


# random seed
seed = 15
np.random.seed(seed)
random.seed(seed)


# define parameters
n_world = 1000
n_test = 1000
n_depth = 2
sig_r = 0.3

n_branch_set = np.linspace(2, 8, 7, dtype = int)
n_sample_set = np.linspace(0, 20, 20, dtype = int)


# initialize recording
max_rewards = np.zeros((len(n_branch_set), n_world))
expected_rewards = np.zeros((len(n_branch_set), len(n_sample_set), n_world))

best_actions = np.zeros((len(n_branch_set), n_world))
expected_accuracies = np.zeros((len(n_branch_set), len(n_sample_set), n_world))


# simulation
# loop through branching factors
for i_branch, n_branch in enumerate(n_branch_set):

    # get parameters
    prms = get_prms(n_branch = n_branch, n_depth = n_depth, sig_r = sig_r)

    # loop thought environments
    for i_world in range(n_world):

        # draw a world
        mus = assign_mus(prms)

        # get true rewards
        true_action_rewards = get_true_action_rewards(mus, prms) # (n_branch,)
        max_rewards[i_branch, i_world] = np.amax(true_action_rewards)

        # compute best action
        best_action = np.argmax(true_action_rewards)
        best_actions[i_branch, i_world] = best_action

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
            expected_rewards[i_branch, i_sample, i_world] = exrew

            # compute accuracy
            exacc = np.mean(choices == best_action)
            expected_accuracies[i_branch, i_sample, i_world] = exacc
            

    for i_sample, n_sample in enumerate(n_sample_set):
        exrew = expected_rewards[i_branch, i_sample, :]
        print(
            'branch:', n_branch, '|',
            'sample num:', n_sample, '|',
            'reward mean:', np.round(np.mean(exrew), 3), '|',
            'reward std:', np.round(np.std(exrew) / np.sqrt(n_world), 4)
        )
    print()



# save data
data = {
    'n_depth': n_depth,
    'n_branch_set': n_branch_set,
    'n_sample_set': n_sample_set,
    'n_world': n_world,
    'n_test': n_test,
    'sig_r': sig_r,
    'expected_rewards': expected_rewards,
    'max_rewards': max_rewards,
    'expected_accuracies': expected_accuracies,
}

with open('data/data_simulation/data_branch.p', 'wb') as f:
    pickle.dump(data, f)