import numpy as np
import random
import pickle

from modules import *


# random seed
seed = 10
np.random.seed(seed)
random.seed(seed)


# define parameters
n_world = 1000
n_test = 1000
sig_r = 0.3

n_depth = 5
n_branch_set = np.array([2, 3, 4])
rollout_depth_set = np.linspace(1, 5, 5, dtype = int)
budget_set = np.concatenate([np.arange(1, 9), [10, 12, 14, 16], 6 * np.arange(3, 21)]) # 25

# initialize recording
max_rewards = np.zeros((len(n_branch_set), n_world))
expected_rewards = np.zeros((len(n_branch_set), len(rollout_depth_set), len(budget_set), n_world))

best_actions = np.zeros((len(n_branch_set), n_world))
expected_accuracies = np.zeros((len(n_branch_set), len(rollout_depth_set),len(budget_set), n_world))


# simulation
# loop through branches
for i_branch, n_branch in enumerate(n_branch_set):

    # get parameters
    prms = get_prms(n_branch = n_branch, n_depth = n_depth, sig_r = sig_r)

    # loop thought environments
    for i_world in range(n_world):

        # assign reward means
        all_mus = np.array(assign_mus(prms, return_all = True))
        mus = np.sum(all_mus, axis = 1) # compute mus by summing all_mus over depth

        # get true rewards
        true_action_rewards = get_true_action_rewards(mus, prms) # (n_branch,)
        max_rewards[i_branch, i_world] = np.amax(true_action_rewards)

        # compute best action
        best_action = np.argmax(true_action_rewards)
        best_actions[i_branch, i_world] = best_action

        # create agent
        agent = agent_random(prms, n_test)

        # loop though sample number
        for i_rollout_depth, rollout_depth in enumerate(rollout_depth_set):

            for i_budget, budget in enumerate(budget_set):
            
                # reset buffer
                agent.reset_buffer()

                # compute sample number
                # sample number = total node expansion number / rollout depth per sample
                n_sample_raw = budget / rollout_depth
                n_sample = int(np.floor(n_sample_raw)) # round

                # deal with non-diviible case
                if (n_sample_raw - n_sample) > 1e-2: # if not divisible
                    if np.random.uniform() < (n_sample_raw - n_sample): # with probability n_sample_raw - n_sample
                        n_sample = int(n_sample + 1)

                # randomly choose an action if no samples
                if n_sample == 0:
                    choices = np.random.choice(np.arange(prms['n_branch']), n_test)

                else:
                    # loop through samples
                    for step in range(n_sample):
                        actions = agent.rollout(step) # (n_test,)
                        rewards = sample_rewards(actions, mus, prms, max_depth = rollout_depth, all_mus = all_mus) # (n_test,)
                        agent.update_buffer(actions, rewards)
                    
                    # make choice
                    choices = agent.make_choices()

                # compute reward
                exrew = np.mean(true_action_rewards[choices]) # (n_test,) -> (1,), average over n_test
                expected_rewards[i_branch, i_rollout_depth, i_budget, i_world] = exrew

                # compute accuracy
                exacc = np.mean(choices == best_action)
                expected_accuracies[i_branch, i_rollout_depth, i_budget, i_world] = exacc



# save data
data = {
    'n_depth': n_depth,
    'n_branch_set': n_branch_set,
    'rollout_depth_set': rollout_depth_set,
    'budget_set': budget_set,
    'n_world': n_world,
    'n_test': n_test,
    'sig_r': sig_r,
    'expected_rewards': expected_rewards,
    'max_rewards': max_rewards,
    'expected_accuracies': expected_accuracies,
}

with open('data/data_simulation/data_budget.p', 'wb') as f:
    pickle.dump(data, f)