import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from hiive.mdptoolbox.mdp import QLearning, ValueIteration, PolicyIteration


def run_solver(solver, q=False):
    stats = solver.run()
    x, r, mean_r, max_r, e, t = ([e[k] for e in stats] for k in (
        "Iteration", "Reward", "Mean V", "Max V", "Error", "Time"
    ))
    # print('gamma', solver.gamma)
    # print('p_cumulative', solver.p_cumulative)
    # print('policy', solver.policy)
    # print('thresh', solver.thresh)
    # print('iter', solver.iter)
    # print('max_iter', solver.max_iter)
    # print('error_mean', solver.error_mean)
    # print('time', solver.time)
    # print("policies", policies)
    # print("values", values)
    # print("epsilons", epsilons)
    return (x, r, mean_r, max_r, e, t), solver.iter, solver.policy, solver.time, mean_r[-1], max_r[-1], solver.error_mean[-1]


def policy(P, R, verbose=False):
    solver = PolicyIteration(
        P, R, 0.9999, max_iter=1000, eval_type=1,
    )
    if verbose:
        solver.setVerbose()
    return run_solver(solver)


def value(P, R, verbose=False):
    solver = ValueIteration(
        P, R, 0.9999, epsilon=0.01, max_iter=100000, initial_value=0,
    )
    if verbose:
        solver.setVerbose()
    return run_solver(solver)

frozen_lake_map = generate_random_map(size=30, p=0.8)
frozen_lake_map_str = ''.join(frozen_lake_map)
gym.envs.register(
    id="FrozenLake-v9",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    max_episode_steps=None,
    reward_threshold=None,
    order_enforce=False,
)

def q(P, R, verbose=False):
    solver = QLearning(
        P, R, 0.9999,
        iter_callback=check_if_new_episode,
    )
    if verbose:
        solver.setVerbose()
    return run_solver(solver)


def check_if_new_episode(old_s, action, new_s):
    if frozen_lake_map_str[new_s] == 'G' or frozen_lake_map_str[new_s] == 'H':
        return True
    else:
        return False
