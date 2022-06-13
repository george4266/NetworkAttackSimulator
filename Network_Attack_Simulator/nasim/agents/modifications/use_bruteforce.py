"""
Am getting minimum required code to make agent work
in order to be able to do more with it later. 

"""


from itertools import product
import pandas as pd
from rich import print as rprint
import nasim

LINE_BREAK = "-"*60


def run_bruteforce_agent(env, step_limit=1e6, verbose=True):

    env.reset()
    total_reward = 0
    done = False
    steps = 0
    cycle_complete = False

    if env.flat_actions:
        act = 0
    else:
        act_iter = product(*[range(n) for n in env.action_space.nvec])

    while not done and steps < step_limit:
        if env.flat_actions:
            act = (act + 1) % env.action_space.n
            cycle_complete = (steps > 0 and act == 0)
        else:
            try:
                act = next(act_iter)
                cycle_complete = False
            except StopIteration:
                act_iter = product(*[range(n) for n in env.action_space.nvec])
                act = next(act_iter)
                cycle_complete = True

        _, rew, done, x = env.step(act)
        total_reward += rew

        if cycle_complete and verbose:
            print(f"{steps}: {total_reward}")
        steps += 1
        df.loc[len(df.index)] = [_, total_reward,done, x]


        if done and verbose:
            print(f"Goal reached = {env.goal_reached()}")
            print(f"Total steps = {steps}")
            print(f"Total reward = {total_reward}")

        elif verbose:
            print("STEP LIMIT REACHED")
        if done = env.goal_reached()

        return steps, total_reward, done


if __name__ == "main":
    df = pd.DataFrame(columns=["Topology", "Total Reward", "Complete", "Actions"])

    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("env_name", type=str, help="benchmark scenario name")
    args = parser.parse_args()
    nasimenv = nasim.make_benchmark(
        args.env_name,
        args.seed,
        not args.partially_obs,
        not args.param_actions,
        not args.box_obs
    )
    if not args.param_actions:
        print(nasimenv.action_space.n)
    else:
        print(nasimenv.action_space.nvec)
    run_bruteforce_agent(nasimenv)
    df.to_csv("bruteforce_out.csv")
    



