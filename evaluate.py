import os
import multiprocessing
import argparse
import numpy as np
from kaggle_environments import make


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate agent.')
    parser.add_argument('--benchmark', help='Benchmark agent filename')
    parser.add_argument('--num-games', type=int, default=16, help='Number of games to run')
    args = parser.parse_args()

    os.system("cp *.py /kaggle_simulations/agent/")
    print(f"Running {args.num_games} matches against {args.benchmark}")


    def get_reward(x):
        np.random.seed(x)
        env = make("football", configuration={"save_video": False, "scenario_name": "11_vs_11_kaggle",
                                              "running_in_notebook": True, "debug": True})
        opp_agent = args.benchmark
        agents = ["/kaggle_simulations/agent/main.py"]
        if x % 2 == 0:
            agents = agents + [opp_agent]
            which = 0
        else:
            agents = [opp_agent] + agents
            which = 1

        output = env.run(agents)[-1]
        reward = output[which]["reward"]

        if reward is None:
            return env.logs[-1]
        return reward

    pool_size = multiprocessing.cpu_count()
    print(f"Number of matches in parallel: {pool_size}")
    pool = multiprocessing.Pool(processes=pool_size)
    rewards = pool.map(get_reward, [i for i in range(args.num_games)])
    rewards = np.array(rewards)
    pool.close()
    pool.join()

    avg_score_diff = np.round(np.mean(rewards), 4)
    results = np.unique(np.sign(rewards), return_counts=True)

    print(f"Average score difference: {avg_score_diff}")
    print(f"Won {(rewards > 0).sum()} games.")
    print(f"Draw {(rewards == 0).sum()} games.")
    print(f"Lost {(rewards < 0).sum()} games.")
