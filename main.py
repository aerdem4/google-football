import sys
sys.path.insert(0, "/kaggle_simulations/agent")

from agent import Agent
from kaggle_environments.envs.football.helpers import human_readable_agent

lugano = Agent()


@human_readable_agent
def agent(obs):
    global lugano
    if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:
        return lugano.attack(obs)
    else:
        return lugano.defend(obs)
