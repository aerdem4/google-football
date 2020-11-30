import sys
import gc
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
sys.path.insert(0, "/kaggle_simulations/agent")

import numpy as np
from kaggle_environments.envs.football.helpers import *
from agent import Agent54, Agent62
from game_cache import GameCache, DirCache
from macros import MacroList
from collections import defaultdict


gcache = GameCache()
macro_list = MacroList(gcache)
action_counter = defaultdict(lambda: 99)
dir_cache = DirCache(gcache)

lugano = Agent62(gcache, macro_list, action_counter, dir_cache)
alex = Agent54(gcache, macro_list, action_counter, dir_cache)


def agent(obs):
    gc.disable()

    action = Action.ReleaseDribble
    try:
        global lugano, alex, gcache
        obs = obs['players_raw'][0]
        # Turn 'sticky_actions' into a set of active actions (strongly typed).
        obs['sticky_actions'] = {sticky_index_to_action[nr] for nr, action in enumerate(obs['sticky_actions']) if action}
        # Turn 'game_mode' into an enum.
        obs['game_mode'] = GameMode(obs['game_mode'])
        # In case of single agent mode, 'designated' is always equal to 'active'.
        if 'designated' in obs:
            del obs['designated']
        # Conver players' roles to enum.
        obs['left_team_roles'] = [PlayerRole(role) for role in obs['left_team_roles']]
        obs['right_team_roles'] = [PlayerRole(role) for role in obs['right_team_roles']]

        gcache.update(obs)
        if gcache.time < 1500:
            action = lugano.act()
        else:
            action = alex.act()
    except Exception as e:
        print(e)

    gc.enable()
    return [action.value]
