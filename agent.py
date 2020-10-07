from kaggle_environments.envs.football.helpers import Action, PlayerRole, GameMode
from game_cache import GameCache
import numpy as np
import utils


class Agent:
    def __init__(self):
        self.gc = GameCache()
        self.just_got_the_ball = 0
        self.opponent_goal = [1.0, 0]
        self.opponent_penalty = [0.8, 0]
        self.own_goal = [-1.0, 0]
        self.dir_actions = [Action.Right, Action.BottomRight, Action.Bottom, Action.BottomLeft,
                            Action.Left, Action.TopLeft, Action.Top, Action.TopRight]
        self.current_obs = None
        self.controlled_player_pos = None

    def _run_towards(self, source, target):
        which_dir = int(((utils.angle([source[0], source[1]], [target[0], target[1]]) + 22.5) % 360) // 45)
        return self.dir_actions[which_dir]

    def _get_opponent_by_role(self, role):
        # TODO: check if opponent is out by red card
        opponent = [i for i, r in enumerate(self.current_obs["right_team_roles"]) if r == role][0]
        return np.array(self.current_obs["right_team"])[opponent]

    def defend(self):
        self.just_got_the_ball = 0
        controlled_player_pos = self.current_obs['left_team'][self.current_obs['active']]
        if Action.Sprint not in self.current_obs['sticky_actions']:
            return Action.Sprint

        # ball is far
        if self.current_obs['ball'][0] - controlled_player_pos[0] < -0.1:
            self._run_towards(controlled_player_pos, self.own_goal)

        return self._run_towards(controlled_player_pos, self.current_obs['ball'])

    def attack(self):
        opp_gk = self._get_opponent_by_role(PlayerRole.GoalKeeper)

        dist_to_goal = utils.distance(self.controlled_player_pos, self.opponent_goal)
        dist_to_gk = utils.distance(self.controlled_player_pos, opp_gk)
        if (dist_to_goal < 0.3) or (dist_to_gk < 0.3):
            return Action.Shot

        self.just_got_the_ball += 1
        if self.just_got_the_ball == 5:
            if self.controlled_player_pos[0] < -0.2:
                return Action.HighPass

        if self.just_got_the_ball > 1:
            if Action.Sprint not in self.current_obs['sticky_actions']:
                return Action.Sprint

        return self._run_towards(self.controlled_player_pos, self.opponent_penalty)

    def act(self, obs):
        self.current_obs = obs
        self.controlled_player_pos = obs['left_team'][obs['active']]

        if obs["game_mode"] == GameMode.Penalty:
            return Action.Shot
        if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:
            return self.attack()
        else:
            return self.defend()
