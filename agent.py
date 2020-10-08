from kaggle_environments.envs.football.helpers import Action, PlayerRole, GameMode
from game_cache import GameCache
import numpy as np
import utils


class Agent:
    def __init__(self):
        self.gc = GameCache()
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

    def _get_closest_opponent(self, point):
        distances = [utils.distance(opp, point) for opp in self.current_obs["right_team"]]
        closest = np.argmin(distances)
        return distances[closest], self.current_obs["right_team"][closest]

    def defend(self):
        if Action.Sprint not in self.current_obs['sticky_actions']:
            return Action.Sprint

        closest_opp_dist, closest_opp = self._get_closest_opponent(self.controlled_player_pos)
        if utils.distance(self.gc.ball[-1], self.controlled_player_pos) + 0.01 < closest_opp_dist < 0.05:
            return Action.Slide

        # ball is far
        if self.gc.ball[-1][0] - self.controlled_player_pos[0] < -0.1:
            self._run_towards(self.controlled_player_pos, self.own_goal)

        return self._run_towards(self.controlled_player_pos, self.gc.ball[-1] + 2*self.gc.get_ball_speed())

    def attack(self):
        direction = self.current_obs["left_team_direction"][self.current_obs["active"]]

        opp_gk = self._get_opponent_by_role(PlayerRole.GoalKeeper)
        dist_to_goal = utils.distance(self.controlled_player_pos, self.opponent_goal)
        dist_to_gk = utils.distance(self.controlled_player_pos, opp_gk)
        if (dist_to_goal < 0.3) or (dist_to_gk < 0.3):
            return Action.Shot

        if self.gc.time_since_ball == 5:
            if self.controlled_player_pos[0] < -0.4:
                return Action.HighPass
            elif direction[0] < 0 and abs(direction[0]) > abs(direction[1]):
                return Action.ShortPass

        if self.gc.time_since_ball > 1:
            if Action.Sprint not in self.current_obs['sticky_actions']:
                return Action.Sprint

        return self._run_towards(self.controlled_player_pos, self.opponent_penalty)

    def act(self, obs):
        #try:
        self.gc.update(obs)
        self.current_obs = obs
        self.controlled_player_pos = obs['left_team'][obs['active']]

        if obs["game_mode"] == GameMode.Penalty:
            return Action.Shot
        if obs["game_mode"] == GameMode.Corner:
            return Action.HighPass
        if self.gc.attacking[-1]:
            return self.attack()
        else:
            return self.defend()
        #except:
        #    return Action.Shot
