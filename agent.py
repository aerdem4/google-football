from collections import defaultdict
import numpy as np
from kaggle_environments.envs.football.helpers import Action, PlayerRole, GameMode
from game_cache import GameCache
import utils
from macros import MacroList


class Agent:
    def __init__(self):
        self.gc = GameCache()
        self.opponent_goal = [1.0, 0]
        self.opponent_penalty = [0.85, 0]
        self.own_goal = [-1.0, 0]
        self.own_penalty = [-0.85, 0]
        self.dir_actions = [Action.Right, Action.BottomRight, Action.Bottom, Action.BottomLeft,
                            Action.Left, Action.TopLeft, Action.Top, Action.TopRight]
        self.controlled_player_pos = None
        self.macro_list = MacroList(self.gc)
        self.action_counter = defaultdict(lambda: 99)

    def _run_towards(self, source, target):
        which_dir = int(((utils.angle([source[0], source[1]], [target[0], target[1]]) + 22.5) % 360) // 45)
        return self.dir_actions[which_dir]

    def _get_opponent_by_role(self, role):
        # TODO: check if opponent is out by red card
        opponent = [i for i, r in enumerate(self.gc.current_obs["right_team_roles"]) if r == role][0]
        return np.array(self.gc.current_obs["right_team"])[opponent]

    def _get_closest_opponent(self, point):
        distances = [utils.distance(opp, point) for opp in self.gc.current_obs["right_team"]]
        closest = np.argmin(distances)
        return distances[closest], self.gc.current_obs["right_team"][closest]

    def _decide_sliding(self):
        ball_dist = utils.distance(self.gc.ball[-1], self.controlled_player_pos)
        closest_opp_dist, closest_opp = self._get_closest_opponent(self.controlled_player_pos)
        has_yellow = self.gc.current_obs["left_team_yellow_card"][self.gc.current_obs['active']]

        safe_to_slide = (self.controlled_player_pos[0] > -0.7) and not has_yellow
        # TODO: check dir
        if ball_dist + 0.01 < closest_opp_dist < 0.05 and safe_to_slide:
            return True
        return False

    def _decide_clear_ball(self):
        if (self.gc.ball[-1][0] < -0.4) and (self.action_counter[Action.HighPass] > 19):
            return True
        return False

    def defend(self):
        ball_coming = False
        if len(self.gc.ball) > 1:
            ball_dist_now = utils.distance(self.controlled_player_pos, self.gc.ball[-1])
            ball_dist_prev = utils.distance(self.controlled_player_pos, self.gc.ball[-2])
            ball_coming = 0.1 < ball_dist_now < ball_dist_prev
        if Action.Sprint not in self.gc.current_obs['sticky_actions'] and not ball_coming:
            return Action.Sprint
        if Action.Sprint in self.gc.current_obs['sticky_actions'] and ball_coming:
            return Action.ReleaseSprint

        if not self.gc.neutral_ball:
            if self._decide_sliding():
                return Action.Slide

            # ball is far
            if self.gc.ball[-1][0] - self.controlled_player_pos[0] < -0.1:
                return self._run_towards(self.controlled_player_pos, self.own_penalty)

        dir_action = self._run_towards(self.controlled_player_pos, self.gc.ball[-1] + 2*self.gc.get_ball_speed())
        if self._decide_clear_ball() and (dir_action in self.gc.current_obs["sticky_actions"]):
            return Action.HighPass

        return dir_action

    def attack(self):
        if self.gc.current_obs["game_mode"] == GameMode.Penalty:
            return Action.Shot
        if self.gc.current_obs["game_mode"] == GameMode.Corner:
            return Action.HighPass

        if self._decide_clear_ball():
            return self.macro_list.add_macro([Action.Right, Action.HighPass], True)

        direction = self.gc.current_obs["left_team_direction"][self.gc.current_obs["active"]]

        opp_gk = self._get_opponent_by_role(PlayerRole.GoalKeeper)
        dist_to_goal = utils.distance(self.controlled_player_pos, self.opponent_goal)
        dist_to_gk = utils.distance(self.controlled_player_pos, opp_gk)
        if (dist_to_goal < 0.3) or (dist_to_gk < 0.3) and (self.action_counter[Action.Shot] > 19):
            if Action.Right in self.gc.current_obs["sticky_actions"]:
                return self.macro_list.add_macro([Action.ReleaseSprint, Action.TopRight, Action.Shot], True)
            else:
                return self.macro_list.add_macro([Action.ReleaseSprint, Action.Right, Action.Shot], True)

        if self.gc.time_since_ball == 5:
            if direction[0] < 0 and abs(direction[0]) > abs(direction[1]):
                return Action.ShortPass

        if self.gc.time_since_ball > 1:
            if Action.Sprint not in self.gc.current_obs['sticky_actions']:
                return Action.Sprint

        return self._run_towards(self.controlled_player_pos, self.opponent_penalty)

    def act(self, obs):
        #try:
        self.gc.update(obs)
        self.controlled_player_pos = obs['left_team'][obs['active']]
        action = self.macro_list.step()
        if action is None:
            if self.gc.attacking[-1]:
                action = self.attack()
            else:
                action = self.defend()

        for k in self.action_counter.keys():
            self.action_counter[k] += 1
        self.action_counter[action] = 0
        return action
        #except:
        #    return Action.Shot
