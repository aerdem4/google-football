from collections import defaultdict
import numpy as np
from kaggle_environments.envs.football.helpers import Action, PlayerRole, GameMode
from game_cache import GameCache
import utils
from macros import MacroList


OPP_TEAM = "right_team"
OWN_TEAM = "left_team"


class Agent:
    def __init__(self):
        self.gc = GameCache()
        self.opp_goal = np.array([1.0, 0])
        self.opponent_penalty = np.array([0.85, 0])
        self.own_goal = np.array([-1.0, 0])
        self.own_penalty = np.array([-0.85, 0])
        self.dir_actions = [Action.Right, Action.BottomRight, Action.Bottom, Action.BottomLeft,
                            Action.Left, Action.TopLeft, Action.Top, Action.TopRight]
        self.dir_xy = np.array([[1, 0], [1, -1], [0, -1], [-1, -1],
                                [-1, 0], [-1, 1], [0, 1], [1, 1]])
        self.macro_list = MacroList(self.gc)
        self.action_counter = defaultdict(lambda: 99)

    def _run_towards(self, source, target):
        which_dir = int(((utils.angle([source[0], source[1]], [target[0], target[1]]) + 22.5) % 360) // 45)
        return self.dir_actions[which_dir]

    def _get_opponent_by_role(self, role):
        # TODO: check if opponent is out by red card
        opponent = [opp.pos for i, opp in enumerate(self.gc.players[OPP_TEAM]) if opp.role == role][0]
        return np.array(opponent)

    def _get_closest(self, point, team, exclude_gk=False):
        players = self.gc.players[team]
        if exclude_gk:
            players = [p for p in players if p.role != PlayerRole.GoalKeeper]
        distances = [utils.distance(opp.pos, point) for opp in players]
        closest = np.argmin(distances)
        return distances[closest], self.gc.players[team][closest]

    def _decide_sliding(self, intercept_point):
        close_enough = utils.distance(self.gc.controlled_player.pos, intercept_point) < 0.04
        _, closest_opp = self._get_closest(self.gc.controlled_player.pos, OPP_TEAM)
        opponent_between = utils.between(closest_opp.pos, self.gc.ball[-1], self.gc.controlled_player.pos,
                                         threshold=-0.5) and self.gc.get_player_speed()[0] > 0
        has_yellow = self.gc.controlled_player.yellow

        safe_to_slide = (self.gc.controlled_player.pos[0] > -0.7) and not has_yellow
        looking_towards = utils.cosine_sim(self.gc.ball[-1] - self.gc.controlled_player.pos,
                                           self.gc.controlled_player.direction) > 0.8
        last_man_dist, _ = self._get_closest(self.own_goal, OWN_TEAM, exclude_gk=True)
        last_man_standing = utils.distance(self.gc.controlled_player.pos, self.own_goal) < last_man_dist - 0.02
        if close_enough and safe_to_slide and looking_towards and (last_man_standing or not opponent_between):
            return True
        return False

    def _decide_clear_ball(self):
        if (self.gc.ball[-1][0] < -0.7) and (self.action_counter[Action.Shot] > 19):
            return Action.Shot
        if (self.gc.ball[-1][0] < -0.3) and (self.action_counter[Action.HighPass] > 19):
            return Action.HighPass
        return None

    def _decide_sprint(self):
        desired_dir = list(set(self.gc.sticky_actions).intersection(self.dir_actions))
        if len(desired_dir) == 1:
            desired_dir = desired_dir[0]
            which = np.argmax([desired_dir == d for d in self.dir_actions])
            on_dir = utils.cosine_sim(self.gc.controlled_player.direction, self.dir_xy[which]) > 0.2
            if not on_dir and Action.Sprint in self.gc.sticky_actions:
                return Action.ReleaseSprint
            if on_dir and Action.Sprint not in self.gc.sticky_actions:
                return Action.Sprint
        return None

    def defend(self):
        dist_to_ball = utils.distance(self.gc.controlled_player.pos, self.gc.ball[-1])

        speed_constant = 0.02
        intent = self.gc.get_ball_speed()
        if not self.gc.neutral_ball:
            path = self.own_goal - self.gc.ball[-1]
            intent = (intent + speed_constant*path/utils.length(path))/2
        intercept_point = self.gc.ball[-1] + intent*dist_to_ball/speed_constant

        #if self._decide_sliding(intercept_point):
        #    return Action.Slide

        dir_action = self._run_towards(self.gc.controlled_player.pos, intercept_point)
        sprint_action = self._decide_sprint()
        if dir_action in self.gc.sticky_actions and sprint_action is not None:
            return sprint_action
        return dir_action

    def attack(self):
        sprint_action = self._decide_sprint()
        if sprint_action is not None:
            return sprint_action

        clear_action = self._decide_clear_ball()
        if clear_action is not None:
            return self.macro_list.add_macro([Action.Right, clear_action], True)

        opp_gk = self._get_opponent_by_role(PlayerRole.GoalKeeper)
        dist_to_goal = utils.distance(self.gc.controlled_player.pos, self.opp_goal)
        dist_to_gk = utils.distance(self.gc.controlled_player.pos, opp_gk)
        looking_towards_goal = utils.cosine_sim(self.opp_goal - self.gc.controlled_player.pos,
                                                self.gc.controlled_player.direction) > 0.5
        if ((dist_to_goal < 0.3) or (dist_to_gk < 0.3)) and looking_towards_goal:
            if self.action_counter[Action.Shot] > 19:
                last_move = Action.Right
                if Action.Right in self.gc.sticky_actions:
                    last_move = Action.TopRight
                return self.macro_list.add_macro([Action.ReleaseSprint, last_move] + [Action.Shot]*3, True)

        obstacle_detected = self._get_closest(self.gc.controlled_player.pos +
                                              7*self.gc.get_player_speed(), OPP_TEAM)[0] < 0.05
        looking_forward = self.gc.controlled_player.direction[0] > 0
        forward_teammates = [player.offside for player in self.gc.players[OWN_TEAM]
                             if utils.distance(player.pos, self.opp_goal) < dist_to_goal - 0.05]
        any_offside = any(forward_teammates)
        if obstacle_detected and looking_forward and len(forward_teammates) > 0 and not any_offside:
            if self.action_counter[Action.LongPass] > 19:
                return Action.LongPass

        if self.gc.time_since_ball == 5:
            if self.gc.controlled_player.direction[0] < 0:
                return Action.ShortPass

        return self._run_towards(self.gc.controlled_player.pos, self.opponent_penalty)

    def act(self, obs):
        #try:
        self.gc.update(obs)

        action = self.macro_list.step()
        if action is None:
            if self.gc.neutral_ball:
                if self.gc.current_obs["game_mode"] == GameMode.Penalty:
                    return Action.Shot
                if self.gc.current_obs["game_mode"] == GameMode.Corner:
                    return Action.HighPass

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
