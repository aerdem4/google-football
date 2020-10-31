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

    def _get_closest(self, point, team):
        distances = [utils.distance(opp.pos, point) for opp in self.gc.players[team]]
        closest = np.argmin(distances)
        return distances[closest], self.gc.players[team][closest]

    def _decide_sliding(self):
        ball_dist = utils.distance(self.gc.ball[-1], self.gc.controlled_player.pos)
        closest_opp_dist, _ = self._get_closest(self.gc.controlled_player.pos, OPP_TEAM)
        has_yellow = self.gc.controlled_player.yellow

        safe_to_slide = (self.gc.controlled_player.pos[0] > -0.2) and not has_yellow
        # TODO: check dir
        if ball_dist + 0.02 < closest_opp_dist < 0.05 and safe_to_slide:
            return True
        return False

    def _last_man_standing(self):
        dist = utils.distance(self.gc.controlled_player.pos, self.own_goal)
        distances = [utils.distance(player.pos, self.own_goal) for player in self.gc.players[OWN_TEAM]
                     if player.role != PlayerRole.GoalKeeper]
        if sum([d < dist for d in distances]) == 0:
            return True
        return False

    def _tactic_foul(self):
        if not self._last_man_standing():
            return None

        closest_opp_dist, closest_opp = self._get_closest(self.gc.controlled_player.pos, OPP_TEAM)
        opp_between = utils.between(closest_opp.pos, self.gc.controlled_player.pos, self.gc.ball[-1], threshold=-0.5)

        safe = (not self.gc.controlled_player.yellow) or self.gc.time > 2800

        if safe and closest_opp_dist < 0.01 and opp_between and (-0.2 > closest_opp.pos[0] > -0.8):
            return self._run_towards(self.gc.controlled_player.pos, closest_opp.pos)

        return None

    def _decide_clear_ball(self):
        dist = utils.distance(self.own_goal, self.gc.ball[-1])
        danger_zone = dist < 0.5
        if not danger_zone and not self._last_man_standing():
            return None

        if self.gc.controlled_player.direction[0] > 0 and self.action_counter[Action.HighPass] > 9:
            return Action.HighPass

        closest_opp_dist, closest_opp = self._get_closest(self.gc.controlled_player.pos, OPP_TEAM)
        target_dir = self._run_towards(self.gc.controlled_player.pos, self.opponent_penalty)
        if closest_opp_dist > 0.06:
            return target_dir

        opp_between = utils.between(closest_opp.pos, self.gc.controlled_player.pos, self.opponent_penalty, -0.5)
        if opp_between:
            direction = self.gc.controlled_player.pos - closest_opp.pos
            direction /= utils.length(direction)
            current_dir = self.gc.controlled_player.direction / utils.length(self.gc.controlled_player.direction)
            direction += current_dir
            if direction[1] > 0:
                return Action.Bottom
            else:
                return Action.Top
        return None

    def defend(self):
        ball_coming = False
        ball_dist_now = utils.distance(self.gc.controlled_player.pos, self.gc.ball[-1])
        if len(self.gc.ball) > 1:
            ball_dist_prev = utils.distance(self.gc.controlled_player.pos, self.gc.ball[-2])
            ball_coming = 0.1 < ball_dist_now < ball_dist_prev
        if Action.Sprint not in self.gc.sticky_actions and not ball_coming:
            return Action.Sprint
        if Action.Sprint in self.gc.sticky_actions and ball_coming and self.gc.neutral_ball:
            return Action.ReleaseSprint

        time_projection = 7
        dir_action = self._run_towards(self.gc.controlled_player.pos,
                                       self.gc.ball[-1] + time_projection * self.gc.get_ball_speed())
        if not self.gc.neutral_ball:
            direction = self._tactic_foul()
            if direction:
                return self.macro_list.add_macro([direction, Action.Slide], True)

            if not utils.between(self.gc.controlled_player.pos, self.own_goal, self.gc.ball[-1], threshold=-0.5):
                between_point = (3*np.array(self.gc.ball[-1]) + np.array(self.own_goal))/4
                action = self._run_towards(self.gc.controlled_player.pos, between_point)
                if action in self.gc.sticky_actions:
                    return Action.Idle
                else:
                    return action

        return dir_action

    def attack(self):
        if self.gc.current_obs["game_mode"] == GameMode.Penalty:
            return self.macro_list.add_macro([Action.Right, Action.Shot], True)

        desired_dir = list(set(self.gc.sticky_actions).intersection(self.dir_actions))
        if len(desired_dir) == 1:
            desired_dir = desired_dir[0]
            which = np.argmax([desired_dir == d for d in self.dir_actions])
            on_dir = utils.cosine_sim(self.gc.controlled_player.direction, self.dir_xy[which]) > 0.2
            if not on_dir and Action.Sprint in self.gc.sticky_actions:
                return Action.ReleaseSprint
            if on_dir and Action.Sprint not in self.gc.sticky_actions:
                return Action.Sprint

        clear_action = self._decide_clear_ball()
        if clear_action is not None:
            return clear_action

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
        self.gc.update(obs)

        action = self.macro_list.step()
        if action is None:
            if self.gc.neutral_ball:
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
