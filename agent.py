from collections import defaultdict
import numpy as np
from kaggle_environments.envs.football.helpers import Action, PlayerRole, GameMode
from game_cache import GameCache, DirCache
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
        self.dir_cache = DirCache(self.gc)

    def _run_towards(self, source, target):
        which_dir = int(((utils.angle([source[0], source[1]], [target[0], target[1]]) + 22.5) % 360) // 45)
        return self.dir_actions[which_dir]

    def _get_opponent_by_role(self, role):
        # TODO: check if opponent is out by red card
        opponent = [opp.pos for i, opp in enumerate(self.gc.players[OPP_TEAM]) if opp.role == role][0]
        return np.array(opponent)

    def _get_closest(self, point, team, steps=0):
        distances = [utils.distance(opp.get_future_pos(steps), point) for opp in self.gc.players[team]]
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

        if safe and closest_opp_dist < 0.01 and opp_between and (-0.2 > self.gc.controlled_player.pos[0] > -0.7):
            return self._run_towards(self.gc.controlled_player.pos, closest_opp.pos)

        return None

    def _decide_clear_ball(self):
        dist = utils.distance(self.own_goal, self.gc.ball[-1])
        danger_zone = dist < 0.4
        if not danger_zone and not self._last_man_standing():
            return None

        if self.action_counter[Action.Shot] > 9:
            return Action.Shot
        return None

    def _decide_passing(self):
        dist_to_goal = utils.distance(self.gc.controlled_player.pos, self.opp_goal)
        future_pos = self.gc.controlled_player.pos + 3*self.gc.get_player_speed()
        future_dir = self.opponent_penalty - future_pos
        future_dir /= utils.length(future_dir)
        future_pos += 4*0.01*future_dir
        direction = (future_pos - self.gc.controlled_player.pos)/7

        min_dist = np.inf
        for i in range(4, 8):
            dist, opp = self._get_closest(self.gc.controlled_player.pos + i*direction, OPP_TEAM, steps=i)
            opp_dist_to_goal = utils.distance(opp.pos, self.opp_goal)
            if dist < min_dist and dist_to_goal > opp_dist_to_goal:
                min_dist = dist

        return min_dist < 0.05

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
        if self.gc.controlled_player.role == PlayerRole.GoalKeeper:
            direction = self._run_towards(self.gc.controlled_player.pos, self.gc.ball[-1])
            return self.macro_list.add_macro([Action.Sprint, direction], False)

        dist_to_ball = utils.distance(self.gc.controlled_player.pos, self.gc.ball[-1])

        sprint_action = self._decide_sprint()
        if sprint_action is not None:
            return sprint_action

        if self.gc.neutral_ball:
            ball_speed = self.gc.get_ball_speed()
            ball_future = self.gc.ball[-1]
            for t in range(1, 21):
                ball_future = self.gc.ball[-1] + t*ball_speed
                if self.gc.ball_height[-1] < 0.5:
                    ball_speed = 0.9*ball_speed

                dist = utils.distance(ball_future, self.gc.controlled_player.pos)
                if dist / t < 0.01:
                    break

        else:
            ball_speed = self.gc.get_ball_speed()
            steps = dist_to_ball/0.01
            if steps <= 4:
                ball_future = self.gc.ball[-1] + steps * ball_speed
            else:
                ball_future = self.gc.ball[-1] + 4 * ball_speed
                direction = self.own_goal - ball_future
                direction /= utils.length(direction)
                ball_future += 0.01*direction*(steps-4)

        dir_action = self._run_towards(self.gc.controlled_player.pos, ball_future)

        if not self.gc.neutral_ball:
            direction = self._tactic_foul()
            if direction:
                return self.macro_list.add_macro([direction, Action.Slide], True)

        return dir_action

    def attack(self):
        if self.gc.current_obs["game_mode"] == GameMode.Penalty:
            return self.macro_list.add_macro([Action.Right, Action.Shot], True)

        if self.gc.controlled_player.role == PlayerRole.GoalKeeper:
            return self.macro_list.add_macro([Action.ReleaseSprint, Action.Right, Action.HighPass], True)

        if self.gc.controlled_player.pos[0] > 0.75:
            if self.gc.controlled_player.pos[1] > 0.2:
                return self.macro_list.add_macro([Action.ReleaseSprint, Action.HighPass] +
                                                 [Action.Top]*3, False)
            elif self.gc.controlled_player.pos[1] < -0.2:
                return self.macro_list.add_macro([Action.ReleaseSprint, Action.HighPass] +
                                                 [Action.Bottom] * 3, False)

        opp_gk = self._get_opponent_by_role(PlayerRole.GoalKeeper)
        dist_to_goal = utils.distance(self.gc.controlled_player.pos, self.opp_goal)
        dist_to_gk = utils.distance(self.gc.controlled_player.pos, opp_gk)
        looking_towards_goal = utils.cosine_sim(self.opp_goal - self.gc.controlled_player.pos,
                                                self.gc.controlled_player.direction) > 0.8
        if ((dist_to_goal < 0.3) or (dist_to_gk < 0.3)) and looking_towards_goal:
            if self.action_counter[Action.Shot] > 9:
                last_move = Action.Right
                if Action.Right in self.gc.sticky_actions:
                    last_move = Action.TopRight
                self.dir_cache.register(last_move)
                return self.macro_list.add_macro([Action.ReleaseSprint, last_move] + [Action.Shot]*3, True)

        sprint_action = self._decide_sprint()
        if sprint_action is not None:
            return sprint_action

        clear_action = self._decide_clear_ball()
        if clear_action is not None:
            return clear_action

        looking_forward = self.gc.controlled_player.direction[0] > 0

        if looking_forward:
            forward_teammates = [player.offside for player in self.gc.players[OWN_TEAM]
                                 if utils.distance(player.pos, self.opp_goal) < dist_to_goal]
            any_offside = any(forward_teammates)
            if self._decide_passing() and len(forward_teammates) > 0 and not any_offside:
                action = Action.LongPass
                if self.gc.controlled_player.pos[0] < -0.2:
                    action = Action.HighPass
                if self.action_counter[action] > 9:
                    return action
        else:
            closest_opp_dist, closest_opp = self._get_closest(self.gc.controlled_player.pos, OPP_TEAM)

            opp_between = utils.between(closest_opp.pos, self.gc.controlled_player.pos, self.opponent_penalty, -0.5)
            if opp_between and closest_opp_dist < 0.06:
                direction = self.gc.controlled_player.pos - closest_opp.pos
                direction[0] = 0
                current_dir = self.gc.controlled_player.direction
                if utils.cosine_sim(direction, current_dir) < -0.3:
                    return Action.Left
                if direction[1] > 0:
                    return Action.Bottom
                else:
                    return Action.Top

        if self.gc.controlled_player.pos[0] < 0.7:
            return self._run_towards(self.gc.controlled_player.pos, self.opponent_penalty)
        return self._run_towards(self.gc.controlled_player.pos, self.opp_goal)

    def act(self, obs):
        self.gc.update(obs)

        action = self.macro_list.step()
        if action is None:
            action = self.dir_cache.step()
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