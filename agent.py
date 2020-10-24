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
        self.dir_xy = np.array([[1, 0], [1, 1], [0, 1], [-1, 1],
                                [-1, 0], [-1, -1], [0, -1], [1, -1]])
        self.macro_list = MacroList(self.gc)
        self.action_counter = defaultdict(lambda: 99)

    def _neigbor_dirs(self, direction):
        index = int(np.argmax([d == direction for d in self.dir_actions]))
        next_index = (index + 1) % len(self.dir_actions)
        prev_index = (index + len(self.dir_actions) - 1) % len(self.dir_actions)
        return [self.dir_actions[prev_index], self.dir_actions[index], self.dir_actions[next_index]]

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
        closest = int(np.argmin(distances))
        return distances[closest], self.gc.players[team][closest]

    def _decide_sliding(self):
        close_enough = utils.distance(self.gc.controlled_player.pos, self.gc.ball[-1]) < 0.04
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
        press_dist, press_player = self._get_closest(self.gc.ball[-1], OPP_TEAM, exclude_gk=True)
        if self.gc.ball[-1][0] > -0.3 or press_dist > 0.2:
            return None
        if self.gc.controlled_player.direction[0] > 0 and (self.action_counter[Action.HighPass] > 9):
            return [Action.ReleaseSprint, Action.Right] + [Action.HighPass]*3 + [Action.Idle]
        return None

    def _get_desired_dir(self):
        desired_dir = list(set(self.gc.sticky_actions).intersection(self.dir_actions))
        if len(desired_dir) == 1:
            desired_dir = desired_dir[0]
            which = np.argmax([desired_dir == d for d in self.dir_actions])
            return which
        return None

    def _decide_sprint_attack(self):
        which_dir = self._get_desired_dir()
        if which_dir is not None:
            on_dir = utils.cosine_sim(self.gc.controlled_player.direction, self.dir_xy[which_dir]) > 0.2
            if not on_dir and Action.Sprint in self.gc.sticky_actions:
                return Action.ReleaseSprint
            if on_dir and Action.Sprint not in self.gc.sticky_actions:
                return Action.Sprint
        return None

    def _decide_sprint_defense(self):
        ball_coming = False
        if len(self.gc.ball) > 1:
            ball_dist_now = utils.distance(self.gc.controlled_player.pos, self.gc.ball[-1])
            ball_dist_prev = utils.distance(self.gc.controlled_player.pos, self.gc.ball[-2])
            ball_coming = 0.1 < ball_dist_now < ball_dist_prev
        if Action.Sprint not in self.gc.sticky_actions and not ball_coming:
            return Action.Sprint
        if Action.Sprint in self.gc.sticky_actions and ball_coming and self.gc.neutral_ball:
            return Action.ReleaseSprint
        return None

    def _decide_dir(self):
        vectors = [opp.pos - self.gc.controlled_player.pos for opp in self.gc.players[OPP_TEAM]]
        obstacle_points = defaultdict(float)
        for i in range(len(self.dir_actions)):
            action = self.dir_actions[i]
            direction = self.dir_xy[i]

            for goal, c in [(self.opponent_penalty, -20), (self.own_goal, 20)]:
                v = goal - self.gc.controlled_player.pos
                dist = max(0.02, utils.length(v))
                dir_sim = (utils.cosine_sim(v, direction) + 1) / 2
                obstacle_points[action] += c*dir_sim / dist

            for v in vectors:
                dist = max(0.02, utils.length(v))
                dir_sim = (utils.cosine_sim(v, direction) + 1)/2
                obstacle_points[action] += dir_sim / dist

            next_pos = np.abs(self.gc.controlled_player.pos + direction*0.1)
            if next_pos[0] > 0.9 or next_pos[1] > 0.4:
                obstacle_points[action] = np.inf

        dirs, points = list(obstacle_points.keys()), list(obstacle_points.values())

        return dirs[int(np.argmin(points))]

    def _get_best_pass_option(self):
        scores = []
        dist_to_opp_goal = max(0.02, utils.distance(self.gc.controlled_player.pos, self.opp_goal))
        dist_to_own_goal = max(0.02, utils.distance(self.gc.controlled_player.pos, self.own_goal))
        pos_score = dist_to_own_goal/dist_to_opp_goal
        for player in self.gc.players[OWN_TEAM]:
            v = player.pos - self.gc.controlled_player.pos
            if np.sum(np.abs(v)) == 0 or player.offside:  # same player or offside
                scores.append(-1)
                continue

            dist_to_opp_goal = max(0.02, utils.distance(player.pos, self.opp_goal))
            dist_to_own_goal = max(0.02, utils.distance(player.pos, self.own_goal))

            ps_rate = max(dist_to_own_goal/dist_to_opp_goal/pos_score - 1, 0)
            looking_towards = max(utils.cosine_sim(self.gc.controlled_player.direction, v), 0)*2
            opp_in_between = 0
            for opp in self.gc.players[OPP_TEAM]:
                if utils.between(opp.pos, player.pos, self.gc.controlled_player.pos, threshold=-0.5):
                    opp_in_between += 1

            optimal_dist = 0.05 < utils.distance(self.gc.controlled_player.pos, player.pos) < 0.5

            scores.append(ps_rate*looking_towards*optimal_dist*(opp_in_between == 0))

        best_option = self.gc.players[OWN_TEAM][int(np.argmax(scores))]
        best_score = np.max(scores)
        best_option_dir = np.argmax([utils.cosine_sim(xy, best_option.pos - self.gc.controlled_player.pos)
                                     for xy in self.dir_xy])
        best_option_dir = self.dir_actions[int(best_option_dir)]

        return best_option_dir, best_score

    def defend(self):
        sprint_action = self._decide_sprint_defense()
        if sprint_action is not None:
            return sprint_action

        ball_speed = self.gc.get_ball_speed()
        ball_close_to_goal = utils.distance(self.own_goal, self.gc.ball[-1]) < 0.3
        if self.gc.controlled_player.role == PlayerRole.GoalKeeper or ball_close_to_goal:
            return self._run_towards(self.gc.controlled_player.pos, self.gc.ball[-1] + 2*ball_speed)

        dir_action = self._run_towards(self.gc.controlled_player.pos, self.gc.ball[-1] + 9*ball_speed)
        dist_to_ball = utils.distance(self.gc.controlled_player.pos, self.gc.ball[-1])

        if not self.gc.neutral_ball:
            in_between = utils.between(self.gc.controlled_player.pos, self.own_goal, self.gc.ball[-1], threshold=-0.5)
            if not in_between or (dist_to_ball > 0.1 and self.gc.get_ball_speed()[0] < 0 and self.gc.ball[-1][0] < 0):
                between_point = (3*np.array(self.gc.ball[-1]) + np.array(self.own_goal))/4
                dist_to_intercept = utils.distance(self.gc.controlled_player.pos, between_point)

                teammates = self.gc.players[OWN_TEAM]
                teammates = [p for p in teammates if p.role != PlayerRole.GoalKeeper]
                teammates_available = sum([utils.distance(p.pos, between_point) + 0.02 < dist_to_intercept
                                           for p in teammates])

                if teammates_available > 1 and dist_to_ball > 0.1:
                    return Action.ReleaseDirection

                action = self._run_towards(self.gc.controlled_player.pos, between_point)
                if action in self.gc.sticky_actions:
                    return Action.Idle
                else:
                    return action

        #if self._decide_sliding():
        #    return Action.Slide

        return dir_action

    def attack(self):
        clear_action = self._decide_clear_ball()
        if clear_action is not None:
            return self.macro_list.add_macro(clear_action, True)

        opp_gk = self._get_opponent_by_role(PlayerRole.GoalKeeper)
        dist_to_goal = utils.distance(self.gc.controlled_player.pos, self.opp_goal)
        dist_to_gk = utils.distance(self.gc.controlled_player.pos, opp_gk)
        looking_towards_goal = utils.cosine_sim(self.opp_goal - self.gc.controlled_player.pos,
                                                self.gc.controlled_player.direction) > 0.5
        angle = np.abs(self.gc.ball[-1] - self.opp_goal)
        if ((dist_to_goal < 0.3) or (dist_to_gk < 0.3)) and looking_towards_goal and angle[0] > angle[1]:
            if self.action_counter[Action.Shot] > 19:
                return self.macro_list.add_macro([Action.ReleaseSprint] + [Action.Shot]*3, True)

        obstacle_detected = self._get_closest(self.gc.controlled_player.pos +
                                              7*self.gc.get_player_speed(), OPP_TEAM)[0] < 0.05
        looking_forward = self.gc.controlled_player.direction[0] >= 0
        forward_teammates = [player.offside for player in self.gc.players[OWN_TEAM]
                             if utils.distance(player.pos, self.opp_goal) < dist_to_goal - 0.05]
        any_offside = any(forward_teammates)
        if obstacle_detected and looking_forward and len(forward_teammates) > 0 and not any_offside:
            if self.action_counter[Action.LongPass] > 19:
                return Action.LongPass

        marked = self._get_closest(self.gc.controlled_player.pos, OPP_TEAM)[0] < 0.1
        best_option_dir, best_score = self._get_best_pass_option()
        if marked and best_score > 0.1 and self.action_counter[Action.ShortPass] > 9:
            return self.macro_list.add_macro([best_option_dir, Action.ShortPass, best_option_dir], False)

        direction = self._decide_dir()
        sprint_action = self._decide_sprint_attack()
        if sprint_action is not None and direction in self.gc.sticky_actions:
            return sprint_action
        return direction

    def act(self, obs):
        #try:
        self.gc.update(obs)

        action = self.macro_list.step()
        if action is None:
            if self.gc.neutral_ball:
                if self.gc.current_obs["game_mode"] == GameMode.Penalty:
                    return self.macro_list.add_macro([Action.Right] + [Action.Shot]*3, True)
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
