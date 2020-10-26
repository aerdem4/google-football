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
        self.pass_discounts = [1.0, 1.0, 1.0, 0.75, 0.5, 0.75, 1.0, 1.0]
        self.macro_list = MacroList(self.gc)
        self.action_counter = defaultdict(lambda: 99)
        self.landmarks = [(self.own_goal, -20), (self.opp_goal, 20), (self.opponent_penalty, 20)]

    def _check_pass_counter(self, delay):
        return all([self.action_counter[action] > delay
                    for action in [Action.ShortPass, Action.LongPass, Action.HighPass]])

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
        opps = [(opp.pos, 1, 0.2) for opp in self.gc.players[OPP_TEAM]]
        boundaries = [[self.gc.controlled_player.pos[0], 0.45], [self.gc.controlled_player.pos[0], -0.45],
                      [[1, self.gc.controlled_player.pos[1]], [-1, self.gc.controlled_player.pos[1]]]]
        boundaries = [(np.array(b), 4, 0.1) for b in boundaries]
        landmarks = [(pos, -c, np.inf) for pos, c in self.landmarks]

        obstacle_points = defaultdict(float)
        for i in range(len(self.dir_actions)):
            action = self.dir_actions[i]
            direction = self.dir_xy[i]

            for pos, c, max_dist in landmarks + opps + boundaries:
                v = pos - self.gc.controlled_player.pos
                dist = max(0.02, utils.length(v))
                dir_sim = (utils.cosine_sim(v, direction) + 1) / 2
                if dist < max_dist:
                    obstacle_points[action] += c*dir_sim / dist

        dirs, points = list(obstacle_points.keys()), list(obstacle_points.values())

        return dirs[int(np.argmin(points))]

    def _get_pos_score(self, pos):
        pos_score = 0
        for landmark, c in self.landmarks:
            pos_score += c / max(0.02, utils.distance(pos, landmark))

        for opp in self.gc.players[OPP_TEAM]:
            dist = max(0.02, utils.distance(pos, opp.pos))
            pos_score -= 1/dist

        return pos_score

    def _get_best_pass_option(self):
        scores, pass_types = [], []
        base_pos_score = self._get_pos_score(self.gc.controlled_player.pos)
        for player in self.gc.players[OWN_TEAM]:
            v = player.pos - self.gc.controlled_player.pos
            if np.sum(np.abs(v)) == 0 or player.offside:  # same player or offside
                scores.append(-1)
                pass_types.append(Action.HighPass)
                continue

            pos_score = self._get_pos_score(player.pos)

            pos_advantage = max(pos_score - base_pos_score, 0)
            looking_towards = max(utils.cosine_sim(self.gc.controlled_player.direction, v), 0) + 1
            opp_in_between = 0

            for opp in self.gc.players[OPP_TEAM]:
                if utils.between(opp.pos, player.pos, self.gc.controlled_player.pos, threshold=-0.5):
                    opp_in_between += 1

            dist = utils.distance(self.gc.controlled_player.pos, player.pos)
            optimal_dist = 0.05 < dist < 1.0
            pass_convenience = 1/(1 + opp_in_between)

            pass_type = Action.ShortPass
            if dist > 0.25:
                pass_type = Action.LongPass
            if opp_in_between > 0:
                pass_type = Action.HighPass

            scores.append(pos_advantage*looking_towards*optimal_dist*pass_convenience)
            pass_types.append(pass_type)

        best_ind = int(np.argmax(scores))
        best_option = self.gc.players[OWN_TEAM][best_ind]
        best_score = scores[best_ind]
        best_option_dir = np.argmax([utils.cosine_sim(self.dir_xy[ix], best_option.pos - self.gc.controlled_player.pos)
                                     * self.pass_discounts[ix] for ix in range(len(self.dir_xy))])
        best_option_dir = self.dir_actions[int(best_option_dir)]

        return best_option_dir, best_score, pass_types[best_ind]

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
        opp_gk = self._get_opponent_by_role(PlayerRole.GoalKeeper)
        dist_to_goal = utils.distance(self.gc.controlled_player.pos, self.opp_goal)
        dist_to_gk = utils.distance(self.gc.controlled_player.pos, opp_gk)
        looking_towards_goal = utils.cosine_sim(self.opp_goal - self.gc.controlled_player.pos,
                                                self.gc.controlled_player.direction) > 0.5
        angle = np.abs(self.gc.ball[-1] - self.opp_goal)
        if ((dist_to_goal < 0.3) or (dist_to_gk < 0.3)) and looking_towards_goal and angle[0] > angle[1]:
            if self.action_counter[Action.Shot] > 19:
                return self.macro_list.add_macro([Action.ReleaseSprint, Action.Right] + [Action.Shot]*3, True)

        future_pos = self.gc.controlled_player.pos + 7*self.gc.get_player_speed()
        obstacle_detected = self._get_closest(future_pos, OPP_TEAM)[0] < 0.05
        if abs(future_pos[0]) > 0.9 or abs(future_pos[1]) > 0.4:
            obstacle_detected = True

        best_option_dir, best_score, pass_type = self._get_best_pass_option()
        if best_score > 50*(2-obstacle_detected) and self._check_pass_counter(9):
            return self.macro_list.add_macro([best_option_dir, Action.ReleaseSprint, pass_type] +
                                             [best_option_dir]*5, False)

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
