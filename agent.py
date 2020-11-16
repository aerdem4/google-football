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
        self.dir_xy = np.array([[1, 0], [1, 1], [0, 1], [-1, 1],
                                [-1, 0], [-1, -1], [0, -1], [1, -1]], dtype=np.float32)
        for i in range(len(self.dir_xy)):
            self.dir_xy[i] = self.dir_xy[i] / utils.length(self.dir_xy[i])
        self.macro_list = MacroList(self.gc)
        self.action_counter = defaultdict(lambda: 99)
        self.dir_cache = DirCache(self.gc)

    def _check_action_counter(self, t):
        for action in [Action.ShortPass, Action.LongPass, Action.HighPass, Action.Shot]:
            if self.action_counter[action] <= t:
                return False
        return True

    def _run_towards(self, target, c=0.0):
        source = self.gc.controlled_player.pos
        v = target - source
        dir_score = np.zeros(len(self.dir_actions))
        obstacles = [p for p in self.gc.players[OPP_TEAM]
                     if utils.distance(p.pos, source) < 0.2]

        speed = self.gc.get_ball_speed()
        magnitude = utils.length(speed)
        for i in range(len(self.dir_actions)):
            dir_score[i] = (1 + utils.cosine_sim(v, self.dir_xy[i]))
            if magnitude > 0.004:
                dir_score[i] *= (utils.cosine_sim(speed, self.dir_xy[i]) > 0)
            if c > 0:
                for obs in obstacles:
                    avoidance_now = max(0, utils.cosine_sim(obs.pos - source, self.dir_xy[i]))**2
                    avoidance_future = max(0, utils.cosine_sim(obs.get_future_pos(9) - source, self.dir_xy[i])) ** 2
                    avoidance = max(avoidance_now, avoidance_future)
                    dir_score[i] -= c*avoidance

                future_loc = source + self.dir_xy[i]*0.1
                if abs(future_loc[0]) > 0.9 or abs(future_loc[1]) > 0.4 or \
                        utils.distance(future_loc, self.own_goal) < 0.2:
                    dir_score[i] = -np.inf
        which_dir = int(np.argmax(dir_score))
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
            return self._run_towards(closest_opp.pos)

        return None

    def _decide_clear_ball(self):
        dist = utils.distance(self.own_goal, self.gc.ball[-1])
        danger_zone = dist < 0.4
        if not danger_zone and not self._last_man_standing():
            return None

        closest_opp_dist, _ = self._get_closest(self.gc.controlled_player.pos, OPP_TEAM)

        if closest_opp_dist < 0.1 and self.action_counter[Action.Shot] > 9:
            return Action.Shot
        return None

    def _detect_obstacle(self, player):
        future_pos = player.get_future_pos(3)
        future_dir = self.opponent_penalty - future_pos
        future_dir /= utils.length(future_dir)
        future_pos += 4*0.01*future_dir

        obstacles = [opp.get_future_pos(7) for opp in self.gc.players[OPP_TEAM]]
        detected = any([utils.distance(future_pos, o) < 0.05 for o in obstacles])

        return detected

    def _decide_sprint(self):
        desired_dir = list(set(self.gc.sticky_actions).intersection(self.dir_actions))
        if len(desired_dir) == 1:
            desired_dir = desired_dir[0]
            which = np.argmax([desired_dir == d for d in self.dir_actions])
            on_dir = utils.cosine_sim(self.gc.controlled_player.direction, self.dir_xy[which]) > 0.65
            if not on_dir and Action.Sprint in self.gc.sticky_actions:
                return Action.ReleaseSprint
            if on_dir and Action.Sprint not in self.gc.sticky_actions:
                return Action.Sprint
        return None

    def _calc_ball_future(self):
        if self.gc.neutral_ball:
            ball_speed = self.gc.get_ball_speed()
            ball_future = self.gc.ball[-1]

            ball_z = self.gc.ball_height[-1]
            z_speed = 0
            if len(self.gc.ball_height) > 1:
                z_speed = self.gc.ball_height[-1] - self.gc.ball_height[-2]

            for t in range(1, 21):
                ball_future = self.gc.ball[-1] + t*ball_speed
                if self.gc.ball_height[-1] < 0.5:
                    ball_speed = 0.95*ball_speed

                dist = utils.distance(ball_future, self.gc.controlled_player.pos)
                closest_opp_dist, opp = self._get_closest(ball_future, OPP_TEAM)

                if ball_future[0] < 0 and ball_z < 1.0 and closest_opp_dist < 0.04 < dist:
                    # opponent intercepts
                    ball_future = self.own_penalty
                    break

                if (dist / t < 0.01) and (ball_z + z_speed*t - 0.1*t*t <= 0):
                    break

        else:
            ball_speed = self.gc.get_ball_speed()
            ball_future = self.gc.ball[-1] + 3*ball_speed

            direction = self.own_goal - ball_future
            direction /= utils.length(direction)
            for t in range(4, 21):
                ball_future = ball_future + 0.008*direction
                dist = utils.distance(ball_future, self.gc.controlled_player.pos)
                if dist / t < 0.01:
                    break

        return ball_future

    def _find_best_pass_option(self):
        players = self.gc.players[OWN_TEAM]

        best_index = None
        best_opp_dist = np.inf
        for i, p in enumerate(players):
            dist_to_goal = utils.distance(p.pos, self.opp_goal)

            valid = True
            if p.offside and i != self.gc.current_obs['active']:
                valid = False
            else:
                for opp in self.gc.players[OPP_TEAM]:
                    dist = utils.distance(p.pos, opp.pos)
                    between = utils.cosine_sim(opp.pos - p.pos, self.opp_goal - opp.pos)

                    if dist < 0.1 or between > 0.7:
                        valid = False
                        break

            if valid and dist_to_goal < best_opp_dist:
                best_index = i

        if best_index is not None and best_index != self.gc.current_obs['active']:
            return players[best_index]

        return None

    def defend(self):
        if self.gc.controlled_player.role == PlayerRole.GoalKeeper:
            direction = self._run_towards(self.gc.ball[-1])
            if Action.Sprint in self.gc.sticky_actions:
                return Action.ReleaseSprint
            return direction
        elif self.gc.own_gk_ball:
            return self._run_towards(np.array([-0.4, 0.0]), c=2)

        if self.gc.neutral_ball:
            dist1 = utils.distance(self.gc.controlled_player.pos, self.opp_goal)
            looking_towards = utils.cosine_sim(self.opp_goal - self.gc.controlled_player.pos,
                                               self.gc.controlled_player.direction) > 0.7
            enable_shot = dist1 < 0.25 and looking_towards
            dist2 = utils.distance(self.gc.controlled_player.pos, self.own_goal)
            closest_opp_dist, _ = self._get_closest(self.gc.controlled_player.pos, OPP_TEAM)
            enable_shot = enable_shot or (dist2 < 0.3 and closest_opp_dist < 0.15)
            if enable_shot and self.action_counter[Action.Shot] > 9:
                return Action.Shot

        ball_coming = False
        ball_dist_now = utils.distance(self.gc.controlled_player.pos, self.gc.ball[-1])
        if len(self.gc.ball) > 1:
            ball_dist_prev = utils.distance(self.gc.controlled_player.pos, self.gc.ball[-2])
            ball_coming = ball_dist_now < ball_dist_prev < 0.1

        if ball_coming and self.gc.neutral_ball:
            if Action.Sprint in self.gc.sticky_actions:
                return Action.ReleaseSprint
        elif Action.Sprint not in self.gc.sticky_actions:
            return Action.Sprint

        ball_future = self._calc_ball_future()
        dir_action = self._run_towards(ball_future)

        if not self.gc.neutral_ball:
            direction = self._tactic_foul()
            if direction:
                return self.macro_list.add_macro([direction, Action.Slide], True)

        return dir_action

    def attack(self):
        if self.gc.controlled_player.role == PlayerRole.GoalKeeper:
            self.dir_cache.register(Action.Right)
            return self.macro_list.add_macro([Action.Right, Action.HighPass], True)

        if self.gc.controlled_player.pos[0] > 0.75:
            direction = None
            action = Action.ShortPass
            if self.gc.controlled_player.pos[1] > 0.15:
                direction = Action.Top
            elif self.gc.controlled_player.pos[1] < -0.15:
                direction = Action.Bottom
            if abs(self.gc.controlled_player.pos[1]) > 0.25:
                action = Action.HighPass

            if direction is not None and self.action_counter[action] > 29:
                self.dir_cache.register(direction)
                return self.macro_list.add_macro([Action.ReleaseSprint, direction, action], True)

        opp_gk = self._get_opponent_by_role(PlayerRole.GoalKeeper)
        dist_to_goal = utils.distance(self.gc.controlled_player.pos, self.opp_goal)
        dist_to_gk = utils.distance(self.gc.controlled_player.pos, opp_gk)
        looking_towards_goal = utils.cosine_sim(self.opp_goal - self.gc.controlled_player.pos,
                                                self.gc.controlled_player.direction) > 0.7
        if ((dist_to_goal < 0.3) or (dist_to_gk < 0.3)) and looking_towards_goal:
            if self.action_counter[Action.Shot] > 9:
                last_move = Action.Right
                if Action.Right in self.gc.sticky_actions:
                    last_move = Action.BottomRight
                    if self.gc.controlled_player.pos[1] > 0:
                        last_move = Action.TopRight
                self.dir_cache.register(last_move)
                return self.macro_list.add_macro([Action.ReleaseSprint, last_move] + [Action.Shot]*3, True)

        sprint_action = self._decide_sprint()
        if sprint_action is not None:
            return sprint_action

        clear_action = self._decide_clear_ball()
        if clear_action is not None:
            return clear_action

        looking_own_goal = utils.cosine_sim(self.own_goal - self.gc.controlled_player.pos,
                                            self.gc.controlled_player.direction) > 0.5

        if not looking_own_goal:
            best_option = self._find_best_pass_option()
            if best_option is not None:
                direction = self._run_towards(best_option.pos + np.array([0.1, 0.0]), c=0.5)
                action = Action.LongPass
                if self.gc.controlled_player.pos[0] < -0.2:
                    action = Action.HighPass
                if self._check_action_counter(9):
                    self.dir_cache.register(direction)
                    return self.macro_list.add_macro([direction, action], True)

            forward_teammates = [player for player in self.gc.players[OWN_TEAM]
                                 if utils.distance(player.pos, self.opp_goal) < dist_to_goal]
            offside = any([p.offside for p in forward_teammates])
            forward_teammates = [p for p in forward_teammates if not p.offside]
            if self._detect_obstacle(self.gc.controlled_player) and len(forward_teammates) > 0 and not offside:
                action = Action.LongPass
                if self.gc.controlled_player.pos[0] < -0.2:
                    action = Action.HighPass
                if self._check_action_counter(9):
                    self.dir_cache.register(Action.Right)
                    return self.macro_list.add_macro([Action.Right, action], True)

        if dist_to_goal > 0.4:
            return self._run_towards(self.opponent_penalty, c=max(0.8, dist_to_goal))
        elif dist_to_goal > 0.2:
            return self._run_towards(self.opp_goal, c=0.5)
        return self._run_towards(self.opp_goal)

    def game_mode_act(self):
        if self.gc.current_obs["game_mode"] == GameMode.Penalty:
            which = int(np.random.randint(3))
            directions = [Action.Right, Action.TopRight, Action.TopLeft]
            return self.macro_list.add_macro([directions[which], Action.Shot], False)

        if self.gc.current_obs["game_mode"] == GameMode.Corner:
            if self.gc.ball[-1][0] > 0:
                direction = Action.BottomRight
                if self.gc.ball[-1][1] > 0:
                    direction = Action.TopRight
                if self.action_counter[Action.HighPass] > 2:
                    return Action.HighPass
                else:
                    return direction
            else:
                return Action.Shot

        if self.gc.current_obs["game_mode"] == GameMode.ThrowIn:
            return Action.Right

        if self.gc.current_obs["game_mode"] == GameMode.GoalKick:
            if self.gc.ball[-1][0] < 0:
                if self.action_counter[Action.ShortPass] > 9:
                    return Action.ShortPass
                else:
                    return Action.Right
            else:
                return self._run_towards(np.array([0.5, 0.1]))

        if self.gc.current_obs["game_mode"] == GameMode.FreeKick:
            direction = self._run_towards(self.opp_goal)
            action = Action.HighPass
            if utils.distance(self.gc.controlled_player.pos, self.opp_goal) < 0.4:
                action = Action.Shot
            if self.action_counter[action] > 9:
                return action
            else:
                return direction

    def act(self, obs):
        self.gc.update(obs)

        action = self.macro_list.step()
        if action is None:
            action = self.dir_cache.step()
        if action is None:
            action = self.game_mode_act()

        if action is None:
            if self.gc.attacking[-1]:
                action = self.attack()
            else:
                action = self.defend()

        for k in self.action_counter.keys():
            self.action_counter[k] += 1
        self.action_counter[action] = 0
        return action