import numpy as np
from kaggle_environments.envs.football.helpers import GameMode, PlayerRole
from player import get_player_obs
import utils


class GameCache:
    def __init__(self):
        self.time = 0
        self.ball = []
        self.ball_height = []
        self.controlled_player_pos = []
        self.controlled_player = None
        self.attacking = []
        self.time_since_ball = 0
        self.current_obs = None
        self.neutral_ball = True
        self.players = None
        self.sticky_actions = []
        self.time_since_controlling = 0
        self.own_gk_ball = False

    def _get_speed(self, obj_array):
        if len(obj_array) < 2:
            return np.array([0, 0])
        return obj_array[-1] - obj_array[-2]

    def update(self, obs):
        self.current_obs = obs
        self.time += 1
        self.players = get_player_obs(obs)

        self.ball.append(np.array(obs["ball"][:2]))
        self.ball_height.append(np.array(obs["ball"][2]))

        teammates = self.players["left_team"]
        # TODO: check jersey number instead
        self.time_since_controlling += 1
        if self.controlled_player and self.controlled_player.role != teammates[obs['active']].role:
            self.time_since_controlling = 0

        self.controlled_player = teammates[obs['active']]
        self.controlled_player_pos.append(self.controlled_player.pos)

        attacking = (self.ball[-1][0] > 0 and self.current_obs["game_mode"] in
                     {GameMode.FreeKick, GameMode.Penalty, GameMode.Corner, GameMode.ThrowIn})
        attacking = attacking or self.controlled_player.ball_owned

        self.attacking.append(attacking)
        if self.attacking[-1]:
            if self.time_since_ball == 0:  # gained the ball
                self.controlled_player_pos = []
            self.time_since_ball += 1
        else:
            if self.time_since_ball > 0:  # lost the ball
                self.controlled_player_pos = []
            self.time_since_ball = 0

        self.neutral_ball = False
        if obs['ball_owned_team'] == -1:
            self.neutral_ball = True

        if obs["ball_owned_team"] == 0:
            ball_owner = self.players["left_team"][obs["ball_owned_player"]]
            self.own_gk_ball = ((ball_owner.role == PlayerRole.GoalKeeper) and
                                (utils.distance(ball_owner.pos, self.ball[-1]) < 0.02))

        self.sticky_actions = obs["sticky_actions"]

    def get_ball_speed(self):
        return self._get_speed(self.ball)

    def get_player_speed(self):
        return self._get_speed(self.controlled_player_pos)


class DirCache:
    def __init__(self, gc):
        self.active = False
        self.dir = None
        self.time = 0
        self.gc = gc

    def register(self, dir):
        self.dir = dir
        self.time = self.gc.time
        self.active = True

    def step(self):
        if self.gc.controlled_player.ball_owned and self.active and self.gc.time < self.time + 10:
            return self.dir
        else:
            self.active = False
            return None
