import numpy as np


class GameCache:
    def __init__(self):
        self.time = 0
        self.ball = []
        self.controlled_player = []
        self.attacking = []
        self.time_since_ball = 0

    def _get_speed(self, obj_array):
        if len(obj_array) < 2:
            return np.array([0, 0])
        return obj_array[-1] - obj_array[-2]

    def update(self, obs):
        self.time += 1
        self.ball.append(np.array(obs["ball"][:2]))
        self.attacking.append((obs['ball_owned_player'] == obs['active']) and (obs['ball_owned_team'] == 0))
        if self.attacking[-1]:
            if self.time_since_ball == 0:  # gained the ball
                self.controlled_player = []
            self.time_since_ball += 1
        else:
            if self.time_since_ball > 0:  # lost the ball
                self.controlled_player = []
            self.time_since_ball = 0
        self.controlled_player.append(np.array(obs['left_team'][obs['active']]))

    def get_ball_speed(self):
        return self._get_speed(self.ball)

    def get_player_speed(self):
        return self._get_speed(self.controlled_player)
