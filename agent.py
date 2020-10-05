from kaggle_environments.envs.football.helpers import Action
from game_cache import GameCache
import utils


class Agent:
    def __init__(self):
        self.gc = GameCache()
        self.just_got_the_ball = 0
        self.opponent_goal = [1.0, 0]
        self.dir_actions = [Action.Right, Action.BottomRight, Action.Bottom, Action.BottomLeft,
                            Action.Left, Action.TopLeft, Action.Top, Action.TopRight]

    def _run_towards(self, source, target):
        which_dir = int(((utils.angle([source[0], source[1]], [target[0], target[1]]) + 22.5) % 360) // 45)
        return self.dir_actions[which_dir]

    def defend(self, obs):
        self.just_got_the_ball = 0
        controlled_player_pos = obs['left_team'][obs['active']]
        if Action.Sprint not in obs['sticky_actions']:
            return Action.Sprint

        return self._run_towards(controlled_player_pos, obs['ball'])

    def attack(self, obs):
        controlled_player_pos = obs['left_team'][obs['active']]
        # Shot if we are 'close' to the goal (based on 'x' coordinate).

        dist_to_goal = utils.distance(controlled_player_pos, self.opponent_goal)
        if dist_to_goal < 0.5:
            return Action.Shot
        # Run towards the goal otherwise.
        self.just_got_the_ball += 1
        if self.just_got_the_ball == 5:
            if controlled_player_pos[0] < -0.2:
                return Action.HighPass

        if self.just_got_the_ball > 1:
            if Action.Sprint not in obs['sticky_actions']:
                return Action.Sprint

        return self._run_towards(controlled_player_pos, self.opponent_goal)
