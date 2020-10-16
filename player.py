import numpy as np


class Player:
    def __init__(self, role):
        self.role = role
        self.active = False
        self.ball_owned = False
        self.pos = None
        self.yellow = None
        self.direction = None
        self.tired = None
        self.offside = None


def get_player_obs(obs):
    players = dict()
    for team_index, team_name in enumerate(["left_team", "right_team"]):
        ball_owned_team = obs['ball_owned_team'] == team_index
        players[team_name] = []

        for i in range(len(obs[team_name])):
            player = Player(obs[f"{team_name}_roles"][i])
            player.ball_owned = (i == obs["ball_owned_player"]) and ball_owned_team
            if team_name == "left_team" and obs["active"] == i:
                player.active = True
            player.pos = np.array(obs[f"{team_name}"][i])
            player.yellow = obs[f"{team_name}_yellow_card"][i]
            player.direction = np.array(obs[f"{team_name}_direction"][i])
            player.tired = obs[f"{team_name}_tired_factor"][i]

            if team_name == "left_team":
                player.offside = player.pos[0] > obs["ball"][0]
                if player.offside:
                    player.offside = sum([player.pos[0] < opp[0] for opp in obs["right_team"]]) < 2
            else:
                player.offside = player.pos[0] < obs["ball"][0]
                if player.offside:
                    player.offside = sum([player.pos[0] > opp[0] for opp in obs["left_team"]]) < 2

            players[team_name].append(player)

    return players
