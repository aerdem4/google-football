class Macro:
    def __init__(self, actions, check_sticky):
        self.actions = actions
        self.check_sticky = check_sticky

    def step(self, current_sticky_actions):
        while len(self.actions) > 0:
            action = self.actions.pop(0)
            if self.check_sticky and action not in current_sticky_actions:
                return action
            elif not self.check_sticky:
                return action
        return None


class MacroList:
    def __init__(self, game_cache):
        self.macros = []
        self.gc = game_cache

    def step(self):
        if len(self.macros) == 0:
            return None
        action = self.macros[0].step(self.gc.current_obs["sticky_actions"])
        if action is None:
            del self.macros[0]
        return action

    def add_macro(self, actions, check_sticky):
        self.macros.append(Macro(list(actions), check_sticky))
        return self.step()
