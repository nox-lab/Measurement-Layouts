"""Demands class which takes into account reward dsitance, reward size and orientation of agent and relative x-position of reward to agent."""
class Demands():
    def __init__(self, reward_size, reward_distance, reward_behind, Xpos):
        self.reward_size = reward_size
        self.reward_distance = reward_distance
        self.reward_behind = reward_behind
        self.Xpos = Xpos
    def __repr__(self) -> str:
        return f"Reward size: {self.reward_size}, Reward distance: {self.reward_distance}, Reward behind: {self.reward_behind}, Xpos: {self.Xpos}"