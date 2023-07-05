from agent import Agent
from board import Board

class MiniMaxAgent(Agent):

    def next_action(self, obs):
        while True:
            try:
                input_value = int(input())
                return input_value
            except ValueError:
                print("Please insert a number.")

    def heuristic_utility(self, board: Board):
        return 0
    