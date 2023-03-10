"""
Othello game example with faster othello environment.
Prints board state to stdout with random agents by default.
"""

import re
import sys

sys.path.append("../")

import colorama
import numpy as np
from colorama import Fore, Style

from fights.base import BaseAgent
import faster_othello

class RandomAgent(BaseAgent):
    env_id = ("othello", 0)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def _get_all_actions(self, state: faster_othello.OthelloState):
        actions = []
        for coordinate_x in range(faster_othello.OthelloEnv.board_size):
            for coordinate_y in range(faster_othello.OthelloEnv.board_size):
                action = [coordinate_x, coordinate_y]
                if state.legal_actions[self.agent_id][coordinate_x][coordinate_y]:
                    actions.append(action)
        return actions

    def __call__(self, state: faster_othello.OthelloState) -> faster_othello.OthelloAction:
        actions = self._get_all_actions(state)
        return self._rng.choice(actions)

def fallback_to_ascii(s: str) -> str:
    try:
        s.encode(sys.stdout.encoding)
    except UnicodeEncodeError:
        s = re.sub("[┌┬┐├┼┤└┴┘╋]", "+", re.sub("[─━]", "-", re.sub("[│┃]", "|", s)))
    return s


def colorize_walls(s: str) -> str:
    return s.replace("━", Fore.BLUE + "━" + Style.RESET_ALL).replace(
        "┃", Fore.RED + "┃" + Style.RESET_ALL
    )

def run():
    assert faster_othello.OthelloEnv.env_id == RandomAgent.env_id
    colorama.init()

    state = faster_othello.OthelloEnv().initialize_state()
    agents = [RandomAgent(1), RandomAgent(0)]

    print("\x1b[2J")

    it = 0
    while not state.done:

        print("\x1b[1;1H")
        print(fallback_to_ascii(colorize_walls(str(state))))

        for agent in agents:
            
            if state.need_jump(agent.agent_id): continue

            action = agent(state)
            state = faster_othello.OthelloEnv().step(state, agent.agent_id, action)

            print("\x1b[1;1H")
            print(fallback_to_ascii(colorize_walls(str(state))))

            a = input()

            if state.done:
                print(f"agent {np.argmax(state.reward)} won in {it} iters")
                break

        it += 1

if __name__ == "__main__":
    run()