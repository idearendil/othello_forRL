"""
Othello Environment Speed Test
"""

import numpy as np
import time

from fights.base import BaseAgent
import faster_othello
import fastest_othello
import othello

class RandomAgent(BaseAgent):
    env_id = ("othello", 0)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def _get_all_actions(self, state: othello.OthelloState):
        actions = []
        for coordinate_x in range(othello.OthelloEnv.board_size):
            for coordinate_y in range(othello.OthelloEnv.board_size):
                action = [coordinate_x, coordinate_y]
                if state.legal_actions[self.agent_id][coordinate_x][coordinate_y]:
                    actions.append(action)
        return actions

    def __call__(self, state: othello.OthelloState) -> othello.OthelloAction:
        actions = self._get_all_actions(state)
        return self._rng.choice(actions)

class Faster_RandomAgent(BaseAgent):
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

class Fastest_RandomAgent(BaseAgent):
    env_id = ("othello", 0)  # type: ignore

    def __init__(self, agent_id: int, seed: int = 0) -> None:
        self.agent_id = agent_id  # type: ignore
        self._rng = np.random.default_rng(seed)

    def _get_all_actions(self, state: fastest_othello.OthelloState):
        actions = []
        for coordinate_x in range(fastest_othello.OthelloEnv.board_size):
            for coordinate_y in range(fastest_othello.OthelloEnv.board_size):
                action = [coordinate_x, coordinate_y]
                if state.legal_actions[self.agent_id][coordinate_x][coordinate_y]:
                    actions.append(action)
        return actions

    def __call__(self, state: fastest_othello.OthelloState) -> fastest_othello.OthelloAction:
        actions = self._get_all_actions(state)
        return self._rng.choice(actions)

def run_original():
    assert othello.OthelloEnv.env_id == RandomAgent.env_id

    start = time.time()

    for game in range(100):

        state = othello.OthelloEnv().initialize_state()
        agents = [RandomAgent(1, game), RandomAgent(0, game)]

        it = 0
        while not state.done:

            for agent in agents:
                
                if state.need_jump(agent.agent_id): continue

                action = agent(state)
                state = othello.OthelloEnv().step(state, agent.agent_id, action)

                if state.done:
                    break

            it += 1

    end = time.time()

    print(f"{end - start} sec")

def run_faster():
    assert othello.OthelloEnv.env_id == RandomAgent.env_id

    start = time.time()

    for game in range(100):

        state = faster_othello.OthelloEnv().initialize_state()
        agents = [Faster_RandomAgent(1, game), Faster_RandomAgent(0, game)]

        it = 0
        while not state.done:

            for agent in agents:
                
                if state.need_jump(agent.agent_id): continue

                action = agent(state)
                state = faster_othello.OthelloEnv().step(state, agent.agent_id, action)

                if state.done:
                    break

            it += 1

    end = time.time()

    print(f"{end - start} sec")

def run_fastest():
    assert othello.OthelloEnv.env_id == RandomAgent.env_id

    start = time.time()

    for game in range(100):

        state = fastest_othello.OthelloEnv().initialize_state()
        agents = [Fastest_RandomAgent(1, game), Fastest_RandomAgent(0, game)]

        it = 0
        while not state.done:

            for agent in agents:
                
                if state.need_jump(agent.agent_id): continue

                action = agent(state)
                state = fastest_othello.OthelloEnv().step(state, agent.agent_id, action)

                if state.done:
                    break

            it += 1

    end = time.time()

    print(f"{end - start} sec")

if __name__ == "__main__":
    run_original()
    run_faster()
    run_fastest()