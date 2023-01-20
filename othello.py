"""
Othello(Reversi) Environment

Coordinates are specified in the form of ''(r, c)'', where ''(0, 0)'' is the top left corner.
All coordinates and directions are absolute and does not change between agents.

Directions
    - Top: '-r'
    - Right: '+c'
    - Bottom: '+r'
    - Left: '-c'
"""

import sys
import numpy as np

from typing import Callable, Optional
from numpy.typing import ArrayLike, NDArray

if sys.version_info < (3, 10):
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias

from fights.base import BaseEnv, BaseState

OthelloAction: TypeAlias = ArrayLike
"""
Alias of :obj:'ArrayLike' to describe the action type.
Encoded as an array of shape ''(2,)'',
in the form of [ 'coordinate_r', 'coordinate_c' ].
"""

@dataclass
class OthelloState(BaseState):
    """
    ''OthelloState'' represents the game state.
    """

    board: NDArray[np.int_]
    """
    Array of shape ``(C, W, H)``,
    where C is channel index
    and W, H is board width, height.

    Channels
        - ''C = 0'': one-hot encoded stones of agent 0. (black)
        - ''C = 1'': one-hot encoded stones of agent 1. (white)
    """

    legal_actions: NDArray[np.int_]
    """
    Array of shape ''(C, W, H)'',
    where C is channel index
    and W, H is board width, height.

    Channels
        - ''C = 0'': one-hot encoded possible positions of agent 0. (black)
        - ''C = 1'': one-hot encoded possible positions of agent 1. (white)
    """

    done: bool = False
    """
    Boolean value indicating wheter the game is done.
    """

    def __str__(self) -> str:
        """
        Generate a human-readable string representation of the board.
        Uses unicode box drawing characters.
        """

        table_top = "┌───┬───┬───┬───┬───┬───┬───┬───┬───┐"
        vertical_wall = "│"
        horizontal_wall = "───"
        left_intersection = "├"
        middle_intersection = "┼"
        right_intersection = "┤"
        left_intersection_bottom = "└"
        middle_intersection_bottom = "┴"
        right_intersection_bottom = "┘"
        result = table_top + "\n"

        for y in range(8):
            board_line = self.board[:, :, y]
            result += vertical_wall
            for x in range(8):
                board_cell = board_line[:, x]
                if board_cell[0]:
                    result += " 0 "
                elif board_cell[1]:
                    result += " 1 "
                else:
                    result += "   "
                if x == 7:
                    result += vertical_wall
                    result += "\n"
                else:
                    result += " "
            result += left_intersection_bottom if y == 7 else left_intersection
            for x in range(8):
                board_cell = board_line[:, x]
                if y == 7:
                    result += horizontal_wall
                    result += (
                        right_intersection_bottom if y == 7
                        else right_intersection
                    )
                else:
                    result += "   "
                    result += (
                        middle_intersection_bottom if y == 8
                        else middle_intersection
                    )

            result += "\n"

        return result

    def perspective(self, agent_id: int) -> NDArray[np.int_]:
        """
        Return board observed by the agent whose ID is agent_id.

        :arg agent_id:
            The ID of agent to use as base.

        :returns:
            The board's channel 0 will contain stones of ''agent_id'',
            and channel 1 will contain stones of opponent.
            Considering that every game starts with 4 stones of fixed position,
            it returns flipped ''board'' array if ''agent_id'' is 1.
        """

        if agent_id == 0:
            return self.board
        
        rotated = np.stack(
            np.fliplr(self.board[1]),
            np.fliplr(self.board[0])
        )

        return rotated

class OthelloEnv(BaseEnv[OthelloState, OthelloAction]):
    env_id = ("othello", 0) # type: ignore
    """
    Environment identifier in the form of ''(name, version)''.
    """

    board_size: int = 8
    """
    Size (width and height) of the board.
    """

    def step(
        self,
        state: OthelloState,
        agent_id: int,
        action: OthelloAction,
        *,
        pre_step_fn: Optional[
            Callable[[OthelloState, int, OthelloAction], None]
        ] = None,
        post_step_fn: Optional[
            Callable[[OthelloState, int, OthelloAction], None]
        ] = None,
    ) -> OthelloState:
        """
        Step through the game,
        calculating the next state given the current state and action to take.

        :arg state:
            Current state of the environment.
        
        :arg action:
            ID of the agent that takes the action. (''0'' or ''1'')

        :arg action:
            Agent action, encoded in the form described by :obj:'OthelloAction'.
        
        :arg pre_step_fn:
            Callback to run before executing action. ``state``, ``agent_id`` and
            ``action`` will be provided as arguments.

        :arg post_step_fn:
            Callback to run after executing action. The calculated state, ``agent_id``
            and ``action`` will be provided as arguments.

        :returns:
            A copy of the object with the restored state.
        """

        if pre_step_fn is not None:
            pre_step_fn(state, agent_id, action)

        action = np.asanyarray(action).astype(np.int_)
        r, c = action

        if not self._check_in_range(np.array([r, c])):
            raise ValueError(f"out of board: {(r, c)}")
        if not 0 <= agent_id <= 1:
            raise ValueError(f"invalid agent_id: {agent_id}")
        if state.board[1-agent_id][r][c]:
            raise ValueError("cannot put a stone on opponent's stone")
        if state.board[agent_id][r][c]:
            raise ValueError("cannot put a stone on another stone")     

        board = np.copy(state.board)

        directions = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]

        flipped = False
        for dir in directions:
            stones_to_flip = []
            temp_r = r
            temp_c = c
            for _ in range(1, self.board_size):
                temp_r += dir[0]
                temp_c += dir[1]
                if state.board[1-agent_id][temp_r][temp_c] == 1:
                    stones_to_flip.append((temp_r, temp_c))
                elif state.board[agent_id][temp_r][temp_c] == 1:
                    if stones_to_flip:
                        flipped = True
                        for a_stone in stones_to_flip:
                            board[1-agent_id][a_stone[0]][a_stone[1]] = 0
                            board[agent_id][a_stone[0]][a_stone[1]] = 1
                    break
                else:
                    break

        if not flipped:
            raise ValueError("There is no stones to flip")
        
        # 여기서부터 이제 legal_actions에 대한 코드 짜야함
        # 사실 위에서 판단하는 것들 모두 여기서 미리 처리하고
        # 위에서는 입력된 r, c가 전 state의 legal_actions에 포함되어 있는지만 확인하면 됨

        


    def _check_in_range(self, pos: NDArray[np.int_], bottom_right=None) -> np.bool_:
        if bottom_right is None:
            bottom_right = np.array([self.board_size, self.board_size])
        return np.all(np.logical_and(np.array([0, 0]) <= pos, pos < bottom_right))













