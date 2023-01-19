"""
Othello(Reversi) Environment

Coordinates are specified in the form of '(r, c)', where '(0, 0)' is the top left corner.
All coordinates and directions are absolute and does not change between agents.

Directions
    - Top: '-r'
    - Right: '+c'
    - Bottom: '+r'
    - Left: '-c'
"""

import sys
import numpy as np
from numpy.typing import ArrayLike, NDArray

from fights.base import BaseEnv, BaseState