
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT, SIMPLE_MOVEMENT

from gymnasium import spaces
import numpy as np
import cv2

from typing import Literal

class Tetris():
    def __init__(self,
                 id : str = 'TetrisA-v0',
                 state_format : Literal['image', 'features'] = 'features',
                 action_type: Literal["simple", "full"] = "full",
                 max_steps : int = None):
        super().__init__()

        print(id, state_format, action_type, max_steps)

        env = gym_tetris.make(id)
        self.actions = SIMPLE_MOVEMENT if action_type == 'simple' else MOVEMENT
        self.env = JoypadSpace(env, self.actions)

        self.width = 10
        self.height = 20

        self.state_format = state_format
        if state_format == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(20, 10, 1), dtype=np.uint8)
        elif self.state_format == "features":
            self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)

        self.action_space = self.env.action_space

        self.max_steps = max_steps
        self.num_steps = 0

    def reset(self):
        self.num_steps = 0
        obs = self.env.reset()
        return self._process_obs(obs, rows_cleared=0)

    def _process_obs(self, obs, rows_cleared):

        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        obs = obs[47:207, 95:175]
        obs = cv2.resize(obs, (self.width, self.height))
        obs = (obs > 0).astype(obs.dtype)

        if self.state_format == "image":
            return np.expand_dims(obs, axis=-1)

        invert_heights = np.where(obs.any(axis=0), np.argmax(obs, axis=0), self.height)
        col_heights = self.height - invert_heights
        total_height = np.sum(col_heights)

        bumpiness = np.sum(np.abs(np.diff(col_heights)))

        holes = 0
        for col in range(self.width):
            col_data = obs[:, col]
            first_block = np.argmax(col_data)
            if col_data[first_block] == 1: 
                holes += np.sum(col_data[first_block + 1:] == 0)

        # rows_cleared = np.sum(np.all(obs == 1, axis=1))

        return np.array([total_height, bumpiness, holes, rows_cleared], dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, info = self.env.step(action)
        if self.max_steps is not None:
            self.num_steps += 1
            truncated = (self.num_steps >= self.max_steps)
        else: truncated = False
        return self._process_obs(obs, info['number_of_lines']), reward, terminated, truncated, info        
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()