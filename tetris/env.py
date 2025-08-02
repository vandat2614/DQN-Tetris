import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from .game import Game
from .colors import Colors

# Action constants
# NOOP   = 0
LEFT   = 0
RIGHT  = 1
DOWN   = 2
ROTATE = 3

ACTIONS = {
    # NOOP:   "NOOP",
    LEFT:   "LEFT",
    RIGHT:  "RIGHT",
    DOWN:   "DOWN",
    ROTATE: "ROTATE"
}

class TetrisEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, state_format='discrete', render_mode=None, sound=False):
        super().__init__()

        assert state_format in ["discrete", "image"], "Invalid state_format"
        self.state_format = state_format
        self.render_mode = render_mode

        pygame.init()

        self.game = Game(sound=sound)
        self.window = None
        self.clock = None

        # Fonts and surfaces for GUI
        self.title_font = pygame.font.Font(None, 40)
        self.score_surface = self.title_font.render("Score", True, Colors.white)
        self.next_surface = self.title_font.render("Next", True, Colors.white)
        self.game_over_surface = self.title_font.render("GAME OVER", True, Colors.white)

        # Rectangles for GUI areas
        self.score_rect = pygame.Rect(320, 55, 170, 60)
        self.next_rect = pygame.Rect(320, 215, 170, 180)
        self.game_area = pygame.Rect(11, 11, 10 * 30, 20 * 30)  # x, y, w, h

        self.width = 10
        self.height = 20

        if self.state_format == "discrete":
            self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.int32)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.game_area.height, self.game_area.width, 3), dtype=np.uint8)

        self.action_space = spaces.Discrete(len(ACTIONS))

        self.gravity_timer = 0
        self.gravity_interval = 200  # milliseconds

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        observation = self._get_obs(lines_cleared=0)
        return observation, {}

    def step(self, action):
        assert self.action_space.contains(action)

        lines_cleared = 0
        success = True
        reward = 0.1

        if not self.game.game_over:
            if action == LEFT:
                success = self.game.move_left()
            elif action == RIGHT:
                success = self.game.move_right()
            elif action == DOWN:
                lines_cleared = self.game.move_down()
                self.game.update_score(0, 1)
            elif action == ROTATE:
                reward -= 1
                success = self.game.rotate()

        observation = self._get_obs(lines_cleared)
        
        reward += [0, 2, 5, 10, 12][lines_cleared]
        if self.game.game_over:
            reward -= 10

        if not success:
            reward -= 5

        terminated = self.game.game_over
        truncated = False
        info = {"lines_cleared": lines_cleared, 'score' : self.game.score}

        return observation, reward, terminated, truncated, info

    def _get_obs(self, lines_cleared):

        board = self.game.get_board_state()
        obs = (board > 0).astype(board.dtype)

        if self.state_format == 'image':
            return obs
        
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

        return np.array([total_height, bumpiness, holes, lines_cleared], dtype=np.float32)

    def render(self):
        self._render_frame()
        if self.render_mode == "rgb_array":
            full_image = pygame.surfarray.array3d(self.window).swapaxes(0, 1)
            x, y, w, h = self.game_area
            return full_image[y:y+h, x:x+w, :]

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((500, 620))
            pygame.display.set_caption("Python Tetris")
            self.clock = pygame.time.Clock()

        self.window.fill(Colors.dark_blue)

        score_value_surface = self.title_font.render(str(self.game.score), True, Colors.white)

        self.window.blit(self.score_surface, (365, 20))
        self.window.blit(self.next_surface, (375, 180))

        if self.game.game_over:
            self.window.blit(self.game_over_surface, (320, 450))

        pygame.draw.rect(self.window, Colors.light_blue, self.score_rect, border_radius=10)
        self.window.blit(score_value_surface, score_value_surface.get_rect(center=self.score_rect.center))

        pygame.draw.rect(self.window, Colors.light_blue, self.next_rect, border_radius=10)

        self.game.draw(self.window)

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None
