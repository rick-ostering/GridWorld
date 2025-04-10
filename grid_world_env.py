
import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class GridWorldEnv( gym.Env ):
	metadata = { "render_modes": ["human", "rgb_array"], "render_fps": 5000 }

	def __init__( self, render_mode=None, grid_size = 8 ):
		super().__init__()
		self.grid_size = grid_size  # The size of the square grid
		self.window_size = 800  # The size of the PyGame window

		self.run = True
		self.steps_taken = 0
		self.steps_to_food = 0
		self.food_eaten = 0
		
		self.reset_locs()
		
		# Observations are dictionaries with the agent's and the target's location.
		# Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
		self.observation_space = spaces.Dict(
			{
				"agent": spaces.Box( 0, grid_size - 1, shape=(2,), dtype=int ),
				"target": spaces.Box( 0, grid_size - 1, shape=(2,), dtype=int ),
			}
		)

		# We have 2 actions, corresponding to "left", "right", "up", "down"
		self.action_space = spaces.Discrete( 4 )

		assert render_mode is None or render_mode in self.metadata["render_modes"]
		self.render_mode = render_mode

		"""
		If human-rendering is used, `self.window` will be a reference
		to the window that we draw to. `self.clock` will be a clock that is used
		to ensure that the environment is rendered at the correct framerate in
		human-mode. They will remain `None` until human-mode is used for the
		first time.
		"""
		
		self.window = None
		self.clock = None
		self.frame_number = 0
	

	def _get_obs(self):
		return { "agent": self._agent_location, "target": self._target_location }

	def _get_info(self):
		return {
			"distance": np.linalg.norm(
				self._agent_location - self._target_location, ord=1
			),
			"steps_taken": self.steps_taken,
			"steps_to_food": self.steps_to_food,
			"food_eaten": self.food_eaten
		}

	def reset( self, seed = None, options = None ):
		# We need the following line to seed self.np_random
		super().reset( seed = seed )

		self.steps_taken = 0
		self.food_eaten = 0
		self.steps_to_food = 0

		self.reset_locs()

		observation = self._get_obs()
		info = self._get_info()

		if self.render_mode == "human":
			self._render_frame()

		return observation, info

	def reward_func( self, goal_reached , ap_goal ):
		reward = 0
		if self.food_eaten < 8:
			reward = ap_goal
		if goal_reached:
			reward = 1 + self.grid_size / self.steps_to_food
		return reward
	
	def reset_locs( self ):
		self._agent_location = self.np_random.integers( 0, self.grid_size , size = 2, dtype=int  )
		self._target_location = self._agent_location
		while np.array_equal(self._target_location, self._agent_location):
			self._target_location = self.np_random.integers(
				0, self.grid_size, size=2, dtype=int
			)

	def is_approaching_goal( self, prev, new ):
		prev_dist = np.linalg.norm( prev - self._target_location , ord=1 )
		new_dist = np.linalg.norm( new - self._target_location , ord=1 )
		if new_dist < prev_dist:
			return 1
		else:
			return -1

	def step( self, action ):
		dir = [ 0,0 ]
		if action == 0:		# right
			dir = [ 1, 0 ]
		elif action == 1:	# down
			dir = [ 0, 1 ]
		elif action == 2:	# left
			dir = [ -1,0 ]
		elif action == 3:	# up
			dir = [ 0, -1 ]

		self.steps_taken += 1
		self.steps_to_food += 1

		# We use `np.clip` to make sure we don't leave the grid

		prev_agent_location = self._agent_location.copy()
		self._agent_location = np.clip( self._agent_location + dir, 0, self.grid_size - 1 )
		new_agent_location = self._agent_location.copy()

		approaching = self.is_approaching_goal( prev_agent_location, new_agent_location )

		goal_reached = np.array_equal( self._agent_location, self._target_location )

		reward = self.reward_func( goal_reached , approaching )

		if goal_reached:
			self.food_eaten += 1
			self.steps_to_food = 0
			self.reset_locs()

		terminated = False
		truncated = False
		if self.food_eaten > 20:
			reward += 3
			terminated = True
		if self.steps_to_food > ( self.grid_size * 3 ):
			truncated = True

		observation = self._get_obs()
		info = self._get_info()

		if self.render_mode == "human":
			self._render_frame()


		return observation, reward, terminated, truncated, info


	def render(self):
		if self.render_mode == "rgb_array":
			return self._render_frame()

	def _render_frame( self ):
		
		pix_square_size = ( self.window_size / self.grid_size )  # The size of a single grid square in pixels

		if self.window is None and self.render_mode == "human":
			pygame.init()
			pygame.display.init()
			self.window = pygame.display.set_mode(
				( self.window_size, self.window_size )
			)
		if self.clock is None and self.render_mode == "human":
			self.clock = pygame.time.Clock()

		self.handle_events()

		canvas = pygame.Surface( ( self.window_size, self.window_size ) )

		bg_col = ( 10, 10, 10 )
		canvas.fill( bg_col )
		
		# First we draw the target
		target_top_left = pix_square_size * self._target_location
		target_wh = ( pix_square_size, pix_square_size )
		target_rect = pygame.Rect( target_top_left, target_wh )
		
		pygame.draw.rect(
			canvas,
			(255, 0, 0),
			target_rect
		)
		# Now we draw the agent

		pygame.draw.circle(
			canvas,
			(0, 0, 255),
			(self._agent_location + 0.5) * pix_square_size,
			pix_square_size / 3,
		)

		# Finally, add some gridlines
		for x in range( self.grid_size + 1 ):
			pygame.draw.line(
				canvas,
				( 90,90,90 ),
				(0, pix_square_size * x),
				(self.window_size, pix_square_size * x),
				width=1,
			)
			pygame.draw.line(
				canvas,
				( 90,90,90 ),
				(pix_square_size * x, 0),
				(pix_square_size * x, self.window_size),
				width=1,
			)

		if self.render_mode == "human":
			# The following line copies our drawings from `canvas` to the visible window
			self.window.blit(canvas, canvas.get_rect())
			pygame.event.pump()
			pygame.display.update()

			# We need to ensure that human-rendering occurs at the predefined framerate.
			# The following line will automatically add a delay to keep the framerate stable.
			self.clock.tick( self.metadata["render_fps"] )
					
		else:  # rgb_array
			return np.transpose(
				np.array( pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
			)
		
		# folder = "C:/docs/projs/python/RL/SB3_tests/grid_world_v02/out_render"
		# file_name = "out_render." + str( self.frame_number ) + ".png"
		# full_path = folder + "/" + file_name
		# pygame.image.save( canvas, full_path )
		# self.frame_number += 1


	def close(self):
		if self.window is not None:
			pygame.display.quit()
			pygame.quit()
			
	def handle_events( self ):
		events = pygame.event.get()
		for event in events:
			if event.type == pygame.QUIT:
				self.run = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					self.run = False


