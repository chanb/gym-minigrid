from gym_minigrid.minigrid import Ball
from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.register import register

import numpy as np

class UnlockState(RoomGrid):
    """
    Unlock a door
    """

    def __init__(self, seed=None):
        room_size = 6
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8*room_size**2,
            seed=seed
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.key, _ = self.add_object(0, 0, 'key', door.color)

        self.place_agent(0, 0)

        self.door = door
        self.mission = "open the door"

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if self.door.is_open:
            reward = 1

        return obs, reward, done, info

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        obs = np.array([
            *self.agent_pos,
            self.agent_dir,
            *self.key.cur_pos,
            *self.door.cur_pos,
            int(self.door.is_open),
        ])

        return obs

    def get_task_successes(self, tasks, observation, action, env_info):
        door_opened = int(observation[..., 7] > 0)
        key_obtained = int(observation[..., [3, 4]].sum() < 0)
        return [door_opened, key_obtained, door_opened, action in (3, 5)]

register(
    id='MiniGrid-Unlock-State-v0',
    entry_point='gym_minigrid.envs:UnlockState'
)
