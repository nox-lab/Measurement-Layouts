from typing import Callable

import numpy as np

gen_config_example_deleteme: Callable[[str], str] = (
    lambda object_name: f"""
!ArenaConfig
arenas:
  0: !Arena
    timeLimit: 5
    items:
    - !Item
      name: {object_name}
      positions:
      - !Vector3 {{x: 20, y: 0, z: 20}}
      rotations: [0]
      sizes:
      - !Vector3 {{x: 1, y: 1, z: 1}}
    - !Item
      name: Agent
      positions:
      - !Vector3 {{x: 30, y: 0, z: 20}}
      rotations: [270]
"""
)


def gen_config_from_demands(
    reward_size: float, reward_distance: float, reward_behind: float, x_pos: float
) -> str:
    # Validate
    assert reward_size >= 0 and reward_size <= 1.9
    assert reward_distance >= 0 and reward_distance <= 5.3
    assert reward_behind in [0, 0.5, 1]
    assert x_pos in [-1, 0, 1]

    goal_x_pos, goal_z_pos = 20, 20
    
    angle_from_centre = np.arcsin(x_pos)
    goal_z_pos += reward_distance * np.cos(angle_from_centre)
    goal_x_pos += reward_distance * np.sin(angle_from_centre)
    

    return f"""
!ArenaConfig
arenas:
  0: !Arena
    timeLimit: 5
    items:
    - !Item
      name: GoodGoal
      positions:
      - !Vector3 {{x: {goal_x_pos}, y: 0, z: {goal_z_pos}}}
      rotations: [0]
      sizes:
      - !Vector3 {{x: {reward_size}, y: {reward_size}, z: {reward_size}}}
    - !Item
      name: Agent
      positions:
      - !Vector3 {{x: 20, y: 0, z: 20}}
      rotations: [{reward_behind* 180}]
"""


testval = gen_config_from_demands(1, 5, 0, 1)
with open("gentest.yaml", "w") as text_file:
    text_file.write(testval)