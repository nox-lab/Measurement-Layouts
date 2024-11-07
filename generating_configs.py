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
    reward_size: float, reward_distance: float, reward_behind: float, x_pos: float, time_limit: float, filename: str,
) -> str:
    # Validate
    assert reward_size >= 0 and reward_size <= 1.9
    try:
      assert reward_distance >= 0 and reward_distance <= 5.3
    except:
      print("distance greater than intended range.")
      print(reward_distance)
    assert reward_behind in [0, 0.5, 1]
    assert x_pos in [-1, 0, 1]

    goal_x_pos, goal_z_pos = 20, 20
    
    goal_x_pos += reward_distance*x_pos
    goal_z_pos += reward_distance*(1-abs(x_pos))
    
    # rotation of agnet = f(reward_behind, x_pos)
    
    if reward_behind == 0:
      agent_rotation = 90*x_pos
    elif reward_behind == 0.5:
      agent_rotation = 90*x_pos + 90
    else:
      agent_rotation = 90*x_pos + 180
    generated_config = f"""
    !ArenaConfig
    arenas:
      0: !Arena
        timeLimit: {time_limit}
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
          rotations: [{agent_rotation}]
    """
    with open(filename, "w") as text_file:
      text_file.write(generated_config)
    return generated_config