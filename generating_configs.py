from typing import Callable
from demands import Demands
import numpy as np


def gen_config_from_demands(
    reward_size: float, reward_distance: float, reward_behind: float, x_pos: float, time_limit: float, env_number : int, filename: str, numbered: bool = False
) -> str:
    # Validate
    try:
      assert reward_size >= 0 and reward_size <= 1.9
    except:
      print("size greater than intended range.")
      print(reward_size)
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
      if x_pos == 0:
        agent_rotation = np.random.choice([-90, 90])
      else:
        agent_rotation = 0
    else:
      agent_rotation = 90*x_pos + 180
    agent_rotation = agent_rotation % 360 # This stops negative values.
    final_part = ""
    
    initial_part = """
    !ArenaConfig
    arenas:"""
    generated_config = f"""
      {env_number}: !Arena
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
          skins:
          - "panda"
          positions:
          - !Vector3 {{x: 20, y: 0, z: 20}}
          rotations: [{agent_rotation}]
        """
    if numbered:
      distance_away = 15
      signboard_x = 20+distance_away*np.sin(agent_rotation*np.pi/180)
      signboard_z = 20+distance_away*np.cos(agent_rotation*np.pi/180)
      symbolpattern = ["1"*(env_number//3+2) if i % 2 == 0 else "0"*(env_number//3+2) for i in range(env_number//3+3)]
      symbolpattern = "/".join(symbolpattern)
      final_part = f"""
        - !Item
          name: SignBoard
          positions:
          - !Vector3 {{x: {signboard_x}, y: 0, z: {signboard_z}}}
          sizes:
          - !Vector3 {{x: 2, y: 2, z: 2}}
          rotations: [{agent_rotation+270}]
          symbolNames:
          - "{symbolpattern}"
      """
          
          
      
    with open(filename, "w") as text_file:
      text_file.write(initial_part + generated_config + final_part)
    return generated_config + final_part
  
def gen_config_from_demands_batch(envs : list[Demands], filename: str, time_limit: int = 100, numbered: int = False) -> str:
  number_of_initial_envs = 0
  new_conf = ""
  initial_part = """
    !ArenaConfig
    arenas:"""
  # THIS INITIAL ENVIRONEMNT NEVER GETS USED, IT WILL GET SKIPPED. WE will make the evaluation run an extra time, to ensure that there is no loss in sync.
  for i in range(number_of_initial_envs):
    env_conf = gen_config_from_demands(5, 10, 0, 0, time_limit, i, f"temp", numbered=False)
    new_conf += env_conf
  for i, env in enumerate(envs):
    for j in range(3):
      env_conf = gen_config_from_demands(env.reward_size, env.reward_distance, env.reward_behind, env.Xpos, time_limit, (number_of_initial_envs)+j+3*i, f"temp", numbered=numbered)
      new_conf += env_conf
  with open(filename, "w") as text_file:
    text_file.write(initial_part + new_conf)
  return initial_part + new_conf

  
def gen_config_from_demands_batch_random(n_envs: int, filename: str, size_min = 0, size_max = 1.9, dist_min = None, dist_max = 5.3, time_limit = 100, numbered = False) -> tuple[str, list[Demands]]:
  demands_list = []
  for i in range(n_envs):
    dist_min = size_min + 0.5 if dist_min is None else dist_min
    if dist_min < size_min + 0.5:
      dist_min = size_min + 0.5
      print("Distance min too small, setting to size_min + 0.5, this is to prevent clipping.")
    size = np.random.uniform(size_min, size_max) # This should prevent clipping
    demands_list.append(Demands(size, np.random.uniform(dist_min,dist_max), np.random.choice([0, 0.5, 1]), np.random.choice([-1, 0, 1])))
  return gen_config_from_demands_batch(demands_list, filename, time_limit, numbered = numbered), demands_list

