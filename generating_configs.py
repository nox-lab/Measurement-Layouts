from typing import Callable
from demands import Demands
import numpy as np


def gen_config_from_demands(
    reward_size: float, reward_distance: float, reward_behind: float, x_pos: float, time_limit: float, env_number : int, filename: str,
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
      if x_pos == 0:
        agent_rotation = np.random.choice([-90, 90])
      else:
        agent_rotation = 0
    else:
      agent_rotation = 90*x_pos + 180
    agent_rotation = agent_rotation % 360 # This stops negative values.
      
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
          positions:
          - !Vector3 {{x: 20, y: 0, z: 20}}
          rotations: [{agent_rotation}]
        """
    with open(filename, "w") as text_file:
      text_file.write(initial_part + generated_config)
    return generated_config
  
def gen_config_from_demands_batch(envs : list[Demands], filename: str, time_limit: int = 100) -> str:
  new_conf = ""
  initial_part = """
    !ArenaConfig
    arenas:"""
  for i, env in enumerate(envs):
    env_conf = gen_config_from_demands(env.reward_size, env.reward_distance, env.reward_behind, env.Xpos, time_limit, i, f"temp")
    new_conf += env_conf
  with open(filename, "w") as text_file:
    text_file.write(initial_part + new_conf)
  return new_conf
np.random.seed(0)

  
def gen_config_from_demands_batch_random(n_envs: int, filename: str, size_max = 1.9, dist_max = 5.3, time_limit = 100) -> tuple[str, list[Demands]]:
  demands_list = []
  for i in range(n_envs):
    size = np.random.uniform(0, size_max) # This should prevent clipping
    demands_list.append(Demands(size, np.random.uniform(size+0.5,dist_max), np.random.choice([0, 0.5, 1]), np.random.choice([-1, 0, 1])))
  return gen_config_from_demands_batch(demands_list, filename, time_limit), demands_list

gen_config_from_demands_batch_random(10, r"example_batch_eval.yaml", time_limit=25)