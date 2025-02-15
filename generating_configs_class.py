from typing import Callable, List, Union, Iterable
from demands import Demands
import numpy as np

class ConfigGenerator:
    def __init__(self, precise = False):
        self.initial_part = """
        !ArenaConfig
        arenas:"""
        self.precise = precise

    def gen_config_from_demands(
        self, reward_size: float, reward_distance: float, reward_behind: float, x_pos: float, time_limit: float, env_number: int, filename: str, numbered: bool = False
    ) -> str:
        # Validate
        np.random.seed(0)
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

        goal_x_pos += reward_distance * x_pos
        goal_z_pos += reward_distance * (1 - abs(x_pos))

        # rotation of agent = f(reward_behind, x_pos)

        if reward_behind == 0:
            agent_rotation = 90 * x_pos
        elif reward_behind == 0.5:
            if x_pos == 0:
                agent_rotation = np.random.choice([-90, 90])
            else:
                agent_rotation = 0
        else:
            agent_rotation = 90 * x_pos + 180
        agent_rotation = agent_rotation % 360  # This stops negative values.
        final_part = ""

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
            signboard_x = 20 + distance_away * np.sin(agent_rotation * np.pi / 180)
            signboard_z = 20 + distance_away * np.cos(agent_rotation * np.pi / 180)
            symbolpattern = ["1" * (env_number // 3 + 2) if i % 2 == 0 else "0" * (env_number // 3 + 2) for i in range(env_number // 3 + 3)]
            symbolpattern = "/".join(symbolpattern)
            final_part = f"""
            - !Item
              name: SignBoard
              positions:
              - !Vector3 {{x: {signboard_x}, y: 0, z: {signboard_z}}}
              sizes:
              - !Vector3 {{x: 2, y: 2, z: 2}}
              rotations: [{agent_rotation + 270}]
              symbolNames:
              - "{symbolpattern}"
          """

        with open(filename, "w") as text_file:
            text_file.write(self.initial_part + generated_config + final_part)
        return generated_config + final_part

    def gen_config_from_demands_batch(self, envs: List[Demands], filename: str, time_limit: Union[int, Iterable] = 100, numbered: int = False) -> str:
        number_of_initial_envs = 0
        new_conf = ""
        time_iterable = isinstance(time_limit, Iterable)
        if time_iterable:
            assert len(time_limit) == len(envs)
        for i in range(number_of_initial_envs):
            tl_input = int(time_limit[i]) if time_iterable else time_limit
            env_conf = self.gen_config_from_demands(5, 10, 0, 0, tl_input, i, f"temp", numbered=False)
            new_conf += env_conf
        for i, env in enumerate(envs):
            for j in range(3):
                tl_input = int(time_limit[i]) if time_iterable else time_limit
                if self.precise:
                    env_conf = self.gen_config_from_demands_precise(env.reward_size, env.reward_distance, env.reward_behind, env.Xpos, tl_input, (number_of_initial_envs) + j + 3 * i, f"temp", numbered=numbered)
                else:
                    env_conf = self.gen_config_from_demands(env.reward_size, env.reward_distance, env.reward_behind, env.Xpos, tl_input, (number_of_initial_envs) + j + 3 * i, f"temp", numbered=numbered)
                new_conf += env_conf
        with open(filename, "w") as text_file:
            text_file.write(self.initial_part + new_conf)
        return self.initial_part + new_conf

    def gen_config_from_demands_batch_random(self, n_envs: int, filename: str, size_min=0, size_max=1.9, dist_min=None, dist_max=5.3, time_limit=100, numbered=False, seed = 0) -> tuple[str, List[Demands]]:
        np.random.seed(seed)
        demands_list = []
        i = 0
        dist_min = size_max + 0.5 if dist_min is None else dist_min
        if dist_min < size_max + 0.5:
            dist_min = size_max + 0.5
            print("Distance min too small, setting to size_min + 0.5, this is to prevent clipping.")
        while i < n_envs:
            size = np.random.uniform(size_min, size_max)  # This should prevent clipping
            xpos_choice = np.random.choice([-1, 0, 1])
            reward_behind_choice = np.random.choice([0, 0.5, 1])
            if reward_behind_choice == 0.5 and xpos_choice == 0 and self.precise:
                continue
            demands_list.append(Demands(size, np.random.uniform(dist_min, dist_max), reward_behind_choice, xpos_choice))
            i += 1
        return self.gen_config_from_demands_batch(demands_list, filename, time_limit, numbered=numbered), demands_list

    def gen_config_from_demands_precise(
        self, reward_size: float, reward_distance: float, reward_behind: float, x_pos: float, time_limit: float, env_number: int, filename: str, numbered: bool = False
    ) -> str:
        # Validate
        try:
            assert reward_size >= 0 and reward_size <= 1.9
        except:
            print("size greater than intended range.")
            print(reward_size)
            if reward_size > 6:
                raise ValueError(f"Size is too large, {reward_size}")
        assert reward_behind in [0, 0.5, 1]
        assert x_pos in [-1, 0, 1]
        goal_x_pos, goal_z_pos = 20, 20  # Centre point
        
        # config has fresh definitions:
        # for instance, the reward will now be at an angle theta, measured from the z-axis centre clockwise (so +ve would be right)
        # In order to get an idea of extent to which the reward is left or right, can do sin(theta), so arcsin(theta) will associate with the x_pos
        # To get an extent of behindness of the reward, can do cos(theta), so arccos(theta) will associate with the reward_behind
        
                
        angles_dictionary = {
            "north none" : 0,
            "north east" : 45,
            "north west" : 315,
            "south none" : 180,
            "south east" : 135,
            "south west" : 225,
            "none east" : 90,
            "none west" : 270,
        }
        
        directions_reward_behind = {
            0 : "north",
            1 : "south",
            0.5 : "none",
        }
        directions_reward_xpos = {
            0 : "none", 
            1 : "east",
            -1 : "west",
        }
        
        
        goal_direction = directions_reward_behind[reward_behind] + " " + directions_reward_xpos[x_pos]
        if goal_direction == "none none":
            raise ValueError(f"The reward is in contradicting directions {goal_direction}, this is not allowed.")
        angle_of_reward = angles_dictionary[goal_direction]
        angle_of_reward = angle_of_reward * np.pi / 180 # convert to radians for sin and cos functions.
        goal_x_pos += reward_distance *np.sin(angle_of_reward)
        goal_z_pos += reward_distance * np.cos(angle_of_reward)
     
        agent_rotation = 0
        final_part = ""

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
            signboard_x = 20 + distance_away * np.sin(agent_rotation * np.pi / 180)
            signboard_z = 20 + distance_away * np.cos(agent_rotation * np.pi / 180)
            symbolpattern = ["1" * (env_number // 3 + 2) if i % 2 == 0 else "0" * (env_number // 3 + 2) for i in range(env_number // 3 + 3)]
            symbolpattern = "/".join(symbolpattern)
            final_part = f"""
            - !Item
              name: SignBoard
              positions:
              - !Vector3 {{x: {signboard_x}, y: 0, z: {signboard_z}}}
              sizes:
              - !Vector3 {{x: 2, y: 2, z: 2}}
              rotations: [{agent_rotation + 270}]
              symbolNames:
              - "{symbolpattern}"
          """

        with open(filename, "w") as text_file:
            text_file.write(self.initial_part + generated_config + final_part)
        return generated_config + final_part
    
    def clean_yaml_file(self, filename: str) -> None:
            import re
            # Read the YAML file
            with open(filename, "r") as file:
                lines = file.readlines()

            # Remove unnecessary leading whitespace
            cleaned_lines = [re.sub(r'^\s+', '', line) for line in lines]

            # Write back to a new YAML file
            with open(filename, "w") as file:
                file.writelines(cleaned_lines)

            print("Unnecessary whitespace removed. Cleaned YAML saved as cleaned_output.yaml")
