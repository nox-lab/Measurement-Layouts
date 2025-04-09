# Import the necessary libraries
import sys
import random
import os
from generating_configs_class import ConfigGenerator
from animalai.environment import AnimalAIEnvironment
from mlagents_envs.exception import UnityCommunicationException
from demands import Demands
import numpy as np

# IMPORTANT! Replace configuration file with the correct path here:
configuration_file = r"example_batch_eval.yaml"
# test = False
config_generator = ConfigGenerator(very_precise = True)

config_generator.gen_config_closed(0, 10)

# config_string, demands_list = config_generator.gen_config_from_demands_batch_random(3, configuration_file, time_limit=75, dist_max=15, size_max = 0.2, numbered = True)
# for demand in demands_list:
#     print(demand)

#config_generator.random_positions(configuration_file, configuration_file)
#config, demands = gen_config_from_demands_batch_random(10, configuration_file, time_limit=75, dist_max=15, numbered = True)
# for demand in demands:
#     print(demand)
# if test:
#     # This tests if the class is generating arenas correctly, should match with original function.
#     envs_demands = []
#     xpos_choices = [-1, 0, 1]
#     reward_behind_choices = [0, 0.5, 1]
#     for i in range(10):
#         for j in range(10):
#             xpos = np.random.choice(xpos_choices)
#             reward_behind = np.random.choice(reward_behind_choices)
#             envs_demands.append(Demands(0.5, 5, reward_behind, xpos))

#     config_generator.gen_config_from_demands_batch(envs_demands, r"random_file_yaml.yaml", time_limit=100, numbered = False) == gen_config_from_demands_batch(envs_demands, r"another_random_file.yaml", time_limit = 100, numbered = False)

#     config_generator.clean_yaml_file(r"random_file_yaml.yaml")
#     config_generator.clean_yaml_file(r"another_random_file.yaml")


#     with open(r"random_file_yaml.yaml") as f:
#         random_file = f.read()
#     with open(r"another_random_file.yaml") as f:
#         another_random_file = f.read()


#     assert random_file == another_random_file



# with open(configuration_file) as f:
#     print(f.read()) 
# IMPORTANT! Replace the path to the application .exe here:
env_path = r"..\WINDOWS\AAI\Animal-AI.exe"

port = 5005 + random.randint(
    0, 1000
)  # uses a random port to avoid problems with parallel runs

print("Initializing AAI environment")
try:
    environment = AnimalAIEnvironment(
        file_name=env_path,
        base_port=port,
        arenas_configurations=configuration_file,
        play=True,
    )

except UnityCommunicationException:
    # you'll end up here if you close the environment window directly
    # always try to close it from script (environment.close())
    environment.close()
    print("Environment was closed")

if environment:
    environment.close() # takes a few seconds to close...