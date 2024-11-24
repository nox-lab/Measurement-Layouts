# Import the necessary libraries
import sys
import random
import os
from generating_configs import gen_config_from_demands_batch_random
from animalai.environment import AnimalAIEnvironment
from mlagents_envs.exception import UnityCommunicationException

# IMPORTANT! Replace configuration file with the correct path here:
configuration_file = r"curriculum.yaml"

with open(configuration_file) as f:
    print(f.read()) 

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