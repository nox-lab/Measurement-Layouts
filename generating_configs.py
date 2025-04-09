from generating_configs_class import ConfigGenerator
from demands import Demands

generator_config = ConfigGenerator(closed = True)

demand_inputs = [Demands(1, x+2, 45, 0) for x in range(0, 20)]

generator_config.gen_config_from_demands_batch(demand_inputs, "testing_imermediate_OPEN.yaml", time_limit= 250)