
import yaml

# Custom constructor for handling unknown YAML tags (needed to avoid errors when loading the YAML file for AAI)
def custom_constructor(loader, tag_suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    elif isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return None


yaml.add_multi_constructor("", custom_constructor)


def parse_config_file(file_path):
    with open(file_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        return config


def extract_objects_positions(config):
    objects_positions = {}
    for arena_id, arena in config["arenas"].items():
        for item in arena["items"]:
            object_name = f"{item['name']}_arena{arena_id}"
            positions = item.get("positions", [])
            if object_name not in objects_positions:
                objects_positions[object_name] = []
            objects_positions[object_name].extend(positions)
    return objects_positions


# Specify the file path here
file_path = "example_eval.yaml"
config = parse_config_file(file_path)
objects_positions = extract_objects_positions(config)
print(objects_positions)