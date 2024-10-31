def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines
agent_confirmed = False
position_confirmed = False
file_path_read = 'example.yaml'
file_path_write = 'example2.yaml'
yaml_content = read_yaml_file(file_path_read)
for line in yaml_content:
    if "Agent" in line:
        agent_confirmed = True
    if agent_confirmed and "positions" in line:
        position_confirmed = True
    if agent_confirmed and position_confirmed and "Vector3" in line:
        agent_confirmed = False
        position_confirmed = False
        print(line)
        parts = line.split()
        print(parts)
        for i, part in enumerate(parts):
            if part.startswith('z:'):
                parts[i+1] = '5'  # Replace 'new_value' with the desired z coordinate
        new_line = ' '.join(parts)
        yaml_content[yaml_content.index(line)] = new_line

    with open(file_path_write, 'w') as file:
        file.writelines(yaml_content)
        