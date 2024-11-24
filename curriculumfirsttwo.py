'''gonna make a curriculum for the 01-xx-xx and 02-xx-xx arenas, using some os path join stuff whatever idk, hopefully will train successfully and changes can be evaluated.

NEED HPC WORKING AS WELL. works !

'''
import os
import sys
import re
import numpy as np

initial_dir = r"AAIO_configs\competition\\"
final_yaml_set = "!ArenaConfig\narenas:\n"
number_arena = 0
for file in os.listdir(initial_dir):
    if (file[0:2] == '01' or file[0:2] == '02') and file[3:5] in [str(i).zfill(2) for i in range(1, 25) if i != 18 and i != 19 and (i < 19 or file[0:2] == '01')]:
        # should be 7 x 2 x 3 = 42 arenas
        print(file)
        with open(initial_dir + file, 'r') as fin: 
            arena_part = "".join(fin.readlines()[2:])
            arena_part = arena_part.replace('0', str(number_arena), 1)
            final_yaml_set += arena_part + "\n"
            number_arena += 1

with open("curriculum.yaml", 'w') as fout:
    fout.write(final_yaml_set)
    print("curriculum.yaml written successfully")
    print("number of arenas: ", number_arena)