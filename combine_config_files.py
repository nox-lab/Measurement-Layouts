'''gonna make a curriculum for the 01-xx-xx and 02-xx-xx arenas, using some os path join stuff whatever idk, hopefully will train successfully and changes can be evaluated.

NEED HPC WORKING AS WELL. works !

'''
import os
import sys
import re
import numpy as np

initial_dir = r"C:\Users\talha\Documents\iib_projects\Measurement-Layouts\GIBSON_arenas\\"
save_file = r"GIBSON_arenas/navigation_occlusion_lava.yaml"
final_yaml_set = "!ArenaConfig\narenas:\n"
number_arena = 0
i = 0
for dirpath, dirname, file_list in os.walk(initial_dir):
    # should be 7 x 2 x 3 = 42 arenas
    dirpath_final_part = dirpath.split("\\")[-1]
    for file in file_list:
        print(dirpath)
        print(dirpath + "\\" +file)
        print(file)
        file = dirpath_final_part + "\\" + file
        ending_filename = file.split("\\")[-1]
        if not ending_filename[0:2] == "00" or not file.endswith(".yaml"):
            continue
        with open(initial_dir + file, 'r') as fin: 
            arena_part = "".join(fin.readlines()[2:])
            arena_part = arena_part.replace('0', str(number_arena), 1)
            final_yaml_set += arena_part + "\n"
            number_arena += 1

with open(save_file, 'w') as fout:
    fout.write(final_yaml_set)
    print("curriculum.yaml written successfully")
    print("number of arenas: ", number_arena)