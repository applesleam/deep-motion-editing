"""
Convert the test set defined by XHY to the format fitted in this codebase
source dataset: /data/scratch/acw750/Development/github/deep-motion-editing/retargeting/datasets/mixamo
target file: datasets/Mixamo/test_list.txt

test the path of .bvh file, and copy to a new folder
"""
import os
import os.path as osp
import json
import shutil

root_path = "/data/scratch/acw750/Development/github/deep-motion-editing/retargeting/datasets/mixamo"
input_file = osp.join(root_path, "test.json")

output_path = "/data/scratch/acw750/Development/github/deep-motion-editing/retargeting/datasets/Mixamo_XHY_bvh"

with open(input_file, "r") as f:
    test_dict = json.load(f)

for name in test_dict.keys():
    test_list = []
    if "test_pairs_" in name:
        print("Processing: " + name)
        for sample in test_dict[name]:
            src_char = sample["source_character"]
            tgt_char = sample["terget_character"]
            src_char_name = "_".join(src_char.split())
            tgt_char_name = "_".join(tgt_char.split())
            src_motion = sample["source_motion_file"]
            tgt_motion = sample["target_motion_file"]
            src_bvh = src_motion.replace(".fbx", ".bvh")
            tgt_bvh = tgt_motion.replace(".fbx", ".bvh")
            # src_bvh = src_motion
            # tgt_bvh = tgt_motion

            src_bvh_path = osp.join(root_path, "test_source", src_char_name, src_bvh)
            tgt_bvh_path = osp.join(root_path, "test_source", tgt_char_name, tgt_bvh)

            src_folder = osp.join(output_path, src_char_name)
            tgt_folder = osp.join(output_path, tgt_char_name)
            if not osp.exists(src_folder):
                os.mkdir(src_folder)

            if not osp.exists(tgt_folder):
                os.mkdir(tgt_folder)

            shutil.copy(tgt_bvh_path, tgt_folder)
            shutil.copy(src_bvh_path, src_folder)
            
            if osp.exists(src_bvh_path):
                if src_bvh not in test_list:
                    test_list.append(src_bvh)
            else:
                print("Path not exist! " + src_bvh_path)

            if osp.exists(tgt_bvh_path):
                if tgt_bvh not in test_list:
                    test_list.append(tgt_bvh)
            else:
                print("Path not exist! " + tgt_bvh_path)

        out_file = "datasets/" + name + "_0301.txt"
        with open(out_file, "w") as f:
            for item in test_list:
                f.write(item + "\n")

