import os
import pickle as pkl
from demo_convert_result import load_bvh

file_path = "/data/EECS-YuanLab/XHYang/ToQing/resorted_inference_pairs.pkl"
result_folder = "./pretrained/results_XHY_infer/intra_structure"

with open('./datasets/Mixamo_XHY_infer/test_list.txt', 'r') as file:
    motion_list = file.readlines()
    motion_list = [f.split(".")[0] for f in motion_list]

with open(file_path, 'rb') as f:
    data_list = pkl.load(f)

result_list = []
for sample in data_list:
    source = sample[0]
    target = sample[2]
    src_folder = "from_" + source + "_m"
    tgt_folder = target + "_m"
    motion = sample[1]
    motion_idx = motion_list.index(motion)
    start_frame = sample[3]
    assert start_frame == 0

    result_path = result_folder + "/{}/{}/{}.bvh".format(src_folder, tgt_folder, motion_idx)
    if not os.path.exists(result_path):
        print("File not exists: ", result_path)

    video_name = "{}_from_{}_to_{}.mp4".format(motion, source, target)
    result_local_q, result_global = load_bvh(result_path)
    one_dict = {
        "video_name": video_name,
        "meta_info": sample,
        "result": {
            "local_q": result_local_q,
            "global": result_global
        }
    }
    result_list.append(one_dict)

# # Save the result list to a pickle file
# output_pickle = '/data/EECS-YuanLab/XHYang/ToQing/inference_results_5x20.pkl'
# with open(output_pickle, 'wb') as f:
#     pkl.dump(result_list, f)

