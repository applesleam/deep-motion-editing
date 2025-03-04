import os
import json
import numpy as np
import pickle as pkl
from datasets.bvh_parser import BVH_file
from option_parser import get_std_bvh
from Quaternions import Quaternions
from tqdm import tqdm

def load_bvh(file_path):
    file = BVH_file(file_path)
    rotations = file.anim.rotations[:, file.corps, :]
    rotations = Quaternions.from_euler(np.radians(rotations)).qs
    positions = file.anim.positions[:, 0, :]
    print('Rotations shape: ', rotations.shape)
    print('Positions shape: ', positions.shape)
    return rotations, positions

if __name__ == '__main__':
    test_file = '/data/scratch/acw750/Development/github/deep-motion-editing/retargeting/datasets/mixamo/test.json'
    with open(test_file, 'r') as file:
        test_dict = json.load(file)
    
    new_dict = {}
    
    for name in test_dict.keys():
        # if 'test_pairs_' in name:
        if 'test_pairs_' in name:
            # if name not in ["test_pairs_kk"]: continue
            new_dict[name] = []
            for idx, i in tqdm(enumerate(test_dict[name])):
                # skip the broken sample pair
                if idx == 46 and name == 'test_pairs_uu':
                    continue
                src_char = "_".join(i['source_character'].split())
                tgt_char = "_".join(i['terget_character'].split())
    
                src_bvh = i['source_motion_file'].replace('.fbx', '.bvh')
                tgt_bvh = i['target_motion_file'].replace('.fbx', '.bvh')
    
                folder_name = name + '-' + str(idx)
                tmp_path = './examples/intra_structure/' + folder_name
                if not os.path.exists(tmp_path):
                    print('Path not exists: ', tmp_path)
    
                gt_bvh = os.path.join(tmp_path, 'gt.bvh')
                input_bvh = os.path.join(tmp_path, 'input.bvh')
                result_bvh = os.path.join(tmp_path, 'result.bvh')
    
                gt_local_q, gt_global = load_bvh(gt_bvh)
                input_local_q, input_global = load_bvh(input_bvh)
                result_local_q, result_global = load_bvh(result_bvh)
    
                i['inference_result'] = {
                    "gt": {
                        "local_q": gt_local_q,
                        "global": gt_global
                    },
                    "input": {
                        "local_q": input_local_q,
                        "global": input_global
                    },
                    "result": {
                        "local_q": result_local_q,
                        "global": result_global
                    }
                }
                new_dict[name].append(i)
    
    # Save the new dictionary to a pickle file
    output_pickle = '/data/scratch/acw750/Development/github/deep-motion-editing/retargeting/datasets/mixamo/test_results_del_one.pkl'
    with open(output_pickle, 'wb') as f:
        pkl.dump(new_dict, f)
