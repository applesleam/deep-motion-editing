import os
import json
import torch
from datasets.bvh_parser import BVH_file
from datasets.bvh_writer import BVH_writer
from models.IK import fix_foot_contact
from os.path import join as pjoin
from option_parser import try_mkdir
import option_parser
from models import create_model
from datasets import create_dataset

def eval_preprocess(input_bvh, target_bvh, test_type, output_filename):
    character = []
    file_id = []
    character_names = []
    character_names.append(input_bvh.split('/')[-2])
    character_names.append(target_bvh.split('/')[-2])
    if test_type == 'intra':
        if character_names[0].endswith('_m'):
            character = [['BigVegas', 'BigVegas'], character_names]
            file_id = [[0, 0], [input_bvh, target_bvh]]
            src_id = 1
        else:
            character = [character_names, ['Goblin_m', 'Goblin_m']]
            file_id = [[input_bvh, input_bvh], [0, 0]]
            src_id = 0
    elif test_type == 'cross':
        if character_names[0].endswith('_m'):
            character = [[character_names[1]], [character_names[0]]]
            file_id = [[0], [input_bvh]]
            src_id = 1
        else:
            character = [[character_names[0]], [character_names[1]]]
            file_id = [[input_bvh], [0]]
            src_id = 0
    else:
        raise Exception('Unknown test type')
    return character, file_id, src_id

def eval_single_pair(input_bvh, target_bvh, test_type, output_filename):
    character_names, file_id, src_id = eval_preprocess(input_bvh, target_bvh, test_type, output_filename)
    input_character_name = input_bvh.split('/')[-2]
    output_character_name = target_bvh.split('/')[-2]
    
    test_device = 'cuda:0'
    eval_seq = 0

    para_path = os.path.join('./pretrained', 'para.txt')
    with open(para_path, 'r') as para_file:
        argv_ = para_file.readline().split()[1:]
        args = option_parser.get_parser().parse_args(argv_)

    args.cuda_device = test_device if torch.cuda.is_available() else 'cpu'
    args.is_train = False
    args.rotation = 'quaternion'
    args.eval_seq = eval_seq

    dataset = create_dataset(args, character_names)
    
    model = create_model(args, character_names, dataset)
    model.load(epoch=20000)

    input_motion = []
    for i, character_group in enumerate(character_names):
        input_group = []
        for j in range(len(character_group)):
            new_motion = dataset.get_item(i, j, file_id[i][j])
            new_motion.unsqueeze_(0)
            new_motion = (new_motion - dataset.mean[i][j]) / dataset.var[i][j]
            input_group.append(new_motion[:,:,:100])
        print(input_group[0].shape)
        print(input_group[1].shape)
        input_group = torch.cat(input_group, dim=0)
        input_motion.append([input_group, list(range(len(character_group)))])

    model.set_input(input_motion)
    model.test()

    cmd = 'cp "{}/{}/0_{}.bvh" "{}"'.format(model.bvh_path, output_character_name, src_id, output_filename)
    print(cmd)
    os.system(cmd)


# downsampling and remove redundant joints
def copy_ref_file(src, dst):
    file = BVH_file(src)
    writer = BVH_writer(file.edges, file.names)
    writer.write_raw(file.to_tensor(quater=True)[..., ::2], 'quaternion', dst)


def get_height(file):
    file = BVH_file(file)
    return file.get_height()


def example(src_name, dest_name, src_bvh_name, dest_bvh_name, test_type, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_file = './datasets/Mixamo_XHY_m/{}/{}'.format(src_name, src_bvh_name)
    ref_file = './datasets/Mixamo_XHY_m/{}/{}'.format(dest_name, dest_bvh_name)
    print("input: ", input_file)
    print("ref: ", ref_file)
    copy_ref_file(input_file, pjoin(output_path, 'input.bvh'))
    copy_ref_file(ref_file, pjoin(output_path, 'gt.bvh'))
    height = get_height(input_file)

    eval_single_pair(input_file, ref_file, test_type, pjoin(output_path, 'result.bvh'))

    # cmd = 'python eval_single_pair.py --input_bvh={} --target_bvh={} --output_filename={} --test_type={}'.format(
    #     input_file, ref_file, pjoin(output_path, 'result.bvh'), test_type
    # )
    # print(cmd)
    # os.system(cmd)

    fix_foot_contact(pjoin(output_path, 'result.bvh'),
                     pjoin(output_path, 'input.bvh'),
                     pjoin(output_path, 'result.bvh'),
                     height)


if __name__ == '__main__':
    test_file = '/data/scratch/acw750/Development/github/deep-motion-editing/retargeting/datasets/mixamo/test.json'
    with open(test_file, 'r') as file:
        test_dict = json.load(file)
    for name in test_dict.keys():
        if 'test_pairs_' in name:
            for idx, i in enumerate(test_dict[name]):
                src_char = "_".join(i['source_character'].split())
                tgt_char = "_".join(i['terget_character'].split())
                
                src_bvh = i['source_motion_file'].replace('.fbx', '.bvh')
                tgt_bvh = i['target_motion_file'].replace('.fbx', '.bvh')

                folder_name = name + '-' + str(idx)
                tmp_path = './examples/intra_structure/' + folder_name
                if not os.path.exists(tmp_path):
                    try_mkdir('./examples/intra_structure/' + folder_name)
                else:
                    continue
                example(src_char+'_m', tgt_char+'_m', src_bvh, tgt_bvh, 'intra', './examples/intra_structure/' + folder_name)
    # example('Kaya_m', 'Peasant_Man_m', 'Roar.bvh', 'intra', './examples/intra_structure')
    # example('Aj', 'BigVegas', 'Dancing Running Man.bvh', 'intra', './examples/intra_structure')
    # example('BigVegas', 'Mousey_m', 'Dual Weapon Combo.bvh', 'cross', './examples/cross_structure')
    print('Finished!')
