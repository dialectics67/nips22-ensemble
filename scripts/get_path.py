import os
import os.path as osp

path = '../embeddingVec/'

for model_name in sorted(os.listdir(path)):
    for long_file_name in os.listdir(osp.join(path, model_name)):
        for file_name in os.listdir(osp.join(path, model_name, long_file_name, "autosf_concat-complex_shallow-distmult_shallow-smore-autosf-complex_e-distmult_g-transe_l2-rule")):
            if file_name.split('_')[0] == 'valid':
                print("--infer_rst_val_path_list ", 
                osp.join(path, model_name, long_file_name, "autosf_concat-complex_shallow-distmult_shallow-smore-autosf-complex_e-distmult_g-transe_l2-rule", file_name), '\\')
