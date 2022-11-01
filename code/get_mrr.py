import torch
import numpy as np
import argparse
import numpy as np
from ogb.lsc import WikiKG90Mv2Evaluator
import nni
import os
import time
def _get_args():
    def get_weight_tuple(s):
        weight_tuple = [0.1, 0.2]
        return weight_tuple
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_rst_val_path_list', action='append')
    parser.add_argument('--infer_rst_test_path_list', action='append')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--data_path', default=None)
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--weight_tuple', type=get_weight_tuple)
    args = parser.parse_args()
    return args


def aggregate_score_on_can(can, score_matrix):
    can_unique = -1 * np.ones_like(can, dtype=np.int64)
    score_matrix_unique = -1000000000000 * np.ones_like(score_matrix, dtype=np.float64)
    can_count_matrix=np.zeros_like(can, dtype=np.int64)
    for i in range(can.shape[0]):
        unique_elements, unique_elements_index, unique_elements_count = np.unique(
            can[i, :], return_index=True, return_counts=True)
        if unique_elements[0] == -1:
            unique_elements = unique_elements[1:]
            unique_elements_index = unique_elements_index[1:]
            unique_elements_count = unique_elements_count[1:]
        unique_elements_score = score_matrix[i, unique_elements_index]*unique_elements_count
        can_unique[i, 0:len(unique_elements)] = unique_elements
        score_matrix_unique[i, 0:len(unique_elements)] = unique_elements_score
        can_count_matrix[i, 0:len(unique_elements)]= unique_elements_count
    return can_unique, score_matrix_unique, can_count_matrix


def get_t_pred_top10(candidate, score_matrix_list, weight_list):
    ensemble_score_matirx = 0
    for score_matix, weight in zip(score_matrix_list, weight_list):  # 原来的时候，candidate固定，所以可以直接加，现在candidate也是相同的，所以也是可以直接相加的
        ensemble_score_matirx += score_matix * weight
    ensemble_score_matirx_sort_index = np.argsort(-ensemble_score_matirx, axis=1)
    ensemble_score_matirx_top10_index = ensemble_score_matirx_sort_index[:, :10]
    t_pred_top10 = np.take_along_axis(candidate, ensemble_score_matirx_top10_index, axis=1)
    return t_pred_top10


def normal_score_matrix(candidate, score_matrix, second_candidate, second_score_matrix):
    score_matrix_max = max(score_matrix[candidate != -1].max(),second_score_matrix[second_candidate!=-1].max())
    score_matrix_min = min(score_matrix[candidate != -1].min(),second_score_matrix[second_candidate!=-1].min())
    score_matrix = (score_matrix-score_matrix_min)/(score_matrix_max-score_matrix_min)
    return score_matrix, (score_matrix_max, score_matrix_min)


def preprocessed_can_score_matrix_list(candidate, score_matrix_list, param_list, second_candidate, second_score_matrix_list):
    # 将所有的score缩放到[0~1]
    st = time.time()
    for i in range(len(score_matrix_list)):
        score_matrix_list[i], tmp = normal_score_matrix(candidate, score_matrix_list[i], second_candidate, second_score_matrix_list[i])
        param_list.append(tmp)
    # 合并相同节点的得分
    can_unique = None
    can_count_matrix=None
    for i in range(len(score_matrix_list)):
        can_unique, score_matrix_list[i], can_count_matrix = aggregate_score_on_can(candidate, score_matrix_list[i])
    print("processing use:", time.time() - st)
    return can_unique, score_matrix_list, param_list, can_count_matrix


def get_mrr(val_t, val_can, score_matrix_list, weight_tuple):
    t_pred_top10 = get_t_pred_top10(val_can, score_matrix_list, weight_tuple)
    input_dict = {}
    input_dict['h,r->t'] = {'t_pred_top10': t_pred_top10, 't': val_t}
    evaluator = WikiKG90Mv2Evaluator()
    ret = evaluator.eval(input_dict)
    return ret['mrr']


def get_test_submit_dict(candidate, test_score_matrix_list, weight_list):
    t_pred_top10 = get_t_pred_top10(candidate, test_score_matrix_list, weight_list)
    test_submit_dict = {}
    test_submit_dict['h,r->t'] = {'t_pred_top10': t_pred_top10}
    return test_submit_dict

def apply_base(score_matrix_list,can_count_matrix,base):
    for i in range(len(score_matrix_list)):
        score_matrix_list[i]=score_matrix_list[i]+(can_count_matrix*base)
    return score_matrix_list

if __name__ == "__main__":
    # import pdb
    # pdb.set_trace()
    nni_searching = True
    args = _get_args()
    
    if nni_searching:
        vars(args)['nni'] = True
        if not args.preprocess:
            params = nni.get_next_parameter()
        else:
            params = {'base': 0, 'w_0': 0.1, 'w_1':0.1, 'w_2': 0.1, 'w_3':0.1, 'w_4': 0.1, 'w_5':0.1, 'w_6': 0.1, 'w_7':0.1, 'w_8': 0.1, 'w_9':0.1, 'w_10': 0.1, 'w_11':0.1, 'w_12':0.1}
        # weight_search_range = []
        num_models = 13
        weight_tuples = []
        for i in range(0,num_models):
            weight_tuples.append(params['w_'+str(i)])
        # args.weight_tuples = weight_tuples
        # get_mrr(args)
        vars(args)["weight_tuple"] = weight_tuples
        vars(args)["base"] = params['base']
    else:
        num_models=1
        vars(args)['nni'] = False
      
    # init weight_search_range
    weight_tuple = args.weight_tuple
    print(f"weight {weight_tuple}")

    # load all data
    if args.preprocess:
        tmp_val_infer_rst = torch.load(args.infer_rst_val_path_list[0])

        val_t = tmp_val_infer_rst['t']
        print("val_t.shape:", val_t.shape)
        val_can = tmp_val_infer_rst['t_candidate']
        print("val_can.shape", val_can.shape)

        val_score_matrix_list = []
        # get val_score_matrix_list
        for val_score_matrix_path in args.infer_rst_val_path_list:
            start = torch.load(val_score_matrix_path)
            if 'h,r->t' in start:
                start = start['h,r->t']['t_pred_score'].numpy()
            else:
                start = start['t_pred_score']
                if type(start) != np.ndarray:
                    start = start.numpy()
            val_score_matrix_list.append(start)
            print('valid score_matrix_path:', val_score_matrix_path, 'shpae:', start.shape)
        print('valid score matrix load done, matrix_num:', len(val_score_matrix_list))
        # get test_score_matrix_list
        test_can=torch.load(args.infer_rst_test_path_list[0])['t_candidate']
        test_score_matrix_list = []
        for test_score_matrix_path in args.infer_rst_test_path_list:
            start = torch.load(test_score_matrix_path)
            if 'h,r->t' in start:
                start = start['h,r->t']['t_pred_score'].numpy()
            else:
                start = start['t_pred_score']
                if type(start) != np.ndarray:
                    start = start.numpy()
            test_score_matrix_list.append(start)
            print('valid score_matrix_path:', test_score_matrix_path, 'shpae:', start.shape)
        print('test score matrix load done, matrix_num:', len(val_score_matrix_list))
        # preprocessed matrix
        params_list = []
        save_path = args.save_path
        # import pdb
        # pdb.set_trace()
        # can_count_matrix unique candidate 相应位置的出现次数
        val_can, val_score_matrix_list, params_list, can_count_matrix = preprocessed_can_score_matrix_list(val_can, val_score_matrix_list, params_list, test_can, test_score_matrix_list)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, 'val_can.npy'), val_can)
        np.save(os.path.join(save_path, 'val_t.npy'), val_t)
        np.save(os.path.join(save_path, 'val_can_count_matrix.npy'), can_count_matrix)
        for i in range(len(val_score_matrix_list)): np.savez(os.path.join(save_path, 'model_'+str(i)+'_val_score_matrix_minmax.npz'), score_matrix=val_score_matrix_list[i], max=params_list[i][0], min=params_list[i][1])
    else:
        data_path = args.data_path
        val_can = np.load(os.path.join(data_path, 'val_can.npy'))
        val_t = np.load(os.path.join(data_path, 'val_t.npy'))
        can_count_matrix=np.load(os.path.join(data_path, 'val_can_count_matrix.npy'))
        val_score_matrix_list = []
        for i in range(num_models):
            data = np.load(os.path.join(data_path, 'model_'+str(i)+'_val_score_matrix_minmax.npz'))['score_matrix']
            val_score_matrix_list.append(data)
    val_score_matrix_list=apply_base(val_score_matrix_list,can_count_matrix,args.base)
    mrr = get_mrr(val_t, val_can, val_score_matrix_list, weight_tuple)
    print(mrr)
    if nni_searching:
        nni.report_final_result(mrr)

