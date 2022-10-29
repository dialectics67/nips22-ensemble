import torch
import numpy as np
import argparse
import numpy as np
from ogb.lsc import WikiKG90Mv2Evaluator
import nni
import os
import time
from tqdm import trange
from scipy.special import log_softmax

def _get_args():
    def get_weight_tuple(s):
        weight_tuple = [0.1, 0.2]
        return weight_tuple
    parser = argparse.ArgumentParser()
    parser.add_argument('--infer_rst_val_path_list', action='append')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--data_path', default=None)
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--nni', action='store_true')
    parser.add_argument('--logsoftmax', action='store_true')
    parser.add_argument('--weight_tuple', type=get_weight_tuple)
    args = parser.parse_args()
    return args


def get_can_model_index():
    can_model_index = []
    for i in range(3):
        can_model_index.extend([i for _ in range(1000)])
    can_model_index.extend([3 for i in range(20000)])
    for i in range(4, 8):
        can_model_index.extend([i for _ in range(1000)])
    can_model_index.extend([8 for _ in range(50)])
    can_model_index = np.asarray(can_model_index, dtype=np.int32)
    return can_model_index

def aggregate_score_on_can_with_weight(val_can, sorted_can, argsorted_can, score_matrix, can_model_index, model_weight):
    """


    can: num_query * num_candidates, node index of candidates (including candidates from all the recall models), np.int32

    score_matrix: num_query * num_candidates, node score corresponding to parameter can, np.float32, will be updated to save memory

    can_model_index: num_query * num_candidates, which recall model the candidate from (model index start from 0, value range [0, num_recall_model) ), np.int32, set 0 for candidate ID -1

    model_weight: numpy vector with num_recall_model values, np.float32
    """
    can_ret = -1 * np.ones_like(score_matrix, dtype=np.int32)
    score_matrix_ret = np.zeros_like(score_matrix, dtype=np.float32)
    max_len = 0
    
    #update score_matrix with model weights
    np.multiply(score_matrix, model_weight[can_model_index], out = score_matrix)
    
    for i in trange(score_matrix.shape[0]):
        # 去重后的can
        unq_can = np.unique(val_can[i])
        unq_can_reverse = {v:k for k, v in enumerate(unq_can)}
        
        # sorted can对应uniq list位置的映射，即哪些原始的idx会scatter到哪个can idx
        norm_sorted_can = np.array([unq_can_reverse[item] for item in sorted_can[i]], dtype=np.int32)
        # scatter add implemented by numpy
        np.add.at(score_matrix_ret[i, :], norm_sorted_can, score_matrix[i, argsorted_can[i]])
        
        len_unq = len(unq_can)
        max_len = max(max_len, len_unq)
        if unq_can[0] == -1:
            can_ret[i, 0:len_unq-1] = unq_can[1:len_unq]
            score_matrix_ret[i, 0:len_unq-1] = score_matrix_ret[i, 1:len_unq]
            score_matrix_ret[i, len_unq-1:] = -1000000000000
        else:
            can_ret[i, 0:len_unq] = unq_can
            score_matrix_ret[i, len_unq:] = -1000000000000
        
    return can_ret, score_matrix_ret





    # ensemble_score_matirx[candidate==-1]

def normal_score_matrix(candidate, score_matrix):
    score_matrix_max = score_matrix[candidate != -1].max()
    score_matrix_min = score_matrix[candidate != -1].min()
    score_matrix = (score_matrix-score_matrix_min)/(score_matrix_max-score_matrix_min)
    return score_matrix, (score_matrix_max, score_matrix_min)

def normal12_score_matrix(candidate, score_matrix):
    score_matrix_max = score_matrix[candidate != -1].max()
    score_matrix_min = score_matrix[candidate != -1].min()
    score_matrix = (score_matrix-score_matrix_min)/(score_matrix_max-score_matrix_min) + 1
    return score_matrix, (score_matrix_max, score_matrix_min)

def normal_logSoftmax_score_matrix(candidate, score_matrix):
    score_matrix[candidate != -1] = -1 * log_softmax(score_matrix[candidate != -1])
    return score_matrix, (0, 0)


def preprocessed_can_score_matrix_list(candidate, score_matrix_list, param_list, args):
    # 将所有的score缩放到[0~1]
    st = time.time()
    
    for i in range(len(score_matrix_list)):
        if args.logsoftmax:
            score_matrix_list[i], tmp = normal_logSoftmax_score_matrix(candidate, score_matrix_list[i])
        else:
            score_matrix_list[i], tmp = normal_score_matrix(candidate, score_matrix_list[i])
        param_list.append(tmp)
    # 合并相同节点的得分
    print("processing use:", time.time() - st)
    return score_matrix_list, param_list



def get_t_pred_top10(candidate, ensemble_score_matirx, args):
    # import pdb
    # pdb.set_trace()
    if not args.logsoftmax:
        ensemble_score_matirx_sort_index = np.argsort(-ensemble_score_matirx, axis=1)
    else:
        ensemble_score_matirx_sort_index = np.argsort(ensemble_score_matirx, axis=1)
    ensemble_score_matirx_top10_index = ensemble_score_matirx_sort_index[:, :10]
    t_pred_top10 = np.take_along_axis(candidate, ensemble_score_matirx_top10_index, axis=1)
    return t_pred_top10

def get_mrr(val_t, uniq_can, ensemble_score_matirx, args):
    t_pred_top10 = get_t_pred_top10(uniq_can, ensemble_score_matirx, args)
    input_dict = {}
    input_dict['h,r->t'] = {'t_pred_top10': t_pred_top10, 't': val_t}
    # import pdb
    # pdb.set_trace()
    evaluator = WikiKG90Mv2Evaluator()
    ret = evaluator.eval(input_dict)
    return ret['mrr']


def get_test_submit_dict(candidate, test_score_matrix_list, weight_list):
    t_pred_top10 = get_t_pred_top10(candidate, test_score_matrix_list, weight_list)
    test_submit_dict = {}
    test_submit_dict['h,r->t'] = {'t_pred_top10': t_pred_top10}
    return test_submit_dict


if __name__ == "__main__":
    # import pdb
    # pdb.set_trace()
    
    args = _get_args()
    num_models = 13 
    num_can = 9
    if args.nni:
        params = nni.get_next_parameter()
    else:
        params = {'m_0_c_0': 0.1, 'm_0_c_1': 0.1, 'm_0_c_2': 0.1, 'm_0_c_3': 0.1, 'm_0_c_4': 0.1, 'm_0_c_5': 0.1, 'm_0_c_6': 0.1, 'm_0_c_7': 0.1, 'm_0_c_8': 0.1, 'm_1_c_0': 0.1, 'm_1_c_1': 0.1, 'm_1_c_2': 0.1, 'm_1_c_3': 0.1, 'm_1_c_4': 0.1, 'm_1_c_5': 0.1, 'm_1_c_6': 0.1, 'm_1_c_7': 0.1, 'm_1_c_8': 0.1, 'm_2_c_0': 0.1, 'm_2_c_1': 0.1, 'm_2_c_2': 0.1, 'm_2_c_3': 0.1, 'm_2_c_4': 0.1, 'm_2_c_5': 0.1, 'm_2_c_6': 0.1, 'm_2_c_7': 0.1, 'm_2_c_8': 0.1, 'm_3_c_0': 0.1, 'm_3_c_1': 0.1, 'm_3_c_2': 0.1, 'm_3_c_3': 0.1, 'm_3_c_4': 0.1, 'm_3_c_5': 0.1, 'm_3_c_6': 0.1, 'm_3_c_7': 0.1, 'm_3_c_8': 0.1, 'm_4_c_0': 0.1, 'm_4_c_1': 0.1, 'm_4_c_2': 0.1, 'm_4_c_3': 0.1, 'm_4_c_4': 0.1, 'm_4_c_5': 0.1, 'm_4_c_6': 0.1, 'm_4_c_7': 0.1, 'm_4_c_8': 0.1, 'm_5_c_0': 0.1, 'm_5_c_1': 0.1, 'm_5_c_2': 0.1, 'm_5_c_3': 0.1, 'm_5_c_4': 0.1, 'm_5_c_5': 0.1, 'm_5_c_6': 0.1, 'm_5_c_7': 0.1, 'm_5_c_8': 0.1, 'm_6_c_0': 0.1, 'm_6_c_1': 0.1, 'm_6_c_2': 0.1, 'm_6_c_3': 0.1, 'm_6_c_4': 0.1, 'm_6_c_5': 0.1, 'm_6_c_6': 0.1, 'm_6_c_7': 0.1, 'm_6_c_8': 0.1, 'm_7_c_0': 0.1, 'm_7_c_1': 0.1, 'm_7_c_2': 0.1, 'm_7_c_3': 0.1, 'm_7_c_4': 0.1, 'm_7_c_5': 0.1, 'm_7_c_6': 0.1, 'm_7_c_7': 0.1, 'm_7_c_8': 0.1, 'm_8_c_0': 0.1, 'm_8_c_1': 0.1, 'm_8_c_2': 0.1, 'm_8_c_3': 0.1, 'm_8_c_4': 0.1, 'm_8_c_5': 0.1, 'm_8_c_6': 0.1, 'm_8_c_7': 0.1, 'm_8_c_8': 0.1, 'm_9_c_0': 0.1, 'm_9_c_1': 0.1, 'm_9_c_2': 0.1, 'm_9_c_3': 0.1, 'm_9_c_4': 0.1, 'm_9_c_5': 0.1, 'm_9_c_6': 0.1, 'm_9_c_7': 0.1, 'm_9_c_8': 0.1, 'm_10_c_0': 0.1, 'm_10_c_1': 0.1, 'm_10_c_2': 0.1, 'm_10_c_3': 0.1, 'm_10_c_4': 0.1, 'm_10_c_5': 0.1, 'm_10_c_6': 0.1, 'm_10_c_7': 0.1, 'm_10_c_8': 0.1, 'm_11_c_0': 0.1, 'm_11_c_1': 0.1, 'm_11_c_2': 0.1, 'm_11_c_3': 0.1, 'm_11_c_4': 0.1, 'm_11_c_5': 0.1, 'm_11_c_6': 0.1, 'm_11_c_7': 0.1, 'm_11_c_8': 0.1, 'm_12_c_0': 0.1, 'm_12_c_1': 0.1, 'm_12_c_2': 0.1, 'm_12_c_3': 0.1, 'm_12_c_4': 0.1, 'm_12_c_5': 0.1, 'm_12_c_6': 0.1, 'm_12_c_7': 0.1, 'm_12_c_8': 0.1, 'm_13_c_0': 0.1, 'm_13_c_1': 0.1, 'm_13_c_2': 0.1, 'm_13_c_3': 0.1, 'm_13_c_4': 0.1, 'm_13_c_5': 0.1, 'm_13_c_6': 0.1, 'm_13_c_7': 0.1, 'm_13_c_8': 0.1, 'm_14_c_0': 0.1, 'm_14_c_1': 0.1, 'm_14_c_2': 0.1, 'm_14_c_3': 0.1, 'm_14_c_4': 0.1, 'm_14_c_5': 0.1, 'm_14_c_6': 0.1, 'm_14_c_7': 0.1, 'm_14_c_8': 0.1}
    model_weights = np.zeros((num_models, num_can))
    
    for i in range(num_models):
        for j in range(num_can):
            model_weights[i, j] = params["m_%s_c_%s"%(i, j)]

    vars(args)["weight_tuple"] = model_weights
    # init weight_search_range
    weight_tuple = args.weight_tuple
    
    print(args)
   
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

        # preprocessed matrix
        params_list = []
        save_path = args.save_path
        
        val_can = val_can.astype(np.int32)
        val_t = val_t.astype(np.int32)
        val_score_matrix_list, params_list = preprocessed_can_score_matrix_list(val_can, val_score_matrix_list, params_list, args)
        
        # can的idx进行排序
        sorted_can = np.sort(val_can, axis=1, kind='stable')
        # can排序前的位置
        argsorted_can = np.argsort(val_can, axis=1, kind='stable')
        argsorted_can = argsorted_can.astype(np.int32)
       
        # for i in range(len(val_score_matrix_list)): print(score_matrix=val_score_matrix_list[i].min(), score_matrix=val_score_matrix_list[i].max())
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savez(os.path.join(save_path, 'val_can.npz'), val_can=val_can, sorted_can=sorted_can, argsorted_can=argsorted_can)
        np.save(os.path.join(save_path, 'val_t.npy'), val_t)
        for i in range(len(val_score_matrix_list)): np.savez(os.path.join(save_path, 'model_'+str(i)+'_val_score_matrix.npz'), score_matrix=val_score_matrix_list[i], max=params_list[i][0], min=params_list[i][1])
    else:#'model_0_val_score_matrix.npz'
        data_path = args.data_path
        data = np.load(os.path.join(data_path, 'val_can.npz'))
        val_can, sorted_can, argsorted_can = data['val_can'], data['sorted_can'], data['argsorted_can']
        val_t = np.load(os.path.join(data_path, 'val_t.npy'))
        val_score_matrix_list = []
        for i in range(num_models):
            data = np.load(os.path.join(data_path, 'model_'+str(i)+'_val_score_matrix.npz'))['score_matrix']
            val_score_matrix_list.append(data)
    can_model_index = get_can_model_index()
    agg_score_matrics = None
    uniq_can = None
    st = time.time()
    
    for i in range(len(val_score_matrix_list)):
        can, agg_score_matrix = aggregate_score_on_can_with_weight(val_can, sorted_can, argsorted_can, val_score_matrix_list[i], can_model_index, model_weights[i])
        if agg_score_matrics is None:
            agg_score_matrics = agg_score_matrix
        else:
            agg_score_matrics += agg_score_matrix
        assert can.shape == agg_score_matrix.shape
    print(time.time() - st)
    # import pdb
    # pdb.set_trace()
    uniq_can = can
    mrr = get_mrr(val_t, uniq_can, agg_score_matrics, args)
    print(mrr)
    if args.nni:
        nni.report_final_result(mrr)
