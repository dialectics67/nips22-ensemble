import operator
import pickle
import torch
import numpy as np
import argparse
import os
import numpy as np
import multiprocessing
from ogb.lsc import WikiKG90Mv2Evaluator

def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_proc', type=int, default=1)
    parser.add_argument('--num_per_proc', type=int, default=1)
    parser.add_argument('--infer_rst_val_path_list', action='append')
    parser.add_argument('--infer_rst_test_path_list', action='append')
    parser.add_argument('--folder_path_ensemble_rst', type=str)
    args = parser.parse_args()
    assert args.num_proc < multiprocessing.cpu_count()
    return args


def aggregate_score_on_can(can, score_matrix):
    can_unique = -1 * np.ones_like(can, dtype=np.int64)
    score_matrix_unique = -1000000000000 * np.ones_like(score_matrix, dtype=np.float64)
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
    return can_unique, score_matrix_unique


def get_t_pred_top10(candidate, score_matrix_list, weight_list):
    ensemble_score_matirx = 0
    for score_matix, weight in zip(score_matrix_list, weight_list):  # 原来的时候，candidate固定，所以可以直接加，现在candidate也是相同的，所以也是可以直接相加的
        ensemble_score_matirx += score_matix * weight
    ensemble_score_matirx_sort_index = np.argsort(-ensemble_score_matirx, axis=1)
    ensemble_score_matirx_top10_index = ensemble_score_matirx_sort_index[:, :10]
    t_pred_top10 = np.take_along_axis(candidate, ensemble_score_matirx_top10_index, axis=1)
    return t_pred_top10


def normal_score_matrix(candidate, score_matrix):
    score_matrix_max = score_matrix[candidate!=-1].max()
    score_matrix_min = score_matrix[candidate!=-1].min()
    score_matrix = (score_matrix-score_matrix_min)/(score_matrix_max-score_matrix_min)
    return score_matrix


def preprocessed_can_score_matrix_list(candidate, score_matrix_list):
    # 将所有的score缩放到[0~1]
    for i in range(len(score_matrix_list)):
        score_matrix_list[i] = normal_score_matrix(candidate, score_matrix_list[i])
    # 合并相同节点的得分
    can_unique = None
    for i in range(len(score_matrix_list)):
        can_unique, score_matrix_list[i] = aggregate_score_on_can(candidate, score_matrix_list[i])
    return can_unique, score_matrix_list

def grid_search(val_t, val_can, score_matrix_list, weight_list, num, record_queue):
    np.random.seed(os.getpid())
    tmp_record_dict = dict()
    weight_np = np.array(weight_list)
    for i in range(0, num):
        # get weight
        index = np.random.choice(weight_np.shape[1], weight_np.shape[0])
        weight = weight_np[range(weight_np.shape[0]), index]
        weight_tuple = tuple(weight)
        while weight_tuple in tmp_record_dict:  # 判断weight组合是否已经用过，若用过则再生成
            index = np.random.choice(weight_np.shape[1], weight_np.shape[0])
            weight = weight_np[range(weight_np.shape[0]), index]
            weight = set(weight)
        weight_tuple = tuple(weight)

        t_pred_top10 = get_t_pred_top10(val_can, score_matrix_list, weight)
        input_dict = {}
        # input_dict['h,r->t'] = {'t_correct_index': correct_index, 't_pred_top10': ensemble_score_index}
        input_dict['h,r->t'] = {'t_pred_top10': t_pred_top10, 't': val_t}
        evaluator = WikiKG90Mv2Evaluator()
        ret = evaluator.eval(input_dict)
        print(f"weight: {weight}, mrr: {ret['mrr']}")
        # put weight in the tmp_record_dict
        tmp_record_dict[weight_tuple] = ret['mrr']
    print("len of tmp_record_dict", len(tmp_record_dict))
    record_queue.put(tmp_record_dict)


def get_test_submit_dict(candidate, test_score_matrix_list, weight_list):
    t_pred_top10 = get_t_pred_top10(candidate, test_score_matrix_list, weight_list)
    test_submit_dict = {}
    test_submit_dict['h,r->t'] = {'t_pred_top10': t_pred_top10}
    return test_submit_dict


if __name__ == "__main__":
    args = _get_args()
    # result save path
    def path_grid_json(best_val_mrr): return os.path.join(args.folder_path_ensemble_rst, f"grid_search_{best_val_mrr}.pkl")

    def path_test_submit_dict(best_val_mrr): return os.path.join(
        args.folder_path_ensemble_rst, f"test_submit_dict_mrrOnVal_{best_val_mrr}.pkl")

    def path_valid_submit_dict(best_val_mrr): return os.path.join(
        args.folder_path_ensemble_rst, f"valid_submit_dict_mrrOnVal_{best_val_mrr}.p l")

    # init weight_search_range
    weight_search_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    print(f"weight search range {weight_search_range}")

    # load all data
    tmp_val_infer_rst = torch.load(args.infer_rst_val_path_list[0])
    tmp_test_infer_rst = torch.load(args.infer_rst_test_path_list[0])

    val_t = tmp_val_infer_rst['t']
    print("val_t.shape:", val_t.shape)
    val_can = tmp_val_infer_rst['t_candidate']
    print("val_can.shape", val_can.shape)
    test_can = tmp_test_infer_rst['t_candidate']
    print("test_can.shape", test_can.shape)

    val_score_matrix_list = []
    weight_aviable_list = []
    test_score_matrix_list = []

    # get val_score_matrix_list
    for val_score_matrix_path in args.infer_rst_val_path_list:
        start = torch.load(val_score_matrix_path)
        if 'h,r->t' in start:
            start = start['h,r->t']['t_pred_score'].numpy()
        else:
            start = start['t_pred_score'].numpy()
        val_score_matrix_list.append(start)
        weight_aviable_list.append(weight_search_range)
        print('valid score_matrix_path:', val_score_matrix_path, 'weight:', weight_search_range, 'shpae:', start.shape)
    print('valid score matrix load done, matrix_num:', len(val_score_matrix_list))

    # get test_score_matrix_list
    for test_score_matrix_path in args.infer_rst_val_path_list:
        # 通过valid path，获得test path
        dir_name, file_name = os.path.split(test_score_matrix_path)
        file_name = file_name.replace("valid", "test")
        test_score_matrix_path = os.path.join(dir_name, file_name)
        start = torch.load(test_score_matrix_path)
        if 'h,r->t' in start:
            start = start['h,r->t']['t_pred_score'].numpy()
        else:
            start = start['t_pred_score'].numpy()
        test_score_matrix_list.append(start)
        print('test score_matrix_path:', test_score_matrix_path, 'shpae:', start.shape)

    # preprocessed matrix
    val_can, val_score_matrix_list = preprocessed_can_score_matrix_list(val_can, val_score_matrix_list)
    test_can, test_score_matrix_list = preprocessed_can_score_matrix_list(test_can, test_score_matrix_list)

    # search
    record_queue = multiprocessing.Manager().Queue()
    process_list = []
    for i in range(args.num_proc):
        process = multiprocessing.Process(target=grid_search, args=(
            val_t, val_can, val_score_matrix_list, weight_aviable_list, args.num_per_proc, record_queue,))
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()
    print(record_queue.qsize())
    record_dict = dict()
    while not record_queue.empty():
        record = record_queue.get()
        record_dict.update(record)

    # get the best result
    best_item = max(record_dict.items(), key=operator.itemgetter(1))
    best_weight_tuple = best_item[0]
    best_val_mrr = best_item[1]
    print(f"best weight: {best_weight_tuple}")
    print(f"best val mrr: {best_val_mrr}")

    # save test submit result
    os.makedirs(args.folder_path_ensemble_rst, exist_ok=True)
    test_submit_dict = get_test_submit_dict(test_can, test_score_matrix_list, best_weight_tuple)
    with open(path_test_submit_dict(best_val_mrr), "wb") as f:
        pickle.dump(test_submit_dict, f)

    # save valid submit result
    val_submit_dict = get_test_submit_dict(val_can, val_score_matrix_list, best_weight_tuple)
    with open(path_valid_submit_dict(best_val_mrr), "wb") as f:
        pickle.dump(val_submit_dict, f)

    with open(path_grid_json(best_val_mrr),"wb") as file_obj:
        pickle.dump(record_dict, file_obj)

    # double check
    val_submit_dict['h,r->t']['t']=val_t
    evaluator = WikiKG90Mv2Evaluator()
    ret = evaluator.eval(val_submit_dict)
    print(f"double check best mrr: {ret['mrr']}")