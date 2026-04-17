import numpy as np
import random
import torch
import time

def list2tuple(l):
    return tuple(list2tuple(x) if type(x)==list else x for x in l)

def tuple2list(t):
    return list(tuple2list(x) if type(x)==tuple else x for x in t)

def flatten(x):
    if isinstance(x, (list, tuple)):
        return sum([flatten(item) for item in x], [])
    else:
        return [x]

def flatten_query(query):
    return flatten(query)

def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True

def eval_tuple(arg_return):
    if type(arg_return) == tuple:
        return arg_return
    if not arg_return:
        return ()
    if arg_return[0] not in ["(", "["]:
        try:
            arg_return = eval(arg_return)
        except:
            return tuple(arg_return.split(","))
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            item = item.strip()
            if not item:
                continue
            try:
                item = eval(item)
            except:
                pass
            List.append(item)
        arg_return = tuple(List)
    return arg_return

def compute_recall_ndcg(scores: torch.Tensor, pos_pairs: set, k: int = 50):
    n_rel = 10000 
    n_ent = 10000 
    sorted_indices = torch.argsort(scores, descending=True)
    topk_indices = sorted_indices[:k]

    pos_set = set()
    for idx in topk_indices.cpu().numpy():
        r = idx // n_ent
        o = idx % n_ent
        pos_set.add((r, o))

    total_pos = len(pos_pairs)
    hit_pos = len(pos_set & pos_pairs)
    recall = hit_pos / total_pos if total_pos > 0 else 0.0

    dcg = 0.0
    for i, idx in enumerate(topk_indices):
        r = idx // n_ent
        o = idx % n_ent
        if (r, o) in pos_pairs:
            dcg += 1 / np.log2(i + 2) 

    ideal_pos = min(total_pos, k)
    idcg = sum([1 / np.log2(i + 2) for i in range(ideal_pos)])
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return recall, ndcg

def ro_to_flat_idx(r, o, n_entity):
    return r * n_entity + o

def flat_idx_to_ro(idx, n_entity):
    r = idx // n_entity
    o = idx % n_entity
    return r, o