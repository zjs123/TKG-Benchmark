import torch
from torch.utils.data import Dataset
import pickle
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Dict, Any, Set, Tuple
import argparse

MAX_HITS = 500
FORMAT_CONFIG = ["%Y", "%Y-%m", "%Y-%m-%d"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="stamp", type=str) # stamp or span_end
    parser.add_argument("--model", default="chatgpt", type=str)
    parser.add_argument(
        "--dataset",
        choices=["FinWiki","Wikidata", "ICEWSWiki", "YAGO", "ICEWS14", "ICEWS0515", "ICEWS18", "GDELT", "YAGO11K", "Wiki12K"],
        default="FinWiki",
        type=str,
    )
    parser.add_argument(
        "--multi_step", default=False, action="store_true"
    )  # inference in multi_step
    # History Modeling
    parser.add_argument(
        "--history_type", choices=["entity", "pair"], default="entity", type=str
    )  # history type
    parser.add_argument(
        "--history_direction", choices=["uni", "bi"], default="uni", type=str
    )  # history type
    parser.add_argument("--history_len", default=10, type=int)  # length of history
    parser.add_argument("--history_top_k", default=0, type=int)  # length of targets from history
    # Prompt Construction
    parser.add_argument("--label", default=False, action="store_true")  # express prompt with label
    parser.add_argument(
        "--text_style", default=False, action="store_true"
    )  # express prompt in text
    parser.add_argument(
        "--no_entity", default=False, action="store_true"
    )  # express prompt without entity
    parser.add_argument("--sys_instruction", default="", type=str)  # system instcution for ChatGPT
    parser.add_argument(
        "--no_time", default=False, action="store_true"
    )  # express prompt without time
    parser.add_argument("--shuffle_history", default=False, action="store_true")  # shuffle history
    # Hyperparameter
    parser.add_argument("--top_k", default=100, type=int)  # number of predictions to store
    parser.add_argument(
        "--dec_cand", default=5, type=int
    )  # number of candidates to decode at each step
    parser.add_argument("--max_length", default=1, type=int)  # max decoding length
    parser.add_argument("--world_size", default=1, type=int)  # number of chunks
    parser.add_argument("--rank", default=0, type=int)  # rankd of the executor
    parser.add_argument(
        "--tokenizer_revision", default="main", type=str
    )  # change tokenizer revision (for llama)
    parser.add_argument(
        "--fp16", default=False, action="store_true"
    )  # use float16 instead of float32
    parser.add_argument("--verbose", default=False, action="store_true")  # print extra information
    # Evaluation
    parser.add_argument(
        "--eval_filter",
        choices=["none", "static", "time-aware"],
        type=str,
        default="none",
    )

    args = parser.parse_args()
    assert args.label or not args.no_entity
    return args

def load_tkg_dataset(args: argparse.Namespace) -> Tuple[List[Tuple[int, str]], Dict[Tuple[int, str], Set[Tuple[int, int]]], int, int]:
    if args.dataset in []:
        pkl_path = f"/home/zhangjs/experiments/TKGbenchmark/datasets/original/{args.dataset}/train_valid_test.pkl"
    else:
        pkl_path = f"/home/zhangjs/experiments/TKGbenchmark/datasets/{args.dataset}/train_valid_test.pkl"
    
    train_split_time, valid_split_time, train_data, valid_data, test_data, entity_2_des = pickle.load(open(pkl_path, 'rb'))

    entity_dictionary: Dict[Any, int] = {}
    relation_dictionary: Dict[Any, int] = {}
    ent_num, rel_num = 0, 0

    for data_split in [train_data, valid_data, test_data]:
        for fact in data_split:
            if len(fact) == 4 or len(fact) == 5:
                s, r, o = fact[0], fact[1], fact[2]
                if s not in entity_dictionary:
                    entity_dictionary[s] = ent_num
                    ent_num += 1
                if o not in entity_dictionary:
                    entity_dictionary[o] = ent_num
                    ent_num += 1
                if r not in relation_dictionary:
                    relation_dictionary[r] = rel_num
                    rel_num += 1

    test_st_tuples: List[Tuple[int, str]] = [] 
    st2pos: Dict[Tuple[int, str], Set[Tuple[int, int]]] = {} 

    for fact in test_data:
        if len(fact) == 4:
            s_ori, r_ori, o_ori, t_ori = fact[0], fact[1], fact[2], str(fact[3])
            
            if '-' in t_ori:
                resolution = len(t_ori.strip().split('-')) - 1
            else:
                resolution = 0
            try:
                time_convert = datetime.strptime(t_ori, FORMAT_CONFIG[resolution])
                t = time_convert.strftime(FORMAT_CONFIG[resolution])
            except:
                t = t_ori

            s_id = entity_dictionary[s_ori]
            r_id = relation_dictionary[r_ori]
            o_id = entity_dictionary[o_ori]

            st_key = (s_id, t)
            if st_key not in test_st_tuples:
                test_st_tuples.append(st_key)
            if st_key not in st2pos:
                st2pos[st_key] = set()
            st2pos[st_key].add((r_id, o_id))

        elif len(fact) == 5:
            s_ori, r_ori, o_ori, t_s_ori, t_e_ori = fact[0], fact[1], fact[2], str(fact[3]), str(fact[4])
            
            if '-' in t_s_ori:
                resolution = len(t_s_ori.strip().split('-')) - 1
            else:
                resolution = 0
            t_s = datetime.strptime(t_s_ori, FORMAT_CONFIG[resolution])
            t_e = datetime.strptime(t_e_ori, FORMAT_CONFIG[resolution])

            current = t_s
            while current <= t_e:
                t = current.strftime(FORMAT_CONFIG[resolution])
                s_id = entity_dictionary[s_ori]
                r_id = relation_dictionary[r_ori]
                o_id = entity_dictionary[o_ori]

                st_key = (s_id, t)
                if st_key not in test_st_tuples:
                    test_st_tuples.append(st_key)
                if st_key not in st2pos:
                    st2pos[st_key] = set()
                st2pos[st_key].add((r_id, o_id))

                if resolution == 1:
                    current += relativedelta(months=1)
                elif resolution == 2:
                    current += relativedelta(days=1)
                elif resolution == 0:
                    current += relativedelta(years=1)

    return test_st_tuples, st2pos, ent_num, rel_num

class TKGTestDataset(Dataset):
    def __init__(self, test_st_tuples: list, st2pos: dict, nentity: int, nrelation: int):
        self.test_st = test_st_tuples
        self.st2pos = st2pos
        self.nentity = nentity
        self.nrelation = nrelation

    def __len__(self):
        return len(self.test_st)

    def __getitem__(self, idx):
        s, t = self.test_st[idx]
        pos_pairs = self.st2pos[(s, t)]
        return s, t, pos_pairs

    @staticmethod
    def collate_fn(data):
        
        s_list = [d[0] for d in data]
        
        t_list = [d[1] for d in data]
        pos_list = [d[2] for d in data]
    
        return torch.LongTensor(s_list), t_list, pos_list


def build_tkg_test_dataset(args: argparse.Namespace = None) -> TKGTestDataset:
    if args is None:
        args = get_args()
    test_st_tuples, st2pos, nentity, nrelation = load_tkg_dataset(args)
    return TKGTestDataset(test_st_tuples, st2pos, nentity, nrelation)
