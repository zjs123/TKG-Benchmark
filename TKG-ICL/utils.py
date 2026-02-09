import argparse
from dataclasses import dataclass
import json
import os
import pickle
import random
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Any, Dict, List, Optional

import math

MAX_HITS = 500
FORMAT_CONFIG = ["%Y", "%Y-%m", "%Y-%m-%d"]


@dataclass
class HitsMetric:
    total: int = 0
    hit1: int = 0
    hit50: int = 0
    hit100: int = 0
    hit500: int = 0
    MRR: float = 0.0

    def update(self, rank):
        if rank <= 1:
            self.hit1 += 1
        if rank <= 50:
            self.hit50 += 1
        if rank <= 100:
            self.hit100 += 1
        if rank <= 500:
            self.hit500 += 1
            
        self.MRR += float(1/float(rank))

    def dump(self):
        if self.total == 0:
            return{
            "total": 0,
            "hit1": 0,
            "hit50": 0,
            "hit100": 0,
            "hit500": 0,
            "MRR": 0
        }

        return {
            "total": self.total,
            "hit1": self.hit1 / self.total,
            "hit50": self.hit50 / self.total,
            "hit100": self.hit100 / self.total,
            "hit500": self.hit500 / self.total,
            "MRR": self.MRR / self.total
        }

@dataclass
class GenerativeMetric:
    total: int = 0
    recall50: int = 0
    recall100: int = 0
    recall500: int = 0

    ndcg50: int = 0
    ndcg100: int = 0
    ndcg500: int = 0
    

    def update(self, grountruth_ro, predict_ro):
        self.total += 1
        self.recall50 += float(len(grountruth_ro & set(list(predict_ro)[:50])))/float(len(grountruth_ro))
        self.recall100 += float(len(grountruth_ro & set(list(predict_ro)[:100])))/float(len(grountruth_ro))
        self.recall500 += float(len(grountruth_ro & set(list(predict_ro)[:500])))/float(len(grountruth_ro))

        dcg_50 = 0
        for i in range(len(list(predict_ro)[:50])):
            if list(predict_ro)[:50][i] in grountruth_ro:
                dcg_50 += 1/float(math.log2(i+1+1))
        
        dcg_100 = 0
        for i in range(len(list(predict_ro)[:100])):
            if list(predict_ro)[:100][i] in grountruth_ro:
                dcg_100 += 1/float(math.log2(i+1+1))

        dcg_500 = 0
        for i in range(len(list(predict_ro)[:500])):
            if list(predict_ro)[:500][i] in grountruth_ro:
                dcg_500 += 1/float(math.log2(i+1+1))
        
        idcg = 0
        for i in range(len(grountruth_ro)):
            idcg += 1/float(math.log2(i+1+1))
        
        self.ndcg50 += dcg_50 / idcg
        self.ndcg100 += dcg_100 / idcg
        self.ndcg500 += dcg_500 / idcg

    def dump(self):
        if self.total == 0:
            return{
            "total": 0,
            "recall50": 0,
            "recall100": 0,
            "recall500": 0,
            "ndcg50": 0,
            "ndcg100": 0,
            "ndcg500": 0,
        }

        return {
            "total": self.total,
            "recall50": float(self.recall50)/float(self.total),
            "recall100": float(self.recall100)/float(self.total),
            "recall500": float(self.recall500)/float(self.total),
            "ndcg50": float(self.ndcg50)/float(self.total),
            "ndcg100": float(self.ndcg100)/float(self.total),
            "ndcg500": float(self.ndcg500)/float(self.total),
        }

class MAEMetric:
    total: set = set()
    hit1: int = 0
    MAE: float = 0.0

    def update(self, test_fact, span):
        if span == 0:
            self.hit1 += 1
        
        self.MAE += float(span)
        self.total.add(tuple(test_fact))

    def dump(self):
        if len(self.total) == 0:
            return{
            "total": 0,
            "hit1": 0,
            "MAE": 0
        }
        return {
            "total": len(self.total),
            "hit1": self.hit1 / len(self.total),
            "MAE": self.MAE / len(self.total)
        }


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


# Read entity2id, relation2id
def load_dictionary(in_path: str, file_name: str) -> Dict[int, str]:
    _dict = {}
    with open(os.path.join(in_path, file_name), "r", encoding="utf-8") as fr:
        for line in fr:
            line_split = line.split("\t")
            node = line_split[0]
            index = int(line_split[1])

            _dict[index] = node
    return _dict


# Read train, valid data to construct search space
def load_quadruples(search_dictionary, background_space, file, entity_dictionary, relation_dictionary, query):
    for fact in file:
        if len(fact) == 4:
            if '-' in str(fact[3]):
                resolution = len(str(fact[3]).strip().split('-'))-1
            else:
                resolution = 0
            s, r, o, t = entity_dictionary[fact[0]], relation_dictionary[fact[1]], entity_dictionary[fact[2]], str(fact[3])
            try:
                time_convert = datetime.strptime(t, FORMAT_CONFIG[resolution])
                time_convert = time_convert.strftime(FORMAT_CONFIG[resolution])
            except:
                time_convert = t
            if s not in search_dictionary:
                search_dictionary[s] = {}
            if time_convert not in search_dictionary[s]:
                search_dictionary[s][time_convert] = {}
            if r not in search_dictionary[s][time_convert]:
                search_dictionary[s][time_convert][r] = []
            search_dictionary[s][time_convert][r].append(o)
                
        if len(fact) == 5:
            if '-' in str(fact[3]):
                resolution = len(str(fact[3]).strip().split('-'))-1
            else:
                resolution = 0
            s, r, o, t_s, t_e = entity_dictionary[fact[0]], relation_dictionary[fact[1]], entity_dictionary[fact[2]], str(fact[3]), str(fact[4])
            time_start_convert = datetime.strptime(t_s, FORMAT_CONFIG[resolution])
            time_end_convert = datetime.strptime(t_e, FORMAT_CONFIG[resolution])
            current = time_start_convert
            while current <= time_end_convert:
                current_str = current.strftime(FORMAT_CONFIG[resolution])
                if current_str not in background_space.keys():
                    background_space[current_str] = []
                background_space[current_str].append([s, r, o])
                # 根据格式递增
                if resolution == 1:
                    current += relativedelta(months=1)
                elif resolution == 2:
                    current += relativedelta(days=1)
                elif resolution == 0:
                    current += relativedelta(years=1)


# Read test data to inferencee
def load_quadruples_for_test(seen_e, file, background_space, entity_dictionary, relation_dictionary, t_s_ro) -> List[List[Any]]:
    test_instances = []
    for fact in file:
        if len(fact) == 4:
            if '-' in str(fact[3]):
                resolution = len(str(fact[3]).strip().split('-'))-1
            else:
                resolution = 0
            s, r, o, t = entity_dictionary[fact[0]], relation_dictionary[fact[1]], entity_dictionary[fact[2]], str(fact[3])
            #if s in seen_e:
            #    continue
            try:
                time_convert = datetime.strptime(t, FORMAT_CONFIG[resolution])
                time_convert = time_convert.strftime(FORMAT_CONFIG[resolution])
            except:
                time_convert = t
            test_instances.append([s,r,o,time_convert])
            if time_convert not in t_s_ro.keys():
                t_s_ro[time_convert] = {}
            if s not in t_s_ro[time_convert].keys():
                t_s_ro[time_convert][s] = set()
            t_s_ro[time_convert][s].add((r,o))
                
        if len(fact) == 5:
            if '-' in str(fact[3]):
                resolution = len(str(fact[3]).strip().split('-'))-1
            else:
                resolution = 0
            s, r, o, t_s, t_e = entity_dictionary[fact[0]], relation_dictionary[fact[1]], entity_dictionary[fact[2]], str(fact[3]), str(fact[4])
            time_start_convert = datetime.strptime(t_s, FORMAT_CONFIG[resolution])
            time_end_convert = datetime.strptime(t_e, FORMAT_CONFIG[resolution])
            current = time_start_convert
            while current <= time_end_convert:
                current_str = current.strftime(FORMAT_CONFIG[resolution])
                if current_str not in background_space.keys():
                    background_space[current_str] = []
                background_space[current_str].append([s, r, o])
                # 根据格式递增
                if resolution == 1:
                    current += relativedelta(months=1)
                elif resolution == 2:
                    current += relativedelta(days=1)
                elif resolution == 0:
                    current += relativedelta(years=1)
    
    return test_instances


def format_data(data):
    ro_prediction = {}
    for head, rel, tail, time in data:
        ro_key = (head, time)
        if ro_key not in ro_prediction:
            ro_prediction[ro_key] = []
        ro_prediction[ro_key].append((rel, tail))

    formatted_data = list(sorted([[k[0], list(set(v)), k[1]] for k, v in ro_prediction.items()], key=lambda x: x[2]))
    return formatted_data

def load_data_span(args: argparse.Namespace):
    train_split_time, valid_split_time, train_data, valid_data, test_data, entity_2_des = pickle.load(open("/home/zhangjs/experiments/TKGbenchmark/datasets/"+args.dataset+"/train_valid_test.pkl", 'rb'))
    entity_dictionary, relation_dictionary = {}, {}
    ent_num, rel_num = 0, 0
    all_times = set()
    for file in [train_data, valid_data, test_data]:
        for fact in file:
            if len(fact) == 4:
                s, r, o, t = fact[0], fact[1], fact[2], str(fact[3])
                all_times.add(t)
                if s not in entity_dictionary.keys():
                    entity_dictionary[s] = ent_num
                    ent_num += 1
                if o not in entity_dictionary.keys():
                    entity_dictionary[o] = ent_num
                    ent_num += 1
                if r not in relation_dictionary.keys():
                    relation_dictionary[r] = rel_num
                    rel_num += 1
            if len(fact) == 5:
                s, r, o, t_s, t_e = fact[0], fact[1], fact[2], str(fact[3]), str(fact[4])
                all_times.add(t_s)
                all_times.add(t_e)
                if s not in entity_dictionary.keys():
                    entity_dictionary[s] = ent_num
                    ent_num += 1
                if o not in entity_dictionary.keys():
                    entity_dictionary[o] = ent_num
                    ent_num += 1
                if r not in relation_dictionary.keys():
                    relation_dictionary[r] = rel_num
                    rel_num += 1
    
    sorted_all_timestamp = sorted(list(all_times))
    background_space = {}
    test_sample = {}
    for fact in test_data:
        if len(fact) == 4:
            s, r, o, t = entity_dictionary[fact[0]], relation_dictionary[fact[1]], entity_dictionary[fact[2]], str(fact[3])
            if t not in background_space.keys():
                background_space[t] = []
            background_space[t].append([s, r, o])
        
        if len(fact) == 5:
            s, r, o, t_s, t_e = entity_dictionary[fact[0]], relation_dictionary[fact[1]], entity_dictionary[fact[2]], str(fact[3]), str(fact[4])
            if str(fact[3]) > str(valid_split_time) and str(fact[4]) < str(sorted_all_timestamp[-1]):
                if t_s not in test_sample.keys():
                    test_sample[t_s] = []
                test_sample[t_s].append([s, r, o, t_s, t_e])

    return test_sample, background_space, entity_dictionary, relation_dictionary, entity_2_des
    

def load_data(args: argparse.Namespace):
    if args.dataset in ['ICEWS14','ICEWS18','ICEWS0515','GDELT','Wiki12K','YAGO11K']:
         train_split_time, valid_split_time, train_data, valid_data, test_data, entity_2_des = pickle.load(open("/home/zhangjs/experiments/TKGbenchmark/datasets/original/"+args.dataset+"/train_valid_test.pkl", 'rb'))
    else:
        train_split_time, valid_split_time, train_data, valid_data, test_data, entity_2_des = pickle.load(open("/home/zhangjs/experiments/TKGbenchmark/datasets/"+args.dataset+"/train_valid_test.pkl", 'rb'))
    
    entity_dictionary, relation_dictionary = {}, {}
    seen_e = set()
    ent_num, rel_num = 0, 0
    for file in [train_data, valid_data]:
        for fact in file:
            if len(fact) == 4:
                s, r, o, t = fact[0], fact[1], fact[2], str(fact[3])
                seen_e.add(s)
                if s not in entity_dictionary.keys():
                    entity_dictionary[s] = ent_num
                    ent_num += 1
                if o not in entity_dictionary.keys():
                    entity_dictionary[o] = ent_num
                    ent_num += 1
                if r not in relation_dictionary.keys():
                    relation_dictionary[r] = rel_num
                    rel_num += 1
            if len(fact) == 5:
                s, r, o, t_s, t_e = fact[0], fact[1], fact[2], str(fact[3]), str(fact[4])
                seen_e.add(s)
                if s not in entity_dictionary.keys():
                    entity_dictionary[s] = ent_num
                    ent_num += 1
                if o not in entity_dictionary.keys():
                    entity_dictionary[o] = ent_num
                    ent_num += 1
                if r not in relation_dictionary.keys():
                    relation_dictionary[r] = rel_num
                    rel_num += 1
    
    for file in [test_data]:
        for fact in file:
            if len(fact) == 4:
                s, r, o, t = fact[0], fact[1], fact[2], str(fact[3])
                seen_e.add(s)
                if s not in entity_dictionary.keys():
                    entity_dictionary[s] = ent_num
                    ent_num += 1
                if o not in entity_dictionary.keys():
                    entity_dictionary[o] = ent_num
                    ent_num += 1
                if r not in relation_dictionary.keys():
                    relation_dictionary[r] = rel_num
                    rel_num += 1
            if len(fact) == 5:
                s, r, o, t_s, t_e = fact[0], fact[1], fact[2], str(fact[3]), str(fact[4])
                seen_e.add(s)
                if s not in entity_dictionary.keys():
                    entity_dictionary[s] = ent_num
                    ent_num += 1
                if o not in entity_dictionary.keys():
                    entity_dictionary[o] = ent_num
                    ent_num += 1
                if r not in relation_dictionary.keys():
                    relation_dictionary[r] = rel_num
                    rel_num += 1

    head_search_space = {}
    background_space = {}
    t_s_ro = {}
    
    load_quadruples(
        head_search_space,
        background_space,
        train_data,
        entity_dictionary,
        relation_dictionary,
        query="tail",
    )
    load_quadruples(
        head_search_space,
        background_space,
        valid_data,
        entity_dictionary,
        relation_dictionary,
        query="tail",
    )
    #print(background_space)
    #print(head_search_space)

    test_data = load_quadruples_for_test(
        seen_e,
        test_data,
        background_space,
        entity_dictionary,
        relation_dictionary,
        t_s_ro
    )

    formatted_test_data = format_data(test_data)
    #print(formatted_test_data)

    return formatted_test_data, head_search_space, background_space, entity_dictionary, relation_dictionary, entity_2_des, t_s_ro


def adjust_top_k(test_data, args):
    max_targets_len = max([len(x[1]) for x in test_data])
    args.top_k = max(args.top_k, MAX_HITS, max_targets_len + MAX_HITS)
    if args.verbose:
        print(f"max targets len: {max_targets_len}")
        print(f"adjusted top k: {args.top_k}")


def get_filename(args: argparse.Namespace, is_eval: bool = False):
    model_name = args.model.split("/")[-1]
    filename_args = "_".join(
        [
            model_name,
            args.dataset,
            f"multi_step_{args.multi_step}",
            f"history_len_{args.history_len}",
            f"history_type_{args.history_type}",
            f"history_direction_{args.history_direction}",
            f"no_time_{args.no_time}",
            f"shuffle_history_{args.shuffle_history}",
            f"label_{args.label}",
            f"text_style_{args.text_style}",
            f"no_entity_{args.no_entity}",
            f'world_size_{"*" if is_eval else args.world_size}',
            f'rank_{"*" if is_eval else args.rank}',
        ]
    )
    filename = f"outputs/{filename_args}.jsonl"
    print(f"output file: {filename}")
    return filename


def construct_history_by_search(search_space, entity, history_type):
    if entity not in search_space:
        return {}
    search_graph = {entity: {}}
    search_graph[entity] = search_space[entity]

    return search_graph


def format_history(history_graph, background_space, ent_dict, rel_dict, entity_2_des, history_len, question, args, return_prompt = True):
    reverse_ent_dict = dict(zip(ent_dict.values(), ent_dict.keys()))
    reverse_rel_dict = dict(zip(rel_dict.values(), rel_dict.keys()))
    quadruples = []
    for entity in history_graph:
        for time in history_graph[entity]:
            if time >= question[0]:
                continue
            for relation in history_graph[entity][time]:
                for target in history_graph[entity][time][relation]:
                    quadruples.append([entity, relation, target, time])
    
    back_ground_num = 0
    # id version
    # add span knowledge
    '''
    temp_prompt = "Some facts that valid in "+question[0]+": \n"
    if len(background_space[question[0]]) != 0:
        back_ground_num = 0
        for fact in background_space[question[0]]:
            if question[1] == fact[0]:
                temp_prompt += f"{question[0]}:"
                temp_prompt += f"[{fact[0]},{fact[1]},{fact[2]}]\n"
                back_ground_num += 1
            if back_ground_num >= int(0.5*history_len):
                break
    '''
    if back_ground_num != 0:
        prompt = temp_prompt
    else:
        prompt = ""
    
    prompt += "History: \n"
    quadruples = sorted(quadruples, key = lambda x: x[-1])
    history = quadruples[-history_len:]
    
    for x in history:
        entity, relation, target, time = x[0], x[1], x[2], x[3]
        prompt += f"{time}:"
        prompt += f"[{entity},{relation},{target}]\n"
    
    prompt += "Query: \n"
    prompt += f"{question[0]}:"
    prompt += f"[{question[1]},"
    
    
    back_ground_num = 0
    # text version
    '''
    temp_prompt = "Some facts that valid in "+question[0]+": \n"
    for fact in background_space[question[0]]:
        if question[1] == fact[0]:
            temp_prompt += f"{question[0]}:"
            temp_prompt += f"[{fact[0]}.{entity_2_des[reverse_ent_dict[fact[0]]]},{fact[1]}.{reverse_rel_dict[fact[1]]},{fact[2]}.{entity_2_des[reverse_ent_dict[fact[2]]]}]\n"
            back_ground_num += 1
        if back_ground_num >= int(0.5*history_len):
            break
    
    if back_ground_num != 0:
        prompt = temp_prompt
    else:
        prompt = ""
    
    prompt += "History: \n"
    quadruples = sorted(quadruples, key = lambda x: x[-1])
    history = quadruples[-history_len:]
    for x in history:
        entity, relation, target, time = x[0], x[1], x[2], x[3]
        prompt += f"{time}:"
        prompt += f"[{entity}.{entity_2_des[reverse_ent_dict[entity]]},{relation}.{reverse_rel_dict[relation]},{target}.{entity_2_des[reverse_ent_dict[target]]}]\n"
    
    prompt += "Query: \n"
    prompt += f"{question[0]}:"
    prompt += f"[{question[1]}.{entity_2_des[reverse_ent_dict[question[1]]]},"
    '''

    print(prompt)
    return prompt


def prepare_input(x, entity_search_space, background_space, ent_dict, rel_dict, entity_2_des, args, return_prompt: bool = True):
    entity, time = x[0], x[2]
    entity_history = construct_history_by_search(
        entity_search_space,
        entity=entity,
        history_type=args.history_type,
    )
    #print(entity_history)
    history_input = format_history(
        entity_history,
        background_space,
        ent_dict,
        rel_dict,
        entity_2_des, 
        args.history_len,
        [time, entity],
        args=args,
        return_prompt=return_prompt,
    )

    return history_input

def prepare_input_span(test_fact, timestamp, background_space, ent_dict, rel_dict, entity_2_des, args, return_prompt: bool = True):
    reverse_ent_dict = dict(zip(ent_dict.values(), ent_dict.keys()))
    reverse_rel_dict = dict(zip(rel_dict.values(), rel_dict.keys()))
    prompt = "The knowledge you need to determine is " + f"[{test_fact[0]}.{entity_2_des[reverse_ent_dict[test_fact[0]]]},{test_fact[1]}.{reverse_rel_dict[test_fact[1]]},{test_fact[2]}.{entity_2_des[reverse_ent_dict[test_fact[2]]]}], which start at "+f"{test_fact[3]}"+"\n"
    #prompt = "The knowledge you need to determine is " + f"[{test_fact[0]},{test_fact[1]},{test_fact[2]}], which start at "+f"{test_fact[3]}"+"\n"
    prompt += "The current timestamp is "+f"{timestamp}"+" \n"
    tmp_prompt = "other facts that occurred within the current timestamp: \n"
    
    back_ground_num = 0
    
    for background_fact in background_space[timestamp]:
        if test_fact[0] == background_fact[0] or test_fact[2] == background_fact[2]:
            #tmp_prompt += f"[{background_fact[0]},{background_fact[1]},{background_fact[2]}]\n"
            tmp_prompt += f"[{background_fact[0]}.{entity_2_des[reverse_ent_dict[background_fact[0]]]},{background_fact[1]}.{reverse_rel_dict[background_fact[1]]},{background_fact[2]}.{entity_2_des[reverse_ent_dict[background_fact[2]]]}]\n"
            back_ground_num += 1
        if back_ground_num >= 5:
            break
    
    if back_ground_num != 0:
        prompt += tmp_prompt
    prompt += "Answer: "

    print(prompt)
    return prompt


def update_history(x, entity_search_space, predictions, args):
    s, r_o_list, t = x[0], x[1], x[2]
    if args.verbose:
        print(
            f"search space:\n{entity},{relation},{time} --> {entity_search_space[entity][time][relation]}"
        )
    for ro in r_o_list:
        r, o = ro[0], ro[1]
        if s not in entity_search_space:
            entity_search_space[s] = {}
        if t not in entity_search_space[s]:
            entity_search_space[s][t] = {}
        if r not in entity_search_space[s][t]:
            entity_search_space[s][t][r] = []
        entity_search_space[s][t][r].append(o)
    if args.verbose:
        print(f"history:\n{entity},{relation},{time} --> {targets}")
        print(
            f"search space:\n{entity},{relation},{time} --> {entity_search_space[entity][time][relation]}"
        )


def write_results(x, predictions, t_s_ro, writer, args):
    print(x)
    print(predictions)
    print(list(t_s_ro[x[2]][x[0]]))
    print(x[1])
    if predictions == None or len(predictions) == 0:
        return {}
    entity, relation_targets, time = x[0], x[1], x[2]
    example = {
        "timestamp": time,
        "entity": entity,
        "relation_targets": list(relation_targets),
        "predictions": list(set([(x[0],x[1]) for x in predictions if len(x) == 2])),
    }
    writer.write(json.dumps(example) + "\n")

    if args.verbose:
        print(f"example:\n{json.dumps(example, indent=2)}")

    return example


def update_metric(example, metric, args):
    '''
    if "predictions" in example.keys():
        if args.verbose:
            print(f'predictions: {example["predictions"]}')
        for ro in example["relation_targets"]:
            metric.total += 1
            index = example["predictions"].index(ro) if ro in example["predictions"] else -1
            if index >= 0:
                _predictions = [
                    x for x in example["predictions"][:index]
                ]
                rank = len(_predictions) + 1
                print(f"target: {ro} --> rank: {rank}")
                metric.update(rank)
    '''
    if "predictions" in example.keys():
        if args.verbose:
            print(f'predictions: {example["predictions"]}')
        grountruth_ro = set(example["relation_targets"])
        predict_ro = set(example["predictions"])
        metric.update(grountruth_ro, predict_ro)


def update_metric_span(test_fact, metric, timestamp, args):
    if '*' in timestamp:
        test_time_len = int(timestamp.strip().replace('*', ''))
        metric.update(test_fact, test_time_len)
    else:
        if '-' in str(test_fact[3]):
            resolution = len(str(test_fact[3]).strip().split('-'))-1
        else:
            resolution = 0
        truth_end_convert = datetime.strptime(str(test_fact[4]), FORMAT_CONFIG[resolution])
        predict_end_convert = datetime.strptime(str(timestamp), FORMAT_CONFIG[resolution])
        
        if truth_end_convert == predict_end_convert:
            span = 0
        else:
            if truth_end_convert < predict_end_convert:
                min_stamp = truth_end_convert
                max_stamp = predict_end_convert
            else:
                min_stamp = predict_end_convert
                max_stamp = truth_end_convert
        
            span = 0
            current = min_stamp
            while current < max_stamp:
                # 根据格式递增
                if resolution == 1:
                    current += relativedelta(months=1)
                elif resolution == 2:
                    current += relativedelta(days=1)
                elif resolution == 0:
                    current += relativedelta(years=1)
                span += 1

        metric.update(test_fact, span)
