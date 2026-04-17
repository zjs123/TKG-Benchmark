import argparse
import logging
import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader

from model import KGReasoning
from dataset import TKGTestDataset
from util import set_global_seed, compute_recall_ndcg, ro_to_flat_idx

K = 50  
query_name_dict = {('e', ('r',)): '1p'}
name_answer_dict = {'1p': ['e', ['r',], 'e']}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--kbc_path', type=str, required=True)
    parser.add_argument('--nentity', type=int, required=True)
    parser.add_argument('--nrelation', type=int, required=True)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--fraction', type=int, default=1)
    parser.add_argument('--thrshd', type=float, default=0.001)
    parser.add_argument('--neg_scale', type=int, default=1)
    parser.add_argument('--mask', type=str, default='nomask')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--test_batch_size', default=1, type=int)
    parser.add_argument('--seed', default=12345, type=int)
    return parser.parse_args()

def load_test_data(data_path, nrelation):
    st2pos = defaultdict(set)
    with open(os.path.join(data_path, 'test.txt'), 'r') as f:
        for line in f:
            s, r, o, t = map(int, line.strip().split())
            st2pos[(s, t)].add((r, o))
    test_st = list(st2pos.keys())
    return test_st, st2pos

def read_adj_list(data_path, nrelation):
    adj_list = [[] for _ in range(nrelation)]
    for name in ['train.txt', 'valid.txt', 'test.txt']:
        path = os.path.join(data_path, name)
        if not os.path.exists(path): continue
        with open(path) as f:
            for line in f:
                h, r, t = map(int, line.strip().split())
                adj_list[r].append((h, t))
    freq = [defaultdict(int) for _ in range(nrelation)]
    return adj_list, freq

@torch.no_grad()
def evaluate_s_query(model, s, t, n_entity, n_relation, device):
    total_scores = torch.zeros(n_relation * n_entity, device=device)
    
    # 遍历所有关系 r
    for r in range(n_relation):
        query = torch.LongTensor([s, r]).unsqueeze(0).to(device)
        query_structure = ('e', ('r',))
        
        embedding, _, _ = model.embed_query(query, query_structure, 0)
        o_scores = embedding.squeeze(0) 
        
        start_idx = r * n_entity
        end_idx = start_idx + n_entity
        total_scores[start_idx:end_idx] = o_scores
    
    return total_scores

def test(model, test_loader, args, device):
    total_recall = 0.0
    total_ndcg = 0.0
    num_samples = 0

    logging.info(f"\n开始测试 (s,?,?,t) 任务 | 指标：Recall@{K}、NDCG@{K}")
    for s_batch, t_batch, pos_batch in tqdm(test_loader):
        s = s_batch[0].item()
        t = t_batch[0].item()
        pos_pairs = pos_batch[0]

        scores = evaluate_s_query(
            model=model,
            s=s, t=t,
            n_entity=args.nentity,
            n_relation=args.nrelation,
            device=device
        )

        recall, ndcg = compute_recall_ndcg(scores, pos_pairs, k=K)

        total_recall += recall
        total_ndcg += ndcg
        num_samples += 1

    avg_recall = total_recall / num_samples
    avg_ndcg = total_ndcg / num_samples

    logging.info("="*50)
    logging.info(f"测试结果 (s,?,?,t) 任务")
    logging.info(f"Recall@{K}: {avg_recall:.4f}")
    logging.info(f"NDCG@{K}: {avg_ndcg:.4f}")
    logging.info("="*50)
    
    return avg_recall, avg_ndcg

def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = args.device

    test_st, st2pos = load_test_data(args.data_path, args.nrelation)
    adj_list, freq = read_adj_list(args.data_path, args.nrelation)

    model = KGReasoning(
        args=args,
        device=device,
        adj_list=adj_list,
        query_name_dict=query_name_dict,
        name_answer_dict=name_answer_dict,
        freq=freq
    ).to(device)
    model.eval()

    test_dataset = TKGTestDataset(test_st, st2pos, args.nentity, args.nrelation)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=TKGTestDataset.collate_fn)

    if args.do_test:
        test(model, test_loader, args, device)

if __name__ == '__main__':
    main()