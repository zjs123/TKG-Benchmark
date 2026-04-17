import torch
from torch.utils.data import Dataset
from util import flatten
class TKGTestDataset(Dataset):
    def __init__(self, test_st_tuples: list, st2pos: dict, nentity, nrelation):
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
        return torch.LongTensor(s_list), torch.LongTensor(t_list), pos_list