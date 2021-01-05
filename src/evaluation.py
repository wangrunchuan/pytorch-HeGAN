from typing import Dict
from torch.nn import Embedding
from components import *
from args import args
from dataset import read_graph
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class dblp_evaluation:
    def __init__(self, graph: Dict, batch_size=16):
        self.graph = graph

        self.link_label = self.get_triple()
        self.dataset = EvaluationDataset(self.link_label)

        self.batch_size = batch_size

    def get_triple(self):
        link_label = []
        for source_node, relation in self.graph.items():
            for label, target_nodes in relation.items():
                for target_node in target_nodes:
                    link_label.append([source_node, target_node, label])
        return link_label

    def link_prediction(self, node_embed: Embedding, relation_embed: Embedding):

        node_embed_size = node_embed.weight.shape[1]
        relation_embed = relation_embed.weight.reshape((-1, node_embed_size, node_embed_size))

        correct = 0
        total = 0
        dataloader = DataLoader(self.dataset, shuffle=False, batch_size=self.batch_size)
        iterator = tqdm(dataloader, desc="Evaluation")
        for (batch_idx, data) in (enumerate(iterator)):
            src_idx = data[0].cuda()
            tgt_idx = data[1].cuda()
            label = data[2]

            src_embed = node_embed(src_idx)  # [bs, 64]
            tgt_embed = node_embed(tgt_idx)  # [bs, 64]
            tgt_embed = tgt_embed.reshape((-1, 1, node_embed_size))  # [bs, 1, 64]
            temp = torch.matmul(src_embed, relation_embed).permute(1, 0, 2)
            score = torch.mul(temp, tgt_embed).sum(2)
            # predict = torch.max(score, 1)[1]
            predict = torch.argmax(score, dim=1).cpu()
            correct += predict.eq(label).sum().item()
            total += predict.size(0)

        return correct / total


class EvaluationDataset(Dataset):
    def __init__(self, link_table):
        self.src_nodes = []
        self.tgt_nodes = []
        self.labels = []
        for (src_node, tgt_node, label) in link_table:
            self.src_nodes.append(src_node)
            self.tgt_nodes.append(tgt_node)
            self.labels.append(label)

    def __getitem__(self, item):
        return self.src_nodes[item], self.tgt_nodes[item], self.labels[item]

    def __len__(self):
        return len(self.src_nodes)


def do_evaluation(model_path='../model'):
    # args.pretrain_embed = np.random.rand(args.node_size, args.node_embed_size)
    discriminator = Discriminator(args, model_path=model_path).cuda()
    node_size, relation_size, graph = read_graph('../data/DBLP/dblp_triple.dat')
    evaluator = dblp_evaluation(graph)
    evaluator.link_prediction(discriminator.node_embed, discriminator.relation_embed)


# do_evaluation()
