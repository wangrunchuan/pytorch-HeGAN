from collections import OrderedDict

import torch
import torch.nn as nn
import os


class Discriminator(nn.Module):
    def __init__(self, args, model_path=None):
        super(Discriminator, self).__init__()
        self.args = args

        if model_path:
            node_embed_path = os.path.join(model_path, 'dis_node_embed')
            self.node_embed = torch.load(node_embed_path)

            relation_embed_path = os.path.join(model_path, 'dis_relation_embed')
            self.relation_embed = torch.load(relation_embed_path)

        else:
            self.node_embed = nn.Embedding.from_pretrained(torch.from_numpy(args.pretrain_embed), freeze=False).float()
            self.relation_embed = nn.Embedding(args.relation_size, args.node_embed_size * args.node_embed_size)

            nn.init.xavier_uniform(self.relation_embed.weight)

        self.sigmoid = nn.Sigmoid()

    def forward_fake(self, node_idx, relation_idx, fake_node_embed):
        node_embed = self.node_embed(node_idx)
        node_embed = node_embed.reshape((-1, 1, self.args.node_embed_size))

        relation_embed = self.relation_embed(relation_idx)
        relation_embed = relation_embed.reshape((-1, self.args.node_embed_size, self.args.node_embed_size))

        temp = torch.matmul(node_embed, relation_embed)  # [bs, 1, 64]

        score = torch.sum(torch.mul(temp, fake_node_embed), 2)  # [bs, 1]

        prob = self.sigmoid(score)
        return prob

    def forward(self, node_idx, relation_idx, node_neighbor_idx):
        node_embed = self.node_embed(node_idx)  # [bs, 64]
        node_embed = node_embed.reshape((-1, 1, self.args.node_embed_size))  # [bs, 1, 64]

        relation_embed = self.relation_embed(relation_idx)  # [bs, 64 * 64]
        # print('relation0', relation_embed)
        relation_embed = relation_embed.reshape((-1, self.args.node_embed_size, self.args.node_embed_size))  # [bs, 64, 64]
        # print('node', node_embed)
        # print('relation', relation_embed)
        temp = torch.matmul(node_embed, relation_embed)  # [bs, 1, 64]
        # print('temp', temp)

        neighbor_embed = self.node_embed(node_neighbor_idx)
        neighbor_embed = neighbor_embed.reshape((-1, 1, self.args.node_embed_size))  # [bs, 1, 64]

        score = torch.sum(torch.mul(temp, neighbor_embed), 2)  # [bs, 1]
        # print(score)
        prob = self.sigmoid(score)
        return prob

    def multify(self, node_idx, relation_idx):
        """
        get e_u^D * M_r^b
        :param node_idx:
        :param relation_idx:
        :return:
        """
        node_embed = self.node_embed(node_idx)
        node_embed = node_embed.reshape((-1, 1, self.args.node_embed_size))
        relation_embed = self.relation_embed(relation_idx)
        relation_embed = relation_embed.reshape((-1, self.args.node_embed_size, self.args.node_embed_size))
        temp = torch.matmul(node_embed, relation_embed)
        return temp

    def save(self, model_path):
        node_embed_path = os.path.join(model_path, 'dis_node_embed')
        torch.save(self.node_embed, node_embed_path)

        relation_embed_path = os.path.join(model_path, 'dis_relation_embed')
        torch.save(self.relation_embed, relation_embed_path)


class Generator(nn.Module):
    def __init__(self, args, model_path=None):
        super(Generator, self).__init__()
        self.args = args

        node_embed_size = args.node_embed_size

        if model_path:
            node_embed_path = os.path.join(model_path, 'gen_node_embed')
            self.node_embed = torch.load(node_embed_path)

            relation_embed_path = os.path.join(model_path, 'gen_relation_embed')
            self.relation_embed = torch.load(relation_embed_path)

            fc_path = os.path.join(model_path, 'gen_fc')
            self.fc = torch.load(fc_path)

        else:
            self.node_embed = nn.Embedding.from_pretrained(torch.from_numpy(args.pretrain_embed), freeze=False).float()
            self.relation_embed = nn.Embedding(args.relation_size, args.node_embed_size * args.node_embed_size)

            nn.init.xavier_uniform(self.relation_embed.weight)

            self.fc = nn.Sequential(
                OrderedDict([
                    ("w_1", nn.Linear(node_embed_size, node_embed_size)),
                    ("a_1", nn.LeakyReLU()),
                    ("w_2", nn.Linear(node_embed_size, node_embed_size)),
                    ("a_2", nn.LeakyReLU())
                ])
            )

            nn.init.xavier_uniform(self.fc[0].weight)
            nn.init.xavier_uniform(self.fc[2].weight)

        self.sigmoid = nn.Sigmoid()

    def forward(self, node_idx, relation_idx, dis_temp):
        # dis_temp [bs, 1, 64]
        fake_nodes = self.generate_fake_nodes(node_idx, relation_idx)  # [bs, 1, 64]

        score = torch.sum(torch.mul(dis_temp, fake_nodes), 2)
        prob = self.sigmoid(score)
        return prob

    def loss(self, prob):
        loss = -torch.mean(torch.log(prob)) * (1 - self.args.label_smooth) + -torch.mean(torch.log(1 - prob + 1e-5)) * self.args.label_smooth
        return loss

    def generate_fake_nodes(self, node_idx, relation_idx):
        node_embed = self.node_embed(node_idx)
        node_embed = node_embed.reshape((-1, 1, self.args.node_embed_size))
        relation_embed = self.relation_embed(relation_idx)
        relation_embed = relation_embed.reshape((-1, self.args.node_embed_size, self.args.node_embed_size))
        temp = torch.matmul(node_embed, relation_embed)

        # add noise
        temp = temp + torch.randn(temp.shape, requires_grad=False).cuda()
        output = self.fc(temp)

        return output

    def save(self, model_path):
        node_embed_path = os.path.join(model_path, 'gen_node_embed')
        torch.save(self.node_embed, node_embed_path)

        relation_embed_path = os.path.join(model_path, 'gen_relation_embed')
        torch.save(self.relation_embed, relation_embed_path)

        fc_path = os.path.join(model_path, 'gen_fc')
        torch.save(self.fc, fc_path)
