import argparse
import datetime
import logging
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn as nn
import sys

sys.path.append('.')
from torch.utils.data import DataLoader
from args import args
from dataset import *
from components import *
from evaluation import dblp_evaluation

import math

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


class HeGAN(object):
    def __init__(self, args, model_path=None):
        self.args = args
        logging.info("Loading pretrain embedding file...")
        # pretrain_emb = self.read_pretrain_embed('../pretrain/dblp_pre_train.emb', node_size=args.node_size, embed_size=args.node_embed_size)
        pretrain_emb = np.random.rand(args.node_size, args.node_embed_size)
        self.args.pretrain_embed = pretrain_emb
        logging.info("Pretrain embedding file loaded.")

        logging.info("Building Generator...")
        generator = Generator(self.args, model_path)
        self.generator = generator.cuda()

        logging.info("Building Discriminator...")
        discriminator = Discriminator(self.args, model_path)
        self.discriminator = discriminator.cuda()

        node_size, relation_size, graph = read_graph('../data/DBLP/dblp.test_0.2')
        self.evaluator = dblp_evaluation(graph)

        self.name = "HeGAN-" + datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')
        if args.name:
            self.name = args.name + datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S')

    def read_pretrain_embed(self, pretrain_file, node_size, embed_size):
        embedding_matrix = np.random.rand(node_size, embed_size)
        i = -1
        with open(pretrain_file) as infile:
            for line in infile.readlines()[1:]:
                i += 1
                emd = line.strip().split()
                embedding_matrix[int(emd[0]), :] = self.str_list_to_float(emd[1:])
        return embedding_matrix

    def str_list_to_float(self, str_list):
        return [float(item) for item in str_list]

    def train(self):
        writer = SummaryWriter("./log/" + self.name)

        dblp_dataset = DBLPDataset(graph_path='../data/DBLP/dblp.train_0.8')
        gen_data_loader = DataLoader(dblp_dataset.generator_dataset, shuffle=True, batch_size=self.args.batch_size,
                                     num_workers=8, pin_memory=True)
        dis_data_loader = DataLoader(dblp_dataset.discriminator_dataset, shuffle=True, batch_size=self.args.batch_size,
                                     num_workers=8, pin_memory=True)

        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), self.args.dis_lr)
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), self.args.gen_lr)

        best_score = -1.0

        logging.info("Training Begin...")
        for total_idx in range(self.args.epoch):
            # Training discriminator
            for dis_idx in range(self.args.epoch_d):
                dis_batch_loss = 0
                iterator = tqdm(dis_data_loader, desc="Discriminator")
                for (batch_idx, data) in (enumerate(iterator)):
                    pos_node_idx = data[0].cuda()
                    pos_relation_idx = data[1].cuda()
                    pos_node_neighbor_idx = data[2].cuda()
                    """
                    if batch_idx <= 1:
                        for i in range(6):
                            print(self.discriminator.relation_embed(torch.LongTensor([i]).cuda()))
                    """
                    neg_node_idx = data[3].cuda()
                    neg_relation_idx = data[4].cuda()
                    neg_node_neighbor_idx = data[5].cuda()

                    fake_nodes_embed = self.generator.generate_fake_nodes(pos_node_idx, pos_relation_idx)

                    prob_pos = self.discriminator(pos_node_idx, pos_relation_idx, pos_node_neighbor_idx)
                    prob_neg = self.discriminator(neg_node_idx, neg_relation_idx, neg_node_neighbor_idx)
                    prob_fake = self.discriminator.forward_fake(pos_node_idx, pos_relation_idx, fake_nodes_embed)

                    discriminator_loss_pos = -torch.mean(torch.log(prob_pos))
                    discriminator_loss_neg = -torch.mean(torch.log(1-prob_neg))
                    discriminator_loss_fake = -torch.mean(torch.log(1-prob_fake))
                    """
                    if batch_idx <= 1:
                        print(prob_fake)
                        print(discriminator_loss_pos, discriminator_loss_neg, discriminator_loss_fake)
                    """
                    discriminator_loss = discriminator_loss_pos + discriminator_loss_neg + discriminator_loss_fake
                    dis_batch_loss += discriminator_loss.item()

                    discriminator_optimizer.zero_grad()
                    discriminator_loss.backward()
                    discriminator_optimizer.step()

                logging.info("Total epoch: {}, Discriminator epoch: {}, loss: {}.".
                             format(total_idx, dis_idx, dis_batch_loss / len(dis_data_loader)))
                writer.add_scalar("dis_loss", dis_batch_loss / len(dis_data_loader))

            # print(self.discriminator.node_embed(torch.LongTensor([0]).cuda()))
            # print(self.discriminator.node_embed)

            # Training generator
            for gen_idx in range(self.args.epoch_g):
                gen_batch_loss = 0
                iterator = tqdm(gen_data_loader, desc="Generator")
                for (batch_idx, data) in tqdm(enumerate(iterator)):
                    node_idx = data[0].cuda()
                    relation_idx = data[1].cuda()

                    temp = self.discriminator.multify(node_idx, relation_idx)
                    prob = self.generator(node_idx, relation_idx, dis_temp=temp)
                    # print(prob)
                    generator_loss = self.generator.loss(prob)
                    # print(generator_loss)

                    l2_regularization = torch.tensor([0], dtype=torch.float32, device='cuda')
                    for param in self.generator.parameters():
                        l2_regularization += torch.norm(param, 2)
                    """
                    if batch_idx <= 300:
                        print(batch_idx, generator_loss.item(), l2_regularization.item())
                    """
                    if math.isinf(generator_loss.item()):
                        print(batch_idx)
                        print(prob)

                    generator_loss = generator_loss + args.lambda_gen * l2_regularization
                    # print(generator_loss)
                    gen_batch_loss += generator_loss.item()

                    generator_optimizer.zero_grad()
                    generator_loss.backward()
                    generator_optimizer.step()

                logging.info("Total epoch: {}, Generator epoch: {}, loss: {}.".
                             format(total_idx, gen_idx, gen_batch_loss / len(gen_data_loader)))
                writer.add_scalar("gen_loss", gen_batch_loss / len(gen_data_loader))

            # Evaluation
            acc = self.evaluator.link_prediction(self.discriminator.node_embed, self.discriminator.relation_embed)
            logging.info("Total epoch: {}, Evaluation accuracy: {}.".format(total_idx, acc))
            if acc > best_score:
                best_score = acc
                # Save model
                model_path = '../model'
                self.discriminator.save(model_path)
                self.generator.save(model_path)

        writer.close()


def main():
    he_gan = HeGAN(args=args, model_path=None)
    # he_gan = HeGAN(args=args, model_path='../model')
    """
    for i in range(6):
        print(he_gan.discriminator.relation_embed(torch.LongTensor([i]).cuda()))
    """
    he_gan.train()


def test():
    dblp_dataset = DBLPDataset()
    dl = DataLoader(dblp_dataset.discriminator_dataset, shuffle=True, batch_size=args.batch_size)
    for (idx, a) in enumerate(dl):
        print(a)


if __name__ == '__main__':
    # test()
    main()
