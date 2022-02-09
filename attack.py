from textattack.datasets import Dataset
from textattack.attack_recipes import TextFoolerJin2019
from textattack import Attacker, AttackArgs
import textattack
from modules import EmbeddingLayer
from train_classifier import Model
import dataloader
import os
import torch
import argparse


class CustomWrapper(textattack.models.wrappers.ModelWrapper):
    def __init__(self, model):
        self.model = model

    def __call__(self, list_of_texts):

        batch = dataloader.create_one_batch_x(list_of_texts, self.model.word2id)
        if self.model.dist_embeds:
            output, kl = self.model(batch)
            return output
        else:
            output = self.model(batch)
            return output




def main(args):
    if args.dataset == 'mr':
        train_x, train_y = dataloader.read_corpus('data/mr/train.txt')
        test_x, test_y = dataloader.read_corpus('data/mr/test.txt')
    elif args.dataset == 'imdb':
        train_x, train_y = dataloader.read_corpus(os.path.join('/data/medg/misc/jindi/nlp/datasets/imdb',
                                                               'train_tok.csv'),
                                                  clean=False, MR=True, shuffle=True)
        test_x, test_y = dataloader.read_corpus(os.path.join('/data/medg/misc/jindi/nlp/datasets/imdb',
                                                               'test_tok.csv'),
                                                clean=False, MR=True, shuffle=True)
    else:
        train_x, train_y = dataloader.read_corpus('/afs/csail.mit.edu/u/z/zhijing/proj/to_di/data/{}/'
                                                    'train_tok.csv'.format(args.dataset),
                                                  clean=False, MR=False, shuffle=True)
        test_x, test_y = dataloader.read_corpus('/afs/csail.mit.edu/u/z/zhijing/proj/to_di/data/{}/'
                                                    'test_tok.csv'.format(args.dataset),
                                                clean=False, MR=False, shuffle=True)

    nclasses = max(train_y) + 1

    model = Model(args.embedding, args.d, args.depth, args.dropout, args.cnn, nclasses, args=args).cuda()
    
    class_model = CustomWrapper(model)

    

    attack = TextFoolerJin2019.build(class_model)

    dataset = []

    with open('data/mr/test.txt', 'r') as f:
        for line in f:
            dataset.append((' '.join(line.split(' ')[1:]).replace('\n', ''), int(line.split(' ')[0])))

    """
    with open('yelp_negative_test.txt') as f:
      for line in f:
        dataset.append((line.replace('\n', ' '), 0))

    with open('yelp_positive_test.txt') as f:
      for line in f:
        dataset.append((line.replace('\n', ' '), 1))
    
    """

    attacker = Attacker(attack, textattack.datasets.Dataset(dataset[:1000]), AttackArgs(num_examples=999))
    attacker.attack_dataset()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--cnn", action='store_true', help="whether to use cnn")
    argparser.add_argument("--lstm", action='store_true', help="whether to use lstm")
    argparser.add_argument("--dataset", type=str, default="mr", help="which dataset")
    argparser.add_argument("--embedding", type=str, required=True, help="word vectors")
    argparser.add_argument("--batch_size", "--batch", type=int, default=32)
    argparser.add_argument("--max_epoch", type=int, default=70)
    argparser.add_argument("--d", type=int, default=150)
    argparser.add_argument("--dropout", type=float, default=0.3)
    argparser.add_argument("--depth", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--lr_decay", type=float, default=0)
    argparser.add_argument("--cv", type=int, default=0)
    argparser.add_argument("--save_path", type=str, default='')
    argparser.add_argument("--save_data_split", action='store_true', help="whether to save train/test split")
    argparser.add_argument("--gpu_id", type=int, default=0)
    argparser.add_argument("--kl_weight", type=float, default = 0.001)
    argparser.add_argument("--dist_embeds", action='store_true')

    args = argparser.parse_args()
    # args.save_path = os.path.join(args.save_path, args.dataset)
    print (args)
    torch.cuda.set_device(args.gpu_id)
    main(args)
