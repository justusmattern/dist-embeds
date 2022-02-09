from textattack.datasets import Dataset
from textattack.attack_recipes import TextFoolerJin2019
from textattack import Attacker, AttackArgs


attack = TextFoolerJin2019.build(class_model)

dataset = []

"""
with open('test.txt', 'r') as f:
    for line in f:
        dataset.append((' '.join(line.split(' ')[1:]).replace('\n', ''), int(line.split(' ')[0])))
"""
with open('yelp_negative_test.txt') as f:
  for line in f:
    dataset.append((line.replace('\n', ' '), 0))

with open('yelp_positive_test.txt') as f:
  for line in f:
    dataset.append((line.replace('\n', ' '), 1))

import random
random.shuffle(dataset)

attacker = Attacker(attack, textattack.datasets.Dataset(dataset[:1000]), AttackArgs(num_examples=999))
attacker.attack_dataset()
