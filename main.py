import sys
import lib
import torch
import random

from model import Model
from utils import check
from munch import Munch
from pprint import pprint

config = Munch()
config.length = 4
config.batch_size = 13
config.device = torch.device("cuda")
config.lr = 0.001
config.test_every = 100
config.log_folder = "logs"
config.max_steps = 5000


def generate_batch_problem(batch_size, length, sampler):
    """ Returns batched data ready for the model. """
    x_batch, x_img_batch, query_batch, y_batch = [], [], [], []
    for _ in range(batch_size):
        x = torch.randint(low=0, high=10, size=(length, ))
        q_low = random.randint(0, length-1)  # low has to be in [0,8]
        q_high = random.randint(q_low, length)  # high has to be in [q_low, 9]

        # solve
        tree = lib.create_segment_tree(x.tolist())
        y = lib.range_min_query(tree=tree,
                                query_low=q_low,
                                query_high=q_high,
                                low=0,
                                high=length-1,
                                pos=0)
        x_imgs = torch.stack([sampler(digit) for digit in x], dim=0)
        check(x_imgs, [length, 28, 28])

        x_batch.append(x)
        x_img_batch.append(x_imgs)
        query_batch.append((q_low, q_high))
        y_batch.append(torch.tensor(y).long())

    return torch.stack(x_batch), torch.stack(x_img_batch), query_batch, torch.stack(y_batch)


def train(c, model, train_img_sampler, test_img_sampler, optimiser):
    """ Runs train loop. """
    model.train()
    x, x_img, q, y = generate_batch_problem(c.batch_size, c.length, train_img_sampler)
    check(x, [c.batch_size, c.length])
    check(x_img, [c.batch_size, c.length, 28, 28])
    check(y, [c.batch_size])

    res = model(x, x_img, q)



def run_exp(c):
    print("Config:")
    pprint(c)

    # setup problems
    train_x, train_y, test_x, test_y = lib.prepare_mnist_data()
    train_img_sampler = lib.get_conditional_sampler(train_x, train_y)
    test_img_sampler = lib.get_conditional_sampler(test_x, test_y)

    # setup model
    model = Model(use_images=False)

    # setup training
    optimiser = torch.optim.Adam(params=model.parameters(), lr=c.lr)
    try:
        train(c, model, train_img_sampler, test_img_sampler, optimiser)
    except KeyboardInterrupt:
        print("Interrupt! Halting training. ")


def main():
    run_exp(config)


if __name__ == "__main__":
    if len(sys.argv) == 1:  # dev hack to load script in notebook without running.
        main()





