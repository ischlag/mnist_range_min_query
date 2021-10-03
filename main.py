import sys
import lib
import torch
import random
import torch.nn.functional as F

from model import Model
from utils import check
from munch import Munch
from pprint import pprint

config = Munch()
config.length = 6
config.batch_size = 64
config.lr = 0.001
config.max_steps = 5000
config.log_every = 200
config.device = torch.device("cuda")


def generate_batch_problem(batch_size, length, sampler, device):
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

    return torch.stack(x_batch).to(device), \
           torch.stack(x_img_batch).to(device), \
           query_batch, \
           torch.stack(y_batch).to(device)


def test(c, model, sampler, length):
    """ Run a test loop """
    model.eval()
    x, x_img, q, y = generate_batch_problem(c.batch_size, length, sampler, c.device)
    check(x, [c.batch_size, length])
    check(x_img, [c.batch_size, length, 28, 28])
    check(y, [c.batch_size])

    logits = model(x, x_img, q)
    check(logits, [c.batch_size, 10])

    loss = F.cross_entropy(input=logits, target=y)
    acc = (torch.argmax(logits, dim=-1) == y).float().mean()
    print(f"test length {length}: loss={loss:.4f} acc={acc:.4f}")


def train(c, model, train_img_sampler, optimiser):
    """ Runs train loop. """
    model.train()

    for step in range(c.max_steps):
        x, x_img, q, y = generate_batch_problem(c.batch_size, c.length,
                                                train_img_sampler, c.device)
        check(x, [c.batch_size, c.length])
        check(x_img, [c.batch_size, c.length, 28, 28])
        check(y, [c.batch_size])

        logits = model(x, x_img, q)
        check(logits, [c.batch_size, 10])

        loss = F.cross_entropy(input=logits, target=y)
        acc = (torch.argmax(logits, dim=-1) == y).float().mean()

        # step
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # log
        if step % c.log_every == 0:
            print(f"step={step} loss={loss:.4f} acc={acc:.4f}")


def run_exp(c):
    print("Config:")
    pprint(c)
    print()

    # setup problems
    train_x, train_y, test_x, test_y = lib.prepare_mnist_data()
    train_img_sampler = lib.get_conditional_sampler(train_x, train_y)
    test_img_sampler = lib.get_conditional_sampler(test_x, test_y)

    # setup model 1
    print(f"### Train with symbol input of length {c.length}")
    model = Model(use_images=False).to(c.device)

    # setup training
    optimiser = torch.optim.Adam(params=model.parameters(), lr=c.lr)
    try:
        train(c, model, train_img_sampler, optimiser)
    except KeyboardInterrupt:
        print("Interrupt! Halting training. ")
    for i in [6, 10, 15, 20, 25, 30, 40, 50, 60, 100, 200, 300, 500]:
        test(c, model, test_img_sampler, length=i)
    print("\n\n")

    # setup model 2
    print(f"### Train with MNIST input of length {c.length}")
    model = Model(use_images=True).to(c.device)

    # setup training
    optimiser = torch.optim.Adam(params=model.parameters(), lr=c.lr)
    try:
        train(c, model, train_img_sampler, optimiser)
    except KeyboardInterrupt:
        print("Interrupt! Halting training. ")
    for i in [6, 10, 15, 20, 25, 30, 40, 50, 60, 100, 200, 300, 500]:
        test(c, model, test_img_sampler, length=i)
    print("\n\n")


def main():
    run_exp(config)


if __name__ == "__main__":
    if len(sys.argv) == 1:  # dev hack to load script in notebook without running.
        main()
