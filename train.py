import torch
import tqdm

import model
import utils
import vocab


def train_test_split(x, y, test_size=0.2):
    train_x = x[: int(len(x) * (1 - test_size))]
    train_y = y[: int(len(y) * (1 - test_size))]
    test_x = x[int(len(x) * (1 - test_size)) :]
    test_y = y[int(len(y) * (1 - test_size)) :]
    return train_x, train_y, test_x, test_y


def make_batch(x, y, batch_size):
    indices = torch.randint(0, len(x), (batch_size,))
    return x[indices], y[indices]


def main():
    dataset = torch.load("data/dataset.pt")
    train_x, train_y, _, _ = train_test_split(*dataset)

    m = model.Model()
    progress_bar = tqdm.tqdm(range(100_000))
    for i in progress_bar:
        x, y = make_batch(train_x, train_y, 100)
        loss = m.train(x, y)
        if i % 100 == 0:
            progress_bar.set_postfix(loss=loss)
        if i % 1000 == 0:
            expr, solution = utils.make_exp()
            gen = m.generate(torch.tensor(vocab.stoi(expr)))
            tqdm.tqdm.write(f"{vocab.itos(gen.tolist())} --> {''.join(solution)}")


if __name__ == "__main__":
    main()
