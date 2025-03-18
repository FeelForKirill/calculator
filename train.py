import random

import torch
import tqdm

import model

_stoi = {d: i for i, d in enumerate(model.VOCAB)}
_itos = {i: d for i, d in enumerate(model.VOCAB)}


def stoi(s):
    return [_stoi[c] for c in s]


def itos(i):
    return "".join(_itos[c] for c in i)


def make_sample():
    a = random.randint(0, 99_999)
    b = random.randint(0, 99_999)
    c = a + b
    expr = list(f"{a}+{b}=")
    solution = list(f"{c}.")

    x = expr + [model.VOID_TOKEN] * (model.CONTEXT_SIZE - len(expr))
    y = solution[0]
    yield list(x), y
    for i, (p, c) in enumerate(zip(solution, solution[1:])):
        x[len(expr) + i] = p
        yield list(x), c


def make_dataset():
    x = []
    y = []
    for _ in range(100_000):
        for _x, _y in make_sample():
            x.append(stoi(_x))
            y.append(stoi([_y]))
    return torch.tensor(x), torch.tensor(y).view(-1)


def train_test_split(x, y, test_size=0.2):
    train_x = x[: int(len(x) * (1 - test_size))]
    train_y = y[: int(len(y) * (1 - test_size))]
    test_x = x[int(len(x) * (1 - test_size)) :]
    test_y = y[int(len(y) * (1 - test_size)) :]
    return train_x, train_y, test_x, test_y


def make_batch(x, y, batch_size):
    indices = torch.randint(0, len(x), (batch_size,))
    return x[indices], y[indices]


def generate_prediction(m):
    x, _ = next(make_sample())
    x = stoi(x)
    y = m.predict(torch.tensor([x]))
    for i in range(x.index(_stoi[model.VOID_TOKEN]), len(x)):
        if _itos[y.item()] == model.END_TOKEN:
            break
        x[i] = y.item()
        y = m.predict(torch.tensor([x]))
    return itos(x)


def main():
    dataset = make_dataset()
    train_x, train_y, test_x, test_y = train_test_split(*dataset)

    m = model.Model()
    progress_bar = tqdm.tqdm(range(10_000))
    for i in progress_bar:
        x, y = make_batch(train_x, train_y, 100)
        loss = m.train(x, y)
        if i % 100 == 0:
            progress_bar.set_postfix(loss=loss)

    print(generate_prediction(m))


if __name__ == "__main__":
    main()
