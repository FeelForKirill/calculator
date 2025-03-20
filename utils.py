import random

import model
import vocab


def make_sample():
    x, solution = make_exp()
    y = solution[0]
    void_idx = x.index(vocab.VOID_TOKEN)
    yield list(x), y
    for i, (p, c) in enumerate(zip(solution, solution[1:])):
        x[void_idx + i] = p
        yield list(x), c


def make_exp():
    a = random.randint(0, 99_999)
    b = random.randint(0, 99_999)
    c = a + b
    expr = list(f"{a}+{b}=")
    expr += [vocab.VOID_TOKEN] * (model.CONTEXT_SIZE - len(expr))
    solution = list(str(c)[::-1]) + [vocab.END_TOKEN]
    return expr, solution
