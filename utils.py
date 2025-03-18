import random

import model
import vocab


def make_sample():
    x, solution = make_exp()
    y = solution[0]
    yield list(x), y
    for i, (p, c) in enumerate(zip(solution, solution[1:])):
        x[len(x) + i] = p
        yield list(x), c


def make_exp():
    a = random.randint(0, 99_999)
    b = random.randint(0, 99_999)
    c = a + b
    expr = list(f"{a}+{b}=")
    expr += [vocab.VOID_TOKEN] * (model.CONTEXT_SIZE - len(expr))
    solution = list(f"{c}.")
    return expr, solution
