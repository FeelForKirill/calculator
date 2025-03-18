DIGITS = "0123456789"
OPERATORS = "+"
VOID_TOKEN = " "  # for padding
EQUAL_TOKEN = "="
END_TOKEN = "."
VOCAB = END_TOKEN + DIGITS + OPERATORS + VOID_TOKEN + EQUAL_TOKEN
VOCAB_SIZE = len(VOCAB)


_stoi = {d: i for i, d in enumerate(VOCAB)}
_itos = {i: d for i, d in enumerate(VOCAB)}


def stoi(s):
    return [_stoi[c] for c in s]


def itos(i):
    if isinstance(i, int):
        return _itos[i]
    return "".join(_itos[c] for c in i)
