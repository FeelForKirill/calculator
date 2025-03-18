import torch
from torch.functional import F

import vocab

N_EMBED = 10
N_HIDDEN = 10
CONTEXT_SIZE = 18


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Embedding(vocab.VOCAB_SIZE, N_EMBED),
            torch.nn.Flatten(),
            torch.nn.Linear(N_EMBED * CONTEXT_SIZE, N_HIDDEN),
            torch.nn.ReLU(),
            torch.nn.Linear(N_HIDDEN, len(vocab.DIGITS) + 1),  # +1 for end token (dot)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x, y):
        logits = self.seq(x)
        loss = F.cross_entropy(logits, y)
        return logits, loss

    def train(self, x, y):
        self.optimizer.zero_grad()
        _, loss = self.forward(x, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x):
        with torch.no_grad():
            logits = self.seq(x)
            return torch.argmax(logits, dim=1)

    def generate(self, x):
        y = self.predict(x.view(1, x.shape[0]))
        for i in range(x.tolist().index(vocab.stoi(vocab.VOID_TOKEN)[0]), len(x)):
            if vocab.itos(y.item()) == vocab.END_TOKEN:
                break
            x[i] = y.item()
            y = self.predict(x.view(1, x.shape[0]))
        return x
