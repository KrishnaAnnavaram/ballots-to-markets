"""A small LSTM regressor with early stopping (extra ``deep``). torch is imported only here.

The input is the feature row of the last ``window`` days. The scaler is fit on the training rows
only. The last 15% of the training rows are the early-stopping set.
"""

from __future__ import annotations

import numpy as np


class LSTMRegressor:
    def __init__(self, window: int = 10, hidden: int = 16, epochs: int = 200, patience: int = 15,
                 lr: float = 1e-3, seed: int = 42):
        self.window, self.hidden, self.epochs, self.patience, self.lr, self.seed = window, hidden, epochs, patience, lr, seed

    def _seq(self, Z: np.ndarray) -> np.ndarray:
        pad = np.vstack([np.repeat(Z[:1], self.window - 1, axis=0), Z])
        return np.lib.stride_tricks.sliding_window_view(pad, (self.window, Z.shape[1]))[:, 0]

    def fit(self, X, y):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        Xv = np.asarray(X, dtype=np.float32)
        self.mu, self.sd = Xv.mean(axis=0), Xv.std(axis=0) + 1e-8
        self.y_sd = float(np.std(y)) or 1.0
        S = torch.tensor(self._seq((Xv - self.mu) / self.sd))
        t = torch.tensor(np.asarray(y, dtype=np.float32) / self.y_sd)
        cut = int(len(t) * 0.85)

        class Net(nn.Module):
            def __init__(s, d, h):
                super().__init__()
                s.lstm = nn.LSTM(d, h, batch_first=True)
                s.head = nn.Linear(h, 1)

            def forward(s, x):
                out, _ = s.lstm(x)
                return s.head(out[:, -1]).squeeze(-1)

        self.net = Net(S.shape[2], self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        best, best_state, wait = np.inf, None, 0
        for _ in range(self.epochs):
            self.net.train()
            opt.zero_grad()
            loss = ((self.net(S[:cut]) - t[:cut]) ** 2).mean()
            loss.backward()
            opt.step()
            self.net.eval()
            with torch.no_grad():
                val = float(((self.net(S[cut:]) - t[cut:]) ** 2).mean())
            if val < best - 1e-6:
                best, wait = val, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state:
            self.net.load_state_dict(best_state)
        self._train_tail = Xv[-(self.window - 1):] if self.window > 1 else Xv[:0]
        return self

    def predict(self, X):
        import torch

        Xv = np.asarray(X, dtype=np.float32)
        Z = (np.vstack([self._train_tail, Xv]) - self.mu) / self.sd
        S = self._seq(Z)[len(self._train_tail):]
        with torch.no_grad():
            return self.net(torch.tensor(S)).numpy() * self.y_sd
