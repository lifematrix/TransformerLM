# -*- coding: utf-8 -*-

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from .models import TSUtils, LMTransformerBilateralcoder
from tqdm import tqdm

class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src, tgt=None, pad=2, sos=0):    # 2 = <blank>
        self.src = src
        self.src_mask = (src!=pad).unsqueeze(-2).unsqueeze(-3)

        if tgt is not None:
            tgt = torch.concatenate([torch.zeros_like(tgt[:, :1]).fill_(sos), tgt], dim=1)
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.n_tokens = (self.tgt_y != pad).data.sum()

    @classmethod
    def make_std_mask(cls, tgt, pad):
        """Create a mask to hide padding and future words"""
        tgt_mask = (tgt!= pad).unsqueeze(-2)
        tgt_mask = tgt_mask & TSUtils.make_causal_mask(tgt)
        tgt_mask = tgt_mask.unsqueeze(-3)

        return tgt_mask


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0



def run_epoch(epoch, data_iter, model, loss_compute,
              optimizer, scheduler, device, mode="train", train_state=None):

    if train_state is None:
        train_state = TrainState()

    start = time.time()
    total_tokens = 0
    total_loss = 0
    n_accum = 0

    for i, batch in enumerate(data_iter):
        # print("batch.src: ", batch.src)
        # print("batch.tgt: ", batch.tgt)
        # print("batch.tgt_y: ", batch.tgt_y)
        out = model(batch.src.to(device), batch.tgt.to(device), batch.src_mask.to(device), batch.tgt_mask.to(device))
        loss, loss_node = loss_compute(out, batch.tgt_y.to(device), batch.n_tokens)

        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.n_tokens

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
            train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.n_tokens

        if (i+1) % 10 == 0 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
               "Epoch[%d] step: %6d | Accumulation Step: %3d | Loss: %8.4f | Tokens / Sec: %7.1f | Learning rate: %6.3e"
                % (epoch, i+1, train_state.accum_step, total_loss/total_tokens, total_tokens/elapsed, lr)
            )
            start = time.time()

        del loss
        del loss_node

    return total_loss / total_tokens, train_state

def lr_fn(step, d_model, factor, warmup):
    if step == 0:
        step = 1

    lr = factor * (d_model**-0.5) * min((step**-0.5), step*(warmup**-1.5))

    return lr


def draw_example_learning_schedule():
    import altair as alt
    import pandas as pd
    import numpy as np

    opts = [
        [512, 1, 4000],
        [512, 1, 8000],
        [256, 1, 4000]
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rate = []
    steps = np.arange(20000)

    for idx, opt in enumerate(opts):
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: lr_fn(step, *opt))
        tmp = []
        for step in steps:
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rate.append(tmp)

    learning_rates = np.array(learning_rate)
    #print("learning rate: ", learning_rates.shape, learning_rates.dtype)

    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning rate": learning_rates[idx, :],
                    "model_size#warmup": f"{opts[idx][0]}#{opts[idx][2]}",
                    "step": steps
                }
            )
            for idx in range(3)
        ]
    )

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning rate", color="model_size#warmup:N")
        .interactive()
    )



class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="batchmean")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size, (f"The 1st size of input 'x' {x.size(1)} should be equal to "
                                        f"the predifined size {self.size}")

        true_dist = x.detach().clone()
        true_dist.fill_(self.smoothing/(self.size-2))
        true_dist.scatter_(1, target.detach().unsqueeze(1), self.confidence)


        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.detach() == self.padding_idx)
        if mask.dim() > 0:
            true_dist = true_dist.index_fill(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        loss = self.criterion(x, true_dist.clone().detach())

        return loss


def illustrate_label_smoothing():
    """Illustrate the label smoothing adjusted cross-entropy loss"""
    import pandas as pd
    import altair as alt

    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0]
        ]
    )

    loss = crit(x=predict, target=torch.LongTensor([2,1,0,3,3]))
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(),
                    "columns": y,
                    "rows": x
                }
            )

            for y in range(5)
            for x in range(5)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=200, width=200)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color(
                "target distribution:Q", scale=alt.Scale(scheme="viridis")
            )
        )
        .interactive()
    )

def loss(x, crit):
    d = x + 3
    epsilon = 1e-6
    predict = torch.FloatTensor([[epsilon, x/d, 1/d, 1/d, 1/d-epsilon]])

    return crit(predict.log(), torch.LongTensor([1])).data


def illustrate_label_smoothing_penalization():
    import numpy as np
    import altair as alt

    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": np.arange(99)
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss"
        )
        .interactive()
    )


def data_gen(V, batch_size, n_batches, seq_len=10):
    """Generate random data for a src-tgt copy task."""

    for i in range(n_batches):
        data = torch.randint(1, V, size=(batch_size, seq_len))
        data[:, 0] = 1    # 1 is <start of sequence>
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)   # 0 is pad


class SimpleLossCompute:
    """A simple loss compute and train function."""

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, target, norm):
        pred = self.generator(x)
        sloss = (
            self.criterion(
                pred.contiguous().view(-1, pred.size(-1)),
                target.contiguous().view(-1))
            / norm
        )

        return sloss.data * norm, sloss

def greedy_decode(model, src, src_mask, max_len, start_symbol):

    memory = model.encode(src, src_mask)
    ys = torch.zeros(1,1).fill_(start_symbol).to(src.dtype)

    for i in range(max_len-1):
        out = model.decode(memory, src_mask, ys, TSUtils.make_causal_mask(ys))
        prob = model.generator(out[:, -1])
        _, next_token = torch.max(prob, dim=1)
        next_token = next_token.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1,1).to(src.dtype).fill_(next_token)],
            dim=1
        )

    return ys

class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        super(DummyOptimizer, self).__init__([], dict())
        self.param_groups = [{"lr": 0}]

    def step(self, closure: None = None):
        pass

    def zero_grad(self, set_to_none=False):
        pass


class DummyScheduler:
    def step(self):
        pass

def illustrate_simple_model_train():
    device = "cpu"
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = LMTransformerBilateralcoder(src_vocab_size=V,
                                        tgt_vocab_size=V,
                                        n_encoder_layers=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.92), eps=1e-9)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: lr_fn(step, d_model=model.d_model, factor=1.0, warmup=400)
    )

    print("lr_fn: ", [lr_fn(step, d_model=model.d_model, factor=1.0, warmup=400) for step in [200, 400, 800]])
    n_batches = 80
    batch_size = 2
    train_state = TrainState()
    for epoch in range(40):
        model.train()
        run_epoch(
            epoch,
            data_gen(V, batch_size, n_batches),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            device=device,
            mode="train",
            train_state=train_state,
        )
        model.eval()
        r = run_epoch(
            epoch,
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            #DummyOptimizer(),
            optimizer,
            DummyScheduler(),
            device=device,
            mode="eval",
            train_state=train_state
        )

    model.eval()
    src = torch.LongTensor(np.arange(10)[np.newaxis, ...])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


if __name__ == "__main__":
    illustrate_simple_model_train()

