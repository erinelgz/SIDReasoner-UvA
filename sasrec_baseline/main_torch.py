import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm


def str2bool(s):
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--full_ranking', action='store_true',
                    help='Evaluate by ranking all items (full ranking) instead of 100 negatives')
parser.add_argument('--loss_type', default='bce', choices=['bce', 'ce'],
                    help='bce: original 1-negative BCE; ce: softmax CE over all items (matches full-ranking training)')
args = parser.parse_args()

device = torch.device('cpu')


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train, user_valid, user_test = {}, {}, {}
    with open('data/%s.txt' % fname) as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u, i = int(u), int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
    for u in User:
        n = len(User[u])
        if n < 3:
            user_train[u] = User[u]
            user_valid[u] = []
            user_test[u] = []
        else:
            user_train[u] = User[u][:-2]
            user_valid[u] = [User[u][-2]]
            user_test[u] = [User[u][-1]]
    return user_train, user_valid, user_test, usernum, itemnum


def random_neq(lo, hi, excluded):
    t = random.randint(lo, hi)
    while t in excluded:
        t = random.randint(lo, hi)
    return t


def sample_batch(user_train, usernum, itemnum, batch_size, maxlen, need_neg=True):
    seqs, pos_seqs, neg_seqs = [], [], []
    for _ in range(batch_size):
        u = random.randint(1, usernum)
        while len(user_train[u]) <= 1:
            u = random.randint(1, usernum)

        seq = np.zeros(maxlen, dtype=np.int64)
        pos = np.zeros(maxlen, dtype=np.int64)
        neg = np.zeros(maxlen, dtype=np.int64)
        nxt = user_train[u][-1]
        idx = maxlen - 1
        ts = set(user_train[u])
        for item in reversed(user_train[u][:-1]):
            seq[idx] = item
            pos[idx] = nxt
            if nxt != 0 and need_neg:
                neg[idx] = random_neq(1, itemnum, ts)
            nxt = item
            idx -= 1
            if idx == -1:
                break
        seqs.append(seq)
        pos_seqs.append(pos)
        neg_seqs.append(neg)
    return np.array(seqs), np.array(pos_seqs), np.array(neg_seqs)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Conv1d(hidden_units, hidden_units, kernel_size=1),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        # x: (batch, seq, hidden) -> transpose for Conv1d -> transpose back
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class SASRec(nn.Module):
    def __init__(self, itemnum, args):
        super().__init__()
        self.args = args
        self.item_emb = nn.Embedding(itemnum + 1, args.hidden_units, padding_idx=0)
        self.pos_emb  = nn.Embedding(args.maxlen + 1, args.hidden_units)
        self.emb_dropout = nn.Dropout(args.dropout_rate)

        self.attn_norms = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        self.ff_norms = nn.ModuleList()
        self.ff_layers = nn.ModuleList()
        for _ in range(args.num_blocks):
            self.attn_norms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.attn_layers.append(
                nn.MultiheadAttention(args.hidden_units, args.num_heads,
                                      dropout=args.dropout_rate, batch_first=True)
            )
            self.ff_norms.append(nn.LayerNorm(args.hidden_units, eps=1e-8))
            self.ff_layers.append(PointWiseFeedForward(args.hidden_units, args.dropout_rate))
        self.last_norm = nn.LayerNorm(args.hidden_units, eps=1e-8)

    def log2feats(self, log_seqs):
        # log_seqs: (batch, maxlen) int64, 0 = padding
        seqs = self.item_emb(log_seqs) * (self.args.hidden_units ** 0.5)
        positions = torch.arange(1, log_seqs.size(1) + 1, device=log_seqs.device).unsqueeze(0)
        seqs = self.emb_dropout(seqs + self.pos_emb(positions))

        pad_mask = (log_seqs == 0)  # (batch, maxlen) — True = padded
        causal_mask = torch.triu(
            torch.full((log_seqs.size(1), log_seqs.size(1)), float('-inf'), device=log_seqs.device),
            diagonal=1,
        )

        for attn_norm, attn, ff_norm, ff in zip(
                self.attn_norms, self.attn_layers, self.ff_norms, self.ff_layers):
            # Pre-norm, attention, residual (residual with normalised input, matching original)
            Q = attn_norm(seqs)
            # No key_padding_mask: mixing float attn_mask with bool key_padding_mask
            # triggers a PyTorch 2.x bug (bool treated as 0/1 not 0/-inf, producing NaN).
            # Padding is handled instead by explicitly zeroing padded positions after each block.
            attn_out, _ = attn(Q, seqs, seqs, attn_mask=causal_mask)
            seqs = Q + attn_out

            # Pre-norm, feedforward, residual
            fn = ff_norm(seqs)
            seqs = fn + ff(fn)

            # Zero padded positions (this is the SASrec paper's padding strategy)
            seqs = seqs * (~pad_mask).unsqueeze(-1)

        return self.last_norm(seqs)

    def forward(self, log_seqs, pos_seqs, neg_seqs):
        feats = self.log2feats(log_seqs)
        pos_logits = (feats * self.item_emb(pos_seqs)).sum(-1)
        neg_logits = (feats * self.item_emb(neg_seqs)).sum(-1)
        return pos_logits, neg_logits

    def predict(self, log_seqs, items):
        feats = self.log2feats(log_seqs)[:, -1, :]  # (batch, hidden)
        return feats @ self.item_emb(items).T         # (batch, n_items)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_seq(history, maxlen):
    seq = np.zeros(maxlen, dtype=np.int64)
    idx = maxlen - 1
    for i in reversed(history):
        seq[idx] = i
        idx -= 1
        if idx == -1:
            break
    return seq


def evaluate_split(model, user_target, user_seq_history, user_exclude,
                   usernum, itemnum, args):
    model.eval()
    metrics = {k: 0.0 for k in ('Recall@5', 'Recall@10', 'NDCG@5', 'NDCG@10')}
    n = 0

    users = (range(1, usernum + 1) if usernum <= 10000
             else random.sample(range(1, usernum + 1), 10000))

    all_items = torch.arange(1, itemnum + 1, device=device) if args.full_ranking else None

    with torch.no_grad():
        for u in users:
            if not user_target.get(u):
                continue
            target = user_target[u][0]

            seq = build_seq(user_seq_history.get(u, []), args.maxlen)
            seq_t = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

            excluded = set(user_exclude.get(u, []))

            if args.full_ranking:
                scores = model.predict(seq_t, all_items)[0].cpu().numpy()
                excl_mask = np.zeros(itemnum, dtype=bool)
                for eid in excluded:
                    if 1 <= eid <= itemnum:
                        excl_mask[eid - 1] = True
                target_score = scores[target - 1]
                rank = int(((scores > target_score) & ~excl_mask).sum())
            else:
                # 100 randomly sampled negatives (original SASRec protocol)
                neg_items = []
                seen = set(excluded) | {target}
                while len(neg_items) < 100:
                    t = random.randint(1, itemnum)
                    if t not in seen:
                        neg_items.append(t)
                        seen.add(t)
                items = torch.tensor([target] + neg_items, dtype=torch.long, device=device)
                scores = model.predict(seq_t, items)[0].cpu().numpy()
                rank = int((scores[1:] > scores[0]).sum())

            n += 1
            if rank < 5:
                metrics['Recall@5']  += 1
                metrics['NDCG@5']    += 1 / np.log2(rank + 2)
            if rank < 10:
                metrics['Recall@10'] += 1
                metrics['NDCG@10']   += 1 / np.log2(rank + 2)

    model.train()
    return {k: v / n for k, v in metrics.items()} if n > 0 else metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

user_train, user_valid, user_test, usernum, itemnum = data_partition(args.dataset)

print('Users: %d  Items: %d' % (usernum, itemnum))
avg_len = sum(len(v) for v in user_train.values()) / max(len(user_train), 1)
print('Average sequence length: %.2f' % avg_len)

out_dir = '%s_%s' % (args.dataset, args.train_dir)
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'args.txt'), 'w') as f:
    f.write('\n'.join('%s,%s' % kv for kv in sorted(vars(args).items())))

model = SASRec(itemnum, args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

num_batch = max(len(user_train) // args.batch_size, 1)
log_file = open(os.path.join(out_dir, 'log.txt'), 'w')

T = 0.0
t0 = time.time()

# Precompute test history (train + valid) once
user_test_history = {u: user_train.get(u, []) + user_valid.get(u, [])
                     for u in range(1, usernum + 1)}

for epoch in range(1, args.num_epochs + 1):
    model.train()
    for _ in tqdm(range(num_batch), ncols=70, leave=False, unit='b'):
        if args.loss_type == 'ce':
            seqs, pos, _ = sample_batch(user_train, usernum, itemnum,
                                        args.batch_size, args.maxlen, need_neg=False)
            seqs = torch.LongTensor(seqs).to(device)
            pos  = torch.LongTensor(pos).to(device)

            # Score against all items (item IDs 1..itemnum → indices 0..itemnum-1)
            feats = model.log2feats(seqs)  # (batch, maxlen, hidden)
            all_embs = model.item_emb.weight[1:]  # (itemnum, hidden)
            logits = feats @ all_embs.T  # (batch, maxlen, itemnum)

            is_target = (pos != 0).reshape(-1)  # (batch*maxlen,)
            # pos is 1-indexed; CE expects 0-indexed targets; padding pos=0 → -1 (ignored)
            targets = (pos - 1).reshape(-1)  # (batch*maxlen,)

            loss = F.cross_entropy(
                logits.reshape(-1, itemnum)[is_target],
                targets[is_target],
            )
        else:
            seqs, pos, neg = sample_batch(user_train, usernum, itemnum,
                                          args.batch_size, args.maxlen, need_neg=True)
            seqs = torch.LongTensor(seqs).to(device)
            pos  = torch.LongTensor(pos).to(device)
            neg  = torch.LongTensor(neg).to(device)

            pos_logits, neg_logits = model(seqs, pos, neg)
            is_target = (pos != 0).float()

            loss = (
                -torch.log(torch.sigmoid(pos_logits) + 1e-24) * is_target
                - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * is_target
            ).sum() / is_target.sum()

        if args.l2_emb != 0:
            loss = loss + args.l2_emb * model.item_emb.weight.norm()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 20 == 0:
        print(f'[epoch {epoch}] loss={loss.item():.6f}', flush=True)
        t1 = time.time() - t0
        T += t1
        print('Evaluating', end=' ', flush=True)

        t_valid = evaluate_split(model, user_valid, user_train, user_train,
                                 usernum, itemnum, args)
        t_test  = evaluate_split(model, user_test,  user_test_history, user_test_history,
                                 usernum, itemnum, args)
        print()
        print('epoch:%d, time: %.1fs' % (epoch, T))
        print('  valid  Recall@5: %.4f  Recall@10: %.4f  NDCG@5: %.4f  NDCG@10: %.4f' % (
            t_valid['Recall@5'], t_valid['Recall@10'], t_valid['NDCG@5'], t_valid['NDCG@10']))
        print('  test   Recall@5: %.4f  Recall@10: %.4f  NDCG@5: %.4f  NDCG@10: %.4f' % (
            t_test['Recall@5'],  t_test['Recall@10'],  t_test['NDCG@5'],  t_test['NDCG@10']))

        log_file.write('epoch:%d valid %s test %s\n' % (epoch, t_valid, t_test))
        log_file.flush()
        t0 = time.time()

log_file.close()
print('Done')
