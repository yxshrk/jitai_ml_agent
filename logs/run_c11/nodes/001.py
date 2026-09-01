"""Wide random-search and successive-pruning FM tuning on the official NPZ path."""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.official.evaluate import evaluate


class FM(torch.nn.Module):
    def __init__(self, total_dim, k=16, dropout=0.0):
        super().__init__()
        self.emb = torch.nn.Embedding(total_dim, k)
        self.lin = torch.nn.Embedding(total_dim, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.dropout = torch.nn.Dropout(dropout)
        torch.nn.init.normal_(self.emb.weight, std=0.01)
        torch.nn.init.zeros_(self.lin.weight)

    def forward(self, x):
        e = self.dropout(self.emb(x))
        s = e.sum(1)
        pair = 0.5 * (s * s - (e * e).sum(1)).sum(1)
        return self.bias + self.lin(x).sum((1, 2)) + pair


def metric_values(m):
    return (float(m["GAUC"] if "GAUC" in m else m["gauc"]),
            float(m.get("nDCG@5", m.get("ndcg5"))),
            float(m["primary"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = np.load(os.path.join(a.data_dir, "train.npz"))
    va = np.load(os.path.join(a.data_dir, "val.npz"))
    field_dims = tr["field_dims"].astype(np.int64)
    total_dim = int(field_dims.sum())
    offsets = np.concatenate(([0], np.cumsum(field_dims[:-1]))).astype(np.int64)
    Xt = torch.from_numpy(tr["X"].astype(np.int64) + offsets).to(device)
    yt = torch.from_numpy(tr["y"].astype(np.float32)).to(device)
    Xv = torch.from_numpy(va["X"].astype(np.int64) + offsets).to(device)
    val_users = va["user"]
    val_labels = va["y"].astype(int)
    n = len(yt)
    bs = 8192

    os.makedirs(a.out_dir, exist_ok=True)
    progress_path = os.path.join(a.out_dir, "progress.log")
    progress = open(progress_path, "a", buffering=1)
    history = []

    smoke_value = os.environ.get("SMOKE_EPOCHS")
    smoke_epochs = int(smoke_value) if smoke_value is not None else None
    timeout_s = float(os.environ.get("NODE_TIMEOUT_S", "7200"))
    search_start = time.monotonic()

    def capped_epochs(requested):
        requested = max(1, int(requested))
        if smoke_epochs is not None:
            return max(1, min(requested, smoke_epochs))
        return requested

    def train_once(cfg, max_epochs, phase, probe_id, keep_scores=False):
        torch.manual_seed(a.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(a.seed)
        model = FM(total_dim, int(cfg["k"]), float(cfg["dropout"])).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]),
                               weight_decay=float(cfg["weight_decay"]))
        bce = torch.nn.BCEWithLogitsLoss()
        best = -1.0
        best_scores = None
        best_epoch = 0
        patience_count = 0
        curve = []
        epochs_here = capped_epochs(max_epochs)
        for epoch in range(epochs_here):
            model.train()
            perm = torch.randperm(n, device=device)
            last_loss = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad(set_to_none=True)
                loss = bce(model(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                last_loss = float(loss.detach().item())
            model.eval()
            parts = []
            with torch.inference_mode():
                for i in range(0, len(Xv), 65536):
                    parts.append(model(Xv[i:i + 65536]).detach().cpu().numpy())
            scores = np.concatenate(parts)
            m = evaluate(val_users, val_labels, scores)
            gauc, ndcg5, primary = metric_values(m)
            curve.append({"epoch": epoch + 1, "train_loss": round(last_loss, 6),
                          "val_gauc": round(gauc, 7), "val_ndcg5": round(ndcg5, 7),
                          "val_primary": round(primary, 7)})
            if primary > best + 1e-7:
                best = primary
                best_scores = scores.copy()
                best_epoch = epoch + 1
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= int(cfg["patience"]):
                    break
        record = {"phase": phase, "probe": int(probe_id),
                  "config": {"k": int(cfg["k"]), "lr": float(cfg["lr"]),
                             "weight_decay": float(cfg["weight_decay"]),
                             "dropout": float(cfg["dropout"]),
                             "patience": int(cfg["patience"])},
                  "epoch_cap": int(epochs_here), "epochs_run": len(curve),
                  "best_epoch": int(best_epoch), "primary": round(float(best), 8),
                  "gauc": curve[best_epoch - 1]["val_gauc"],
                  "ndcg5": curve[best_epoch - 1]["val_ndcg5"]}
        if phase == "final":
            record["curve"] = curve
        history.append(record)
        progress.write(json.dumps(record, separators=(",", ":")) + "\n")
        del model, opt
        return record, (best_scores if keep_scores else None)

    rng = np.random.default_rng(a.seed + 1701)
    anchor = {"k": 16, "lr": 1e-3, "weight_decay": 0.0,
              "dropout": 0.0, "patience": 2}

    def sample_config(index):
        if index == 0:
            return dict(anchor)
        k = int(rng.choice(np.array([4, 8, 12, 16, 24, 32, 48, 64])))
        lr = float(math.exp(rng.uniform(math.log(8e-5), math.log(8e-3))))
        if rng.random() < 0.12:
            wd = 0.0
        else:
            wd = float(math.exp(rng.uniform(math.log(1e-9), math.log(1e-2))))
        dropout = 0.0 if rng.random() < 0.18 else float(rng.uniform(0.02, 0.55))
        patience = int(rng.choice(np.array([1, 2, 2, 3, 4])))
        return {"k": k, "lr": lr, "weight_decay": wd,
                "dropout": dropout, "patience": patience}

    stage1 = []
    if smoke_epochs is not None:
        initial_min = 3
        initial_max = 3
    else:
        initial_min = 96
        initial_max = 4096
    phase1_deadline = search_start + timeout_s * 0.23
    probe_id = 0
    while probe_id < initial_max and (probe_id < initial_min or time.monotonic() < phase1_deadline):
        cfg = sample_config(probe_id)
        rec, _ = train_once(cfg, 3, "probe_3ep", probe_id, False)
        stage1.append((rec["primary"], cfg, probe_id))
        probe_id += 1

    stage1.sort(key=lambda z: z[0], reverse=True)
    keep1 = 1 if smoke_epochs is not None else max(1, int(math.ceil(len(stage1) * 0.45)))
    candidates2 = stage1[:keep1]
    stage2 = []
    phase2_deadline = search_start + timeout_s * 0.46
    for rank, (_, cfg, original_id) in enumerate(candidates2):
        if rank > 0 and time.monotonic() >= phase2_deadline:
            break
        rec, _ = train_once(cfg, 7, "refine_7ep", original_id, False)
        stage2.append((rec["primary"], cfg, original_id))

    stage2.sort(key=lambda z: z[0], reverse=True)
    keep2 = 1 if smoke_epochs is not None else max(1, int(math.ceil(len(stage2) * 0.45)))
    candidates3 = stage2[:keep2]
    stage3 = []
    phase3_deadline = search_start + timeout_s * 0.67
    for rank, (_, cfg, original_id) in enumerate(candidates3):
        if rank > 0 and time.monotonic() >= phase3_deadline:
            break
        rec, _ = train_once(cfg, a.epochs, "refine_full", original_id, False)
        stage3.append((rec["primary"], cfg, original_id))

    if stage3:
        winner_pool = stage3
    elif stage2:
        winner_pool = stage2
    else:
        winner_pool = stage1
    winner_pool.sort(key=lambda z: z[0], reverse=True)
    winner_cfg = winner_pool[0][1]
    final_record, best_scores = train_once(winner_cfg, a.epochs, "final", -1, True)
    progress.close()

    final_metrics = evaluate(val_users, val_labels, best_scores)
    gauc, ndcg5, primary = metric_values(final_metrics)
    with open(os.path.join(a.out_dir, "metrics.json"), "w") as fh:
        json.dump({"gauc": gauc, "ndcg5": ndcg5, "primary": primary,
                   "selected_config": final_record["config"],
                   "selected_epoch": final_record["best_epoch"],
                   "history": history}, fh)
    with open(os.path.join(a.out_dir, "predictions.csv"), "w") as fh:
        fh.write("row_id,user_id,video_id,score\n")
        for i, score in enumerate(best_scores):
            fh.write(f"{i},{val_users[i]},0,{score:.6g}\n")


if __name__ == "__main__":
    main()
