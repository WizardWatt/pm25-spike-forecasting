"""Train a GRU (two heads: PM2.5 at t+6 regression + spike logit) with the shared pipeline.

Needs from Person A: ONLY the cleaned file (Datetime, StationId, PM2.5, y_reg, y_spike).

Step 1 - tune on 2018 (fit 2015-17, early stopping on 2018, picks alert threshold on 2018):
  python train_gru.py --clean data/clean.csv --station KA003 --out preds/
Step 2 - repeat for the second station (separate model):
  python train_gru.py --clean data/clean.csv --station KA009 --out preds/
Step 3 - ONCE, at the very end (refit 2015-18, fixed epochs + threshold from step 1, score 2019):
  python train_gru.py --clean data/clean.csv --station KA003 --out preds/ --final-test

Outputs in --out: {tag}_{station}_{val|test}.csv (shared prediction format, no intervals yet),
{tag}_{station}_meta.json (best epoch, threshold, history), {tag}_{station}.pt (weights).
Torch is imported lazily so the torch-free parts (threshold picking, report) can be tested anywhere.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import eval_config as C
from gru_data import make_splits, predictions_frame
from evaluate import evaluate, common_rows


# ---------------------------------------------------------------- torch-free helpers
def pos_weight(y_spike, cap=10.0):
    y = np.asarray(y_spike)
    pos, neg = float((y == 1).sum()), float((y == 0).sum())
    return 1.0 if pos == 0 else float(min(np.sqrt(neg / pos), cap))


def pick_threshold(y_spike, prob, beta=2.0):
    """Probability cut-off maximising F-beta (beta=2 favours recall) on the given (2018) data.
    Agree the rule with Person A so GRU / XGBoost / RF are compared under the same rule."""
    y, p = np.asarray(y_spike).astype(bool), np.asarray(prob, float)
    if y.sum() == 0:
        return 0.5
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, 97):
        a = p >= t
        tp, fp, fn = (y & a).sum(), (~y & a).sum(), (y & ~a).sum()
        if tp == 0:
            continue
        pr, rc = tp / (tp + fp), tp / (tp + fn)
        f = (1 + beta ** 2) * pr * rc / (beta ** 2 * pr + rc)
        if f > best_f:
            best_t, best_f = float(t), float(f)
    return best_t


def report(clean, station, split, gru_frame, thr, out_dir, tag="gru"):
    """Compare the GRU with the saved baselines on identical hours."""
    preds, thrs = {tag: gru_frame}, {tag: thr}
    for name in ("persistence", "hist_avg"):
        path = os.path.join(out_dir, f"{name}_{split}.csv")
        if os.path.exists(path):
            b = pd.read_csv(path)
            preds[name], thrs[name] = b[b["StationId"] == station], 0.5
    aligned = common_rows(preds)
    rows = []
    for name, p in aligned.items():
        r = evaluate(p, clean, prob_threshold=thrs[name])[station]
        rows.append({"model": name, "station": station, **r})
    cols = ["model", "station", "n", "MAE", "RMSE", "recall", "precision", "false_alarm_rate",
            "n_episodes", "episode_recall", "mean_lead_h"]
    return pd.DataFrame(rows)[cols].round(3)


# ---------------------------------------------------------------- torch parts
def build_model(n_feat, a):
    import torch.nn as nn

    class GRUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(n_feat, a.hidden, num_layers=a.layers, batch_first=True,
                              dropout=a.dropout if a.layers > 1 else 0.0)
            self.drop = nn.Dropout(a.dropout)
            self.reg = nn.Linear(a.hidden, 1)
            self.cls = nn.Linear(a.hidden, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            h = self.drop(out[:, -1])
            return self.reg(h).squeeze(-1), self.cls(h).squeeze(-1)

    return GRUNet()


def predict(model, X, device, batch=1024):
    import torch
    model.eval()
    reg, prob = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            r, lg = model(xb)
            reg.append(r.cpu().numpy()); prob.append(torch.sigmoid(lg).cpu().numpy())
    return np.concatenate(reg), np.concatenate(prob)


def train(data_tr, data_ev, scaler, a, epochs, early_stop):
    """data_ev is None in final-test mode, so 2019 is never used during training."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(data_tr["X"].shape[2], a).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    pw = a.pos_weight if a.pos_weight else pos_weight(data_tr["y_spike"])
    reg_loss, cls_loss = nn.SmoothL1Loss(), nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=dev))

    def tensors(d):
        return (torch.from_numpy(d["X"]), torch.from_numpy(scaler.transform(d["y_reg"]).astype(np.float32)),
                torch.from_numpy(d["y_spike"]))

    def loss_fn(r, lg, yr, ys):
        return reg_loss(r, yr) + a.spike_weight * cls_loss(lg, ys)

    tr_loader = DataLoader(TensorDataset(*tensors(data_tr)), batch_size=a.batch, shuffle=True)
    ev_loader = DataLoader(TensorDataset(*tensors(data_ev)), batch_size=1024) if data_ev is not None else None

    hist, best, best_ep, best_state, bad = [], np.inf, 0, None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); tot, n = 0.0, 0
        for xb, yr, ys in tr_loader:
            xb, yr, ys = xb.to(dev), yr.to(dev), ys.to(dev)
            opt.zero_grad()
            r, lg = model(xb)
            loss = loss_fn(r, lg, yr, ys)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(xb); n += len(xb)
        row = {"epoch": ep, "train_loss": tot / n}
        if ev_loader is not None:
            model.eval(); vt, vn = 0.0, 0
            with torch.no_grad():
                for xb, yr, ys in ev_loader:
                    xb, yr, ys = xb.to(dev), yr.to(dev), ys.to(dev)
                    r, lg = model(xb)
                    vt += loss_fn(r, lg, yr, ys).item() * len(xb); vn += len(xb)
            row["val_loss"] = vt / vn
            if row["val_loss"] < best - 1e-5:
                best, best_ep, bad = row["val_loss"], ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        hist.append(row)
        print(row, flush=True)
        if early_stop and bad >= a.patience:
            print("early stop"); break
        if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
            print("time budget reached"); break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, dev, {"best_epoch": best_ep or len(hist), "history": hist, "pos_weight": pw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--station", default=C.PRIMARY_STATION)
    ap.add_argument("--out", default="preds")
    ap.add_argument("--tag", default="gru")
    ap.add_argument("--final-test", action="store_true", help="refit 2015-18, score 2019 (touch once)")
    ap.add_argument("--lookback", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--spike-weight", type=float, default=0.5)
    ap.add_argument("--pos-weight", type=float, default=0.0, help="0 = auto sqrt(neg/pos), capped at 10")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="time-box for this run (0 = off)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    clean = pd.read_csv(a.clean, parse_dates=["Datetime"])
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"{a.tag}_{a.station}")
    split = "test" if a.final_test else "val"
    data = make_splits(clean, a.station, split, lookback=a.lookback)
    tr, ev, sc = data["train"], data["eval"], data["scaler"]
    print(f"{a.station} {split}: train windows {len(tr['X'])}, eval windows {len(ev['X'])}, "
          f"train spike rate {tr['y_spike'].mean():.3f}")

    if not a.final_test:
        model, dev, info = train(tr, ev, sc, a, a.epochs, early_stop=True)
        _, p_ev = predict(model, ev["X"], dev)
        thr = pick_threshold(ev["y_spike"], p_ev)
        meta = {"best_epoch": info["best_epoch"], "threshold": thr, "pos_weight": info["pos_weight"],
                "hparams": {k: v for k, v in vars(a).items() if k not in ("clean", "out", "final_test")},
                "history": info["history"]}
        json.dump(meta, open(stem + "_meta.json", "w"), indent=2)
    else:
        if not os.path.exists(stem + "_meta.json"):
            raise SystemExit("Run the 2018 tuning step first (no meta json found).")
        meta = json.load(open(stem + "_meta.json"))
        h = meta["hparams"]
        for k in ("lookback", "hidden", "layers", "dropout", "lr", "weight_decay", "batch",
                  "spike_weight", "seed"):                      # use exactly the tuned settings
            setattr(a, k, h[k])
        a.pos_weight = meta["pos_weight"]
        if tr["X"].shape[1] != a.lookback:
            raise SystemExit("lookback mismatch with tuned model")
        model, dev, _ = train(tr, None, sc, a, meta["best_epoch"], early_stop=False)
        thr = meta["threshold"]

    r_s, prob = predict(model, ev["X"], dev)
    y_pred = np.clip(sc.inverse(r_s), 0, None)
    frame = predictions_frame(ev, a.station, y_pred, prob)
    frame.to_csv(f"{stem}_{split}.csv", index=False)
    torch.save(model.state_dict(), stem + ".pt")
    print(f"alert threshold (from 2018): {thr:.2f}")
    print(report(clean, a.station, split, frame, thr, a.out, a.tag).to_string(index=False))


if __name__ == "__main__":
    main()
