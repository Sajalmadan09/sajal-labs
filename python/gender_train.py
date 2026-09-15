"""exp9: train a real classifier on real data (data/names.csv), export it
through the exact same pipeline exp1-8 used. Architecture is deliberately
TinyMLP unchanged (see gender_features.py's docstring) — this experiment is
about proving the equivalence + cold-invocation methodology transfers to a
real trained model, not about designing a new architecture.
"""
import csv
import json
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn

from tiny_mlp import TinyMLP
from gender_features import build_vocab, extract_features

ROOT = pathlib.Path(__file__).parent.parent
HIDDEN_DIM = 64
VAL_FRACTION = 0.15
EPOCHS = 200
SEED = 0


def load_data():
    rows = list(csv.DictReader(open(ROOT / "data" / "names.csv")))
    return [(r["name"], r["gender"]) for r in rows]


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    data = load_data()
    random.shuffle(data)
    n_val = int(len(data) * VAL_FRACTION)
    val, train = data[:n_val], data[n_val:]

    vocab = build_vocab([name for name, _ in train])  # vocab from TRAIN only, no leakage
    label_to_idx = {"M": 0, "F": 1}

    def to_tensors(split):
        X = torch.tensor([extract_features(n, vocab) for n, _ in split], dtype=torch.float32)
        y = torch.tensor([label_to_idx[g] for _, g in split], dtype=torch.long)
        return X, y

    X_train, y_train = to_tensors(train)
    X_val, y_val = to_tensors(val)

    model = TinyMLP(in_dim=len(vocab), hidden_dim=HIDDEN_DIM, out_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.NLLLoss()  # log(softmax) == cross-entropy; TinyMLP.forward() already applies softmax

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(torch.log(out + 1e-9), y_train)
        loss.backward()
        optimizer.step()

        if epoch % 40 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                train_acc = (model(X_train).argmax(1) == y_train).float().mean().item()
                val_acc = (model(X_val).argmax(1) == y_val).float().mean().item()
            print(f"epoch {epoch:>4}  loss={loss.item():.4f}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

    model.eval()
    with torch.no_grad():
        final_train_acc = (model(X_train).argmax(1) == y_train).float().mean().item()
        final_val_acc = (model(X_val).argmax(1) == y_val).float().mean().item()
        val_probs = model(X_val)
        val_pred = val_probs.argmax(1)

    # A few honest, specific examples rather than just the aggregate number.
    idx_to_label = {0: "M", 1: "F"}
    examples = []
    for i in range(min(15, len(val))):
        examples.append({
            "name": val[i][0], "true": val[i][1],
            "pred": idx_to_label[val_pred[i].item()],
            "confidence": round(val_probs[i, val_pred[i]].item(), 3),
        })

    print(f"\nFinal: train_acc={final_train_acc:.3f}  val_acc={final_val_acc:.3f}  "
          f"(train n={len(train)}, val n={len(val)}, vocab size={len(vocab)})")

    # Export through the SAME pattern export_model.py/export_transformer.py used.
    artifacts_dir = ROOT / "artifacts" / "gender_classifier"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    def save_f32(path, tensor):
        tensor.detach().numpy().astype(np.float32).tofile(path)

    save_f32(artifacts_dir / "fc1_weight.bin", model.fc1.weight)
    save_f32(artifacts_dir / "fc1_bias.bin", model.fc1.bias)
    save_f32(artifacts_dir / "fc2_weight.bin", model.fc2.weight)
    save_f32(artifacts_dir / "fc2_bias.bin", model.fc2.bias)
    torch.save(model.state_dict(), artifacts_dir / "model_state_dict.pt")
    (artifacts_dir / "vocab.txt").write_text("\n".join(vocab))
    # Raw val names, same order as test_inputs.bin/ref_outputs.bin — lets exp10
    # test true end-to-end (raw string in) equivalence, not just precomputed-features.
    (artifacts_dir / "val_names.txt").write_text("\n".join(n for n, _ in val))

    n_test = len(val)
    (artifacts_dir / "shapes.txt").write_text(f"{len(vocab)} {HIDDEN_DIM} 2 {n_test}\n")
    save_f32(artifacts_dir / "test_inputs.bin", X_val)
    with torch.no_grad():
        ref_outputs = model(X_val)
    save_f32(artifacts_dir / "ref_outputs.bin", ref_outputs)

    torch.onnx.export(
        model, (X_val[:1],), str(artifacts_dir / "model.onnx"),
        input_names=["input"], output_names=["output"], opset_version=17, dynamo=False,
    )

    metrics = {
        "train_n": len(train), "val_n": len(val), "vocab_size": len(vocab),
        "train_acc": final_train_acc, "val_acc": final_val_acc,
        "examples": examples,
    }
    (artifacts_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"exported to {artifacts_dir}")


if __name__ == "__main__":
    main()
