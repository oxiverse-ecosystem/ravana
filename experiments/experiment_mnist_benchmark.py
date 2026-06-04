"""
RAVANA on Standard Continual Learning Benchmarks: Split-MNIST and Permuted-MNIST

This experiment adapts RAVANA's text-based interface to image classification,
demonstrating that the architecture works on widely accepted public benchmarks
(not just procedurally generated data).

Split-MNIST: 5 sequential binary classification tasks (0v1, 2v3, ..., 8v9)
Permuted-MNIST: 10 tasks, each with a fixed random pixel permutation

Usage:
    python experiments/experiment_mnist_benchmark.py
    python experiments/experiment_mnist_benchmark.py --quick   # fewer samples
"""

import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import time
import json
from ravana_ml.nn.rlm import RLM
from ravana_ml.tokenizer import PixelTokenizer


# ── MNIST Loading ──

def _download_mnist_binary(url, dest_path):
    """Download a gzip-compressed MNIST binary file."""
    import gzip
    import urllib.request
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        print(f"    Downloading {url}...")
        urllib.request.urlretrieve(url, dest_path)
    with gzip.open(dest_path, 'rb') as f:
        return f.read()


def _load_mnist_images(path):
    """Parse MNIST image file (IDX format)."""
    basename = os.path.basename(path)
    # Try multiple mirrors
    mirrors = [
        f"https://ossci-datasets.s3.amazonaws.com/mnist/{basename}.gz",
        f"http://yann.lecun.com/exdb/mnist/{basename}.gz",
    ]
    data = None
    for url in mirrors:
        try:
            data = _download_mnist_binary(url, path)
            break
        except Exception:
            continue
    if data is None:
        raise RuntimeError(f"Failed to download {basename} from all mirrors")
    # IDX format: magic(4) + n_images(4) + n_rows(4) + n_cols(4) + pixels
    n = int.from_bytes(data[4:8], 'big')
    rows = int.from_bytes(data[8:12], 'big')
    cols = int.from_bytes(data[12:16], 'big')
    images = np.frombuffer(data[16:], dtype=np.uint8).reshape(n, rows, cols)
    return images.astype(np.float32) / 255.0


def _load_mnist_labels(path):
    """Parse MNIST label file (IDX format)."""
    basename = os.path.basename(path)
    mirrors = [
        f"https://ossci-datasets.s3.amazonaws.com/mnist/{basename}.gz",
        f"http://yann.lecun.com/exdb/mnist/{basename}.gz",
    ]
    data = None
    for url in mirrors:
        try:
            data = _download_mnist_binary(url, path)
            break
        except Exception:
            continue
    if data is None:
        raise RuntimeError(f"Failed to download {basename} from all mirrors")
    return np.frombuffer(data[8:], dtype=np.uint8).astype(np.int64)


def load_mnist():
    """Load MNIST data via urllib fallback."""
    data_dir = os.path.join(_PROJECT_ROOT, 'data', 'mnist')
    train_images = _load_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    train_labels = _load_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    test_images = _load_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    test_labels = _load_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    return train_images, train_labels, test_images, test_labels


# ── Pixel Tokenization ──
# Uses PixelTokenizer from ravana_ml.tokenizer (vocab_size=266).
# Legacy inline functions removed; see ravana_ml/tokenizer.py for the class.
_pixel_tokenizer = PixelTokenizer()


# ── Dataset Adapters ──

def create_split_mnist_tasks(train_images, train_labels, test_images, test_labels):
    """Split MNIST into 5 sequential binary classification tasks."""
    tasks = []
    for task_id in range(5):
        digit_a, digit_b = task_id * 2, task_id * 2 + 1

        # Filter train
        train_mask = (train_labels == digit_a) | (train_labels == digit_b)
        task_train_imgs = train_images[train_mask]
        task_train_lbls = (train_labels[train_mask] == digit_b).astype(np.int64)

        # Filter test
        test_mask = (test_labels == digit_a) | (test_labels == digit_b)
        task_test_imgs = test_images[test_mask]
        task_test_lbls = (test_labels[test_mask] == digit_b).astype(np.int64)

        tasks.append({
            "name": f"Task {task_id}: {digit_a}v{digit_b}",
            "train_imgs": task_train_imgs,
            "train_lbls": task_train_lbls,
            "test_imgs": task_test_imgs,
            "test_lbls": task_test_lbls,
        })
    return tasks


def create_permutation_tasks(train_images, train_labels, test_images, test_labels,
                              n_tasks=10):
    """Create permuted-MNIST tasks."""
    rng = np.random.RandomState(42)
    perms = [rng.permutation(784) for _ in range(n_tasks)]

    tasks = []
    for task_id, perm in enumerate(perms):
        # Apply permutation
        flat_train = train_images.reshape(-1, 784)[:, perm].reshape(-1, 28, 28)
        flat_test = test_images.reshape(-1, 784)[:, perm].reshape(-1, 28, 28)

        tasks.append({
            "name": f"Permuted Task {task_id}",
            "train_imgs": flat_train,
            "train_lbls": train_labels.copy(),
            "test_imgs": flat_test,
            "test_lbls": test_labels.copy(),
        })
    return tasks


# ── Training and Evaluation ──

def train_on_task(model, images, labels, max_samples=None, batch_log=False):
    """Train RLM on a task's training data. Returns list of (input_ids, target_ids) for Fisher computation."""
    n = len(images)
    if max_samples:
        n = min(n, max_samples)

    indices = np.random.permutation(len(images))[:n]
    errors = []
    experiences = []

    for idx in indices:
        tokens = _pixel_tokenizer.encode_image(images[idx])
        target = np.array([_pixel_tokenizer.encode_label(labels[idx])], dtype=np.int64)
        input_ids = tokens.reshape(1, -1)
        target_ids = target.reshape(1, -1)
        err = model.learn(input_ids, target_ids)
        experiences.append((input_ids.squeeze(0), target_ids.squeeze(0)))
        errors.append(err)

    return errors, experiences


def evaluate_on_task(model, images, labels, max_samples=None):
    """Evaluate RLM on a task's test data. Returns top-1 accuracy."""
    n = len(images)
    if max_samples:
        n = min(n, max_samples)

    indices = np.random.permutation(len(images))[:n]
    correct = 0

    for idx in indices:
        tokens = _pixel_tokenizer.encode_image(images[idx])
        logits = model.forward(tokens.reshape(1, -1))
        data = logits.data if hasattr(logits, 'data') else np.array(logits)
        if data.ndim > 1:
            data = data[0]

        # Only consider class label tokens (256-265 for 10 classes)
        class_logits = data[PixelTokenizer.LABEL_OFFSET:PixelTokenizer.LABEL_OFFSET + PixelTokenizer.N_CLASSES]
        pred = np.argmax(class_logits)
        if pred == labels[idx]:
            correct += 1

    return correct / max(1, len(indices))


# ── Continual Learning Protocol ──

def run_split_mnist(max_samples_per_task=500):
    """Run Split-MNIST benchmark."""
    print("\n  Loading MNIST...")
    train_imgs, train_lbls, test_imgs, test_lbls = load_mnist()

    print("  Creating Split-MNIST tasks...")
    tasks = create_split_mnist_tasks(train_imgs, train_lbls, test_imgs, test_lbls)

    model = RLM(
        vocab_size=266,  # PixelTokenizer: 256 pixel bins + 10 classes
        embed_dim=32,
        concept_dim=32,
        n_concepts=100,
        n_hidden=32,
        n_layers=3,
        max_seq_len=785,  # 784 pixels + 1 label
    )

    # Accuracy matrix: A[i][j] = accuracy on task j after training on task i
    n_tasks = len(tasks)
    accuracy_matrix = np.zeros((n_tasks, n_tasks))

    for train_id, task in enumerate(tasks):
        # Reset hidden state between tasks to prevent cross-task contamination
        model._prev_hidden_state = None

        print(f"\n  Training on {task['name']}...")
        t0 = time.time()
        _, experiences = train_on_task(model, task["train_imgs"], task["train_lbls"],
                      max_samples=max_samples_per_task)
        dt = time.time() - t0
        print(f"    Trained in {dt:.1f}s")

        # Compute Fisher information from this task's experiences before snapshotting
        model.compute_fisher(experiences, n_samples=50)
        del experiences  # free memory
        model.snapshot_weights()

        # Evaluate on all seen tasks
        for eval_id in range(train_id + 1):
            acc = evaluate_on_task(model, tasks[eval_id]["test_imgs"],
                                   tasks[eval_id]["test_lbls"],
                                   max_samples=max_samples_per_task)
            accuracy_matrix[train_id][eval_id] = acc
            print(f"    Task {eval_id} accuracy: {acc:.1%}")

    # Compute metrics
    avg_accuracy = np.mean(accuracy_matrix[n_tasks-1, :])
    # BWT: average change in accuracy on task j after learning task j+1
    bwt_scores = []
    for j in range(n_tasks - 1):
        bwt_scores.append(accuracy_matrix[n_tasks-1][j] - accuracy_matrix[j][j])
    avg_bwt = np.mean(bwt_scores) if bwt_scores else 0.0

    return {
        "accuracy_matrix": accuracy_matrix.tolist(),
        "avg_accuracy": float(avg_accuracy),
        "avg_bwt": float(avg_bwt),
        "per_task_final": [float(accuracy_matrix[n_tasks-1][j]) for j in range(n_tasks)],
    }, model, tasks


def run_permuted_mnist(n_tasks=5, max_samples_per_task=500):
    """Run Permuted-MNIST benchmark."""
    print("\n  Loading MNIST...")
    train_imgs, train_lbls, test_imgs, test_lbls = load_mnist()

    print(f"  Creating {n_tasks} Permuted-MNIST tasks...")
    tasks = create_permutation_tasks(train_imgs, train_lbls, test_imgs, test_lbls,
                                     n_tasks=n_tasks)

    model = RLM(
        vocab_size=266,  # PixelTokenizer: 256 pixel bins + 10 classes
        embed_dim=32,
        concept_dim=32,
        n_concepts=100,
        n_hidden=32,
        n_layers=3,
        max_seq_len=785,  # 784 pixels + 1 label
    )

    accuracy_matrix = np.zeros((n_tasks, n_tasks))

    for train_id, task in enumerate(tasks):
        # Reset hidden state between tasks to prevent cross-task contamination
        model._prev_hidden_state = None

        print(f"\n  Training on {task['name']}...")
        t0 = time.time()
        _, experiences = train_on_task(model, task["train_imgs"], task["train_lbls"],
                      max_samples=max_samples_per_task)
        dt = time.time() - t0
        print(f"    Trained in {dt:.1f}s")

        # Compute Fisher information from this task's experiences before snapshotting
        model.compute_fisher(experiences, n_samples=50)
        del experiences  # free memory
        model.snapshot_weights()

        for eval_id in range(train_id + 1):
            acc = evaluate_on_task(model, tasks[eval_id]["test_imgs"],
                                   tasks[eval_id]["test_lbls"],
                                   max_samples=max_samples_per_task)
            accuracy_matrix[train_id][eval_id] = acc
            print(f"    Task {eval_id} accuracy: {acc:.1%}")

    avg_accuracy = np.mean(accuracy_matrix[n_tasks-1, :])
    bwt_scores = []
    for j in range(n_tasks - 1):
        bwt_scores.append(accuracy_matrix[n_tasks-1][j] - accuracy_matrix[j][j])
    avg_bwt = np.mean(bwt_scores) if bwt_scores else 0.0

    return {
        "accuracy_matrix": accuracy_matrix.tolist(),
        "avg_accuracy": float(avg_accuracy),
        "avg_bwt": float(avg_bwt),
        "per_task_final": [float(accuracy_matrix[n_tasks-1][j]) for j in range(n_tasks)],
    }


def classification_head_diagnostic(model, tasks, max_samples=200):
    """Diagnostic: train a simple MLP on h_final to measure signal in hidden state.

    If accuracy is high (75%+), the GRU is learning digit-distinguishing features
    but the concept graph path can't extract them. If low (~50%), the hidden state
    itself lacks signal.
    """
    print("\n  Classification Head Diagnostic")
    print("  " + "-" * 50)

    # Collect hidden states for all tasks
    all_train_h, all_train_lbls = [], []
    all_test_h, all_test_lbls = [], []

    for task in tasks:
        # Train hidden states
        n_train = min(max_samples, len(task["train_imgs"]))
        indices = np.random.permutation(len(task["train_imgs"]))[:n_train]
        for idx in indices:
            tokens = _pixel_tokenizer.encode_image(task["train_imgs"][idx])
            model.forward(tokens.reshape(1, -1))
            h = model._last_hidden_state.copy()
            all_train_h.append(h)
            all_train_lbls.append(task["train_lbls"][idx])

        # Test hidden states
        n_test = min(max_samples, len(task["test_imgs"]))
        indices = np.random.permutation(len(task["test_imgs"]))[:n_test]
        for idx in indices:
            tokens = _pixel_tokenizer.encode_image(task["test_imgs"][idx])
            model.forward(tokens.reshape(1, -1))
            h = model._last_hidden_state.copy()
            all_test_h.append(h)
            all_test_lbls.append(task["test_lbls"][idx])

    X_train = np.array(all_train_h)  # (N, 32)
    y_train = np.array(all_train_lbls)
    X_test = np.array(all_test_h)
    y_test = np.array(all_test_lbls)

    print(f"  Collected {len(X_train)} train / {len(X_test)} test hidden states")

    # Simple 2-layer MLP: 32 -> 16 -> 2, trained with numpy backprop
    rng = np.random.RandomState(42)
    W1 = rng.randn(32, 16).astype(np.float32) * 0.1
    b1 = np.zeros(16, dtype=np.float32)
    W2 = rng.randn(16, 2).astype(np.float32) * 0.1
    b2 = np.zeros(2, dtype=np.float32)

    lr = 0.01
    for epoch in range(200):
        # Forward
        z1 = X_train @ W1 + b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ W2 + b2
        # Softmax
        exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        # Cross-entropy loss
        N = len(y_train)
        loss = -np.mean(np.log(probs[np.arange(N), y_train] + 1e-10))

        # Backward
        dz2 = probs.copy()
        dz2[np.arange(N), y_train] -= 1
        dz2 /= N
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)
        da1 = dz2 @ W2.T
        dz1 = da1 * (z1 > 0)
        dW1 = X_train.T @ dz1
        db1 = np.sum(dz1, axis=0)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    # Evaluate
    z1 = X_test @ W1 + b1
    a1 = np.maximum(0, z1)
    z2 = a1 @ W2 + b2
    preds = np.argmax(z2, axis=1)
    acc = np.mean(preds == y_test)

    # Also measure per-task accuracy
    offset = 0
    per_task_acc = []
    for task in tasks:
        n_test = min(max_samples, len(task["test_imgs"]))
        task_preds = preds[offset:offset + n_test]
        task_labels = y_test[offset:offset + n_test]
        per_task_acc.append(np.mean(task_preds == task_labels))
        offset += n_test

    print(f"  MLP(32->16->2) accuracy: {acc:.1%}")
    print(f"  Per-task: {[f'{a:.1%}' for a in per_task_acc]}")
    print(f"  Baseline (random): 50.0%")

    return float(acc)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Fewer samples for fast testing")
    parser.add_argument("--tasks", type=int, default=5, help="Number of permuted tasks")
    args = parser.parse_args()

    max_samples = 200 if args.quick else 500

    print("=" * 70)
    print("  RAVANA on Standard CL Benchmarks")
    print("=" * 70)

    # ── Split-MNIST ──
    print("\n" + "=" * 70)
    print("  SPLIT-MNIST")
    print("=" * 70)
    split_results, split_model, split_tasks = run_split_mnist(max_samples_per_task=max_samples)

    print(f"\n  Split-MNIST Results:")
    print(f"    Average Accuracy: {split_results['avg_accuracy']:.1%}")
    print(f"    Average BWT: {split_results['avg_bwt']:.1%}")
    print(f"    Per-task final accuracy: {[f'{a:.1%}' for a in split_results['per_task_final']]}")

    # ── Classification Head Diagnostic ──
    print("\n" + "=" * 70)
    print("  CLASSIFICATION HEAD DIAGNOSTIC")
    print("=" * 70)
    head_acc = classification_head_diagnostic(split_model, split_tasks,
                                              max_samples=max_samples)

    # ── Permuted-MNIST ──
    print("\n" + "=" * 70)
    print(f"  PERMUTED-MNIST ({args.tasks} tasks)")
    print("=" * 70)
    perm_results = run_permuted_mnist(n_tasks=args.tasks, max_samples_per_task=max_samples)

    print(f"\n  Permuted-MNIST Results:")
    print(f"    Average Accuracy: {perm_results['avg_accuracy']:.1%}")
    print(f"    Average BWT: {perm_results['avg_bwt']:.1%}")
    print(f"    Per-task final accuracy: {[f'{a:.1%}' for a in perm_results['per_task_final']]}")

    # ── Save results ──
    output = {
        "split_mnist": split_results,
        "permuted_mnist": perm_results,
        "classification_head_diagnostic": head_acc,
        "config": {"max_samples": max_samples, "n_permuted_tasks": args.tasks},
    }
    out_path = os.path.join(_PROJECT_ROOT, "revisions", "mnist_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # ── Comparison to published baselines (from paper Table 6) ──
    print("\n  Comparison to Published Baselines (forgetting %):")
    print("    EWC (Kirkpatrick 2017):          6-12%")
    print("    Synaptic Intelligence (Zenke):   5-10%")
    print("    GEM (Lopez-Paz 2017):            1-3%")
    print("    A-GEM (Chaudhry 2019):           1-4%")
    print("    PackNet (Mallya 2018):           0%")
    print(f"    RAVANA (this run):               {-split_results['avg_bwt']:.1%} forgetting")


if __name__ == "__main__":
    main()
