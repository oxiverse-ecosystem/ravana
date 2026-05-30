import os

# Prevent pytest from collecting test_* functions inside experiment scripts.
collect_ignore = [
    os.path.join("experiments", f)
    for f in os.listdir(os.path.join(os.path.dirname(__file__), "experiments"))
    if (f.startswith("experiment_") or f.startswith("eval_")) and f.endswith(".py")
]
