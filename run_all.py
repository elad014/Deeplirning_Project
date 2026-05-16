import os
import sys
import time
from datetime import timedelta

sys.path.insert(0, os.path.dirname(__file__))

import cnn_scratch
import transfer_learning
import autoencoder


def _separator(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n")


def main() -> None:
    total_start = time.time()

    # ------------------------------------------------------------------
    # Stage A — CNN from Scratch
    # ------------------------------------------------------------------
    _separator("STAGE A  |  CNN from Scratch (VGG_SmallSigmoid)")
    stage_start = time.time()
    cnn_scratch.main()
    stage_a_time = time.time() - stage_start
    print(f"\nStage A finished in {timedelta(seconds=int(stage_a_time))}")

    # ------------------------------------------------------------------
    # Stage B — Transfer Learning (ResNet-50 & VGG-16, frozen + full)
    # ------------------------------------------------------------------
    _separator("STAGE B  |  Transfer Learning")
    stage_start = time.time()
    transfer_learning.main()
    stage_b_time = time.time() - stage_start
    print(f"\nStage B finished in {timedelta(seconds=int(stage_b_time))}")

    # ------------------------------------------------------------------
    # Stage C — Convolutional Autoencoder + Classifier Fine-tuning
    # ------------------------------------------------------------------
    _separator("STAGE C  |  Semi-Supervised (Autoencoder + Classifier)")
    stage_start = time.time()
    autoencoder.main()
    stage_c_time = time.time() - stage_start
    print(f"\nStage C finished in {timedelta(seconds=int(stage_c_time))}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_time = time.time() - total_start
    _separator("ALL STAGES COMPLETE")
    print(f"  Stage A : {timedelta(seconds=int(stage_a_time))}")
    print(f"  Stage B : {timedelta(seconds=int(stage_b_time))}")
    print(f"  Stage C : {timedelta(seconds=int(stage_c_time))}")
    print(f"  Total   : {timedelta(seconds=int(total_time))}")
    print(f"\n  Results saved to: model_comparison.xlsx")


if __name__ == "__main__":
    main()
