import subprocess
import sys
import os

def run_command(command_list):
    print(f"\n[RUNNING] {' '.join(command_list)}")
    result = subprocess.run(command_list)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("processed", exist_ok=True)

    print("========================================================")
    print("STARTING EDGE-READY ECG CLASSIFICATION PIPELINE")
    print("========================================================")

    # Data Preparation
    #run_command([sys.executable, "ADM ecg/src/prepare_data.py"])

    # Model Selection & Training (FP32 + QAT)
    #run_command([sys.executable, "ADM ecg/src/train.py"])

    # Evaluation (Clean Test Set, Per-Class, Confusion Matrices, Waveform Errors)
    #run_command([sys.executable, "ADM ecg/src/evaluate.py"])

    # Noise Stress Test Robustness
    run_command([sys.executable, "ADM ecg/src/noise_test.py"])

    print("\n========================================================")
    print("PIPELINE COMPLETE SUCCESSFULLY")


if __name__ == "__main__":
    main()