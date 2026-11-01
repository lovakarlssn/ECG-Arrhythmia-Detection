# Edge-Ready ECG Classification using CRISP-DM

An end-to-end, reproducible Python pipeline for 4-class ECG heartbeat classification targeting the Analog Devices MAX78002 Edge-AI Microcontroller. This repository contains the data preparation, exploratory data analysis, and modeling pipeline for the MIT-BIH Edge-AI ECG classification project.

## Executive Summary

This project implements a compact 1D Convolutional Neural Network trained on the MIT-BIH Arrhythmia Database to classify individual heartbeats into four AAMI EC57 classes (Normal *N*, Supraventricular ectopic *S*, Ventricular ectopic *V*, and Fusion *F*). The architecture strictly satisfies embedded hardware constraints (< 2 MB memory footprint) and utilizes Quantization-Aware Training (QAT) to ensure compatibility with the MAX78002 CNN accelerator.

## Repository Structure

- `configs/`: Configuration YAML files.
- `data/`: Raw and processed dataset arrays.
  > **Note:** This directory is excluded from the GitHub repository.
- `notebooks/`: CRISP-DM EDA and walkthrough notebook.
- `src/`: Modular Python source code for data preparation, datasets, models, training, evaluation, and noise robustness testing.
- `results/`: Evaluation metrics, confusion matrices, and figures.
- `max78002/`: Setup instructions and toolchain integration artifacts.