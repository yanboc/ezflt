# config.py
import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 64
TRAIN_SIZE = 4000
TEST_SIZE = 1000
EPOCHS = 100
LEARNING_RATE = 0.1

CONFIG = {
    "experiment_id": "example_experiment_RBAD",
    "description": "Example: numerical evaluation of reconstruction-based anomaly detection",
    "seed": 42,
    "repeats": 1,
    "device": DEVICE,
    "data":{
        "generate_data": True,
        "data_type": "tensor",
        "store_data": True,
        "store_data_path": os.path.join(BASE_DIR, "data", "cache"),
        "parameters": {
            # "P": [20], 
            # "d_P_ratio": [1.1],
            "P": [20 + 8 * n for n in range(6)], # number of patches per sample
            "d_P_ratio": [1.2], # d = floor(P * d_P_ratio)    
            "C_d_ratio": [1.2], # C = floor(d * C_d_ratio)
            "NNF_ratio": [0.5], # NNF = floor(P * NNF_ratio), NNF = number of normal features
            # "NRP_ratio": [0, 0.1, 0.2] # NRP = floor(P * NRP_ratio), NRP = number of replaced patches
            "NRP_ratio": [0.1]
        },
        "train_size": TRAIN_SIZE, # number of training samples
        "test_size": TEST_SIZE, # number of test samples
        "batch_size": BATCH_SIZE, # batch size
        "shuffle": True # shuffle the data
    },
    "model":{
        "task": "generation",
        "arch": "shallow",
        "parameters": {
            "description (optional)": "",
            "input_dim": [BATCH_SIZE, 28, 28], # (bsz, P, d)
            "output_dim": 10, # (bsz, P, d)
            "num_classes": 10
        }
    },
    "training":{
        "training": True,
        "save_checkpoints": True,
        "checkpoint_path": os.path.join(BASE_DIR, "assets", "checkpoints"),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "optimizer": "sgd",
        "supported_optimizers": ["sgd", "adam", "sam"],
        "loss_function": "cross_entropy",
        "supported_loss_functions": ["cross_entropy", "mse"]
    },
    "evaluation":{
        "evaluate": True,
        "metrics": ["accuracy"],
        "supported_metrics": ["accuracy", "precision", "recall", "f1_score"]
    },
    "plotting":{
        "save_plots": True,
        "feature_visualization": True,
        "plot_path": os.path.join(BASE_DIR, "assets", "plots")
    }
}