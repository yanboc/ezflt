"""
This example demonstrates how to use ezflt to perform reconstruction-based anomaly detection (RBAD).

Usage:
    1. Install ezflt: pip install -e . (cf README.md for more details)
    2. Run the example: python example.py

References:
    1. Two-Layer Convolutional Autoencoders Trained on Normal Data Provably Detect Unseen Anomalies, Yanbo Chen & Weiwei Liu, https://openreview.net/forum?id=FnbGlnKbIU
"""
import sys
import os
from os import path
import math
import numpy
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ezflt.data import Generator
from ezflt.model import FLTNetwork
from ezflt.plot import visualize
from config import CONFIG
from config import BASE_DIR

class RBAD_generator(Generator):
    def __init__(self, config=CONFIG):
        super().__init__(config)

    def process_parameters(self, P, d_P_ratio, P_C_ratio, NNF_ratio, NRP_ratio):
        d = math.floor(P * d_P_ratio)
        C = math.floor(P * P_C_ratio)
        NNF = math.floor(P * NNF_ratio)
        NRP = math.ceil(P * NRP_ratio)
        return [P, d, C, NNF, NRP]

    def generate_feature(self, d):
        # Generate orthogonal matrix from random matrix using QR decomposition
        M = torch.randn(d, d, device=self.device)
        Q, _ = torch.linalg.qr(M)
        self._feature = Q
        return Q

    def generate_beta(self, d, NNF):
        # Generate beta, the probability of each feature being selected
        # Since only auxiliary features are randomly selected, we let beta[:NNF] be zeros.
        beta = torch.zeros(d, device=self.device)
        rand_vals = torch.rand(d-NNF, device=self.device)
        beta[NNF:] = rand_vals
        return beta

    def generate_normal(self, feature, beta, P, N, sigma=0.01):
        """ Generate normal data

        Args:
            feature: feature (orthonormal) matrix (d, d), d = feature.shape[1]
            beta: beta probability vector (d), beta[:NNF] = 0 (don't need to be selected), beta[NNF:] = random values
            P: number of patches per sample (P = 20 + 4 * n for n in range(11))
            N: number of training/test samples (N = train_size or test_size)
            sigma: noise level (sigma = 0.01 by default)
        Returns:
            x: normal data (N, P, d)
        """
        feature_index = torch.multinomial(beta.to(self.device), N*(P-NNF), replacement=True)
        feature_index = feature_index.view(N, P-NNF)
        nor_features = feature[:NNF, :]
        x_nor = nor_features.unsqueeze(0).repeat(N, 1, 1)
        x_aux = feature[feature_index]
        x = torch.cat([x_nor, x_aux], dim=1) + torch.randn([N, P, feature.shape[1]], device=self.device) * sigma
        return x

    def generate_semantic_anomaly(self, feature, beta, P, N_sa, NRP, sigma=0.01):
        """ Generate semantic anomaly data by replacing patches with auxiliary features + noise

        Args:
            NRP: number of replaced patches (NRP = floor(P * NRP_ratio))
            sigma: noise level (sigma = 0.01 by default)
        Returns:
            x: semantic anomaly data (N_sa, P, d)
        """

        x = self.generate_normal(self, feature=feature, beta=beta, P=P, N=N_sa)
        feature_index = torch.multinomial(beta.to(self.device), N_sa*NRP, replacement=True)
        feature_index = feature_index.view(N_sa, NRP)
        x[:, :NRP, :] = feature[feature_index] + torch.randn([N_sa, NRP, feature.shape[1]], device=self.device) * sigma

        return x

    def generate_non_semantic_anomaly(self, feature, beta, P, N_nsa, NRP):
        """ Generate non-semantic anomaly data by replacing patches with pure noise

        Args:
            NRP: number of replaced patches (NRP = floor(P * NRP_ratio))
            sigma: noise level (sigma = 0.01 by default)
        Returns:
            x: non-semantic anomaly data (N_nsa, P, d)
        """

        x = self.generate_normal(self, feature=feature, beta=beta, P=P, N=N_nsa)
        x[:, :NRP, :] = torch.randn([N_nsa, NRP, feature.shape[1]], device=self.device) * 1.0 # replace with pure noise

        return x

class ConvAE(nn.Module, FLTNetwork):
    def __init__(self, config, C):
        nn.Module.__init__(self)
        FLTNetwork.__init__(self, config=config)
        self.device = config['device']

        # register the kernels  
        self.kernels = nn.Parameter(torch.randn(C, d, device=self.device) * 0.001, requires_grad=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        bsz, P, d = x.shape
        C = self.kernels.shape[1]

        # Encoder
        x = torch.matmul(x, self.kernels.T) # (bsz, P, d) @ (d, C) = (bsz, P, C)
        x = self.relu(x) # (bsz, P, C)

        # use the max pooling to get the indices of the most active features
        indices = torch.argmax(x.transpose(1, 2), dim=-1, keepdim=True) # (bsz, C)
        expended_kernels = self.kernels.repeat(x.shape[0], 1, 1) # (C, d) -> (bsz, C, d)

        # Decoder
        output = torch.zeros([bsz, P, d], device=self.device)
        output = output.scatter_add_(dim=1, index=indices.expand(-1, -1, d), src=expended_kernels)

        return output

def train(model: nn.Module, data_loader, optimizer, num_epochs: int = 20, learning_rate: float = 1e-3):
    for epoch in range(num_epochs):
        for batch in data_loader:
            x = batch["input"]
            y = batch["target"]
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

def feature_visualization(model, traced_features):
    visualize(traced_features, colored=True, output_size=[1, 0])

def report_accuracy(model):
    pass

if __name__ == "__main__":
    # Generate data
    generator = RBAD_generator(config=CONFIG)

    all_kernels_norm = {}
    all_kernels_inner_product = {}

    epochs = CONFIG["training"]["epochs"]
    lr = CONFIG["training"]["learning_rate"]
    num_epochs = epochs // 10

    for parameters in generator.parameters_combinations:
        P, d, C, NNF, NRP = generator.process_parameters(*parameters)
        exp_id = f"P={P}_d={d}_C={C}_NNF={NNF}"
        N = generator.train_size
        feature = generator.generate_feature(d)
        beta = generator.generate_beta(d=d, NNF=NNF)
        x = generator.generate_normal(feature=feature, beta=beta, P=P, N=N, sigma=0.01)


        model = ConvAE(config=CONFIG, C=C)
        model(x)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        recorded_kernels = torch.zeros(epochs, C, d).to(model.device)
        kernels_norm = torch.zeros(epochs, C).to(model.device)
        kernels_inner_product = torch.zeros(epochs, C).to(model.device)
        for epoch in tqdm(range(epochs), desc="Training"):
            loss = F.mse_loss(model(x), x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            recorded_kernels[epoch, :, :] = model.kernels.detach().clone()
            if epoch % num_epochs == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

            # Record the kernels
            kernels_norm[epoch, :] = torch.norm(recorded_kernels[epoch, :, :], dim=-1)
            current_inner_product = torch.tensordot(recorded_kernels[epoch, :, :], feature, dims=([-1], [-1]))
            kernels_inner_product[epoch, :] = current_inner_product.max(dim=-1).values

        all_kernels_norm[exp_id] = kernels_norm.to("cpu").numpy()
        all_kernels_inner_product[exp_id] = kernels_inner_product.to("cpu").numpy()

    plot_path = CONFIG["plotting"]["plot_path"]
    os.makedirs(plot_path, exist_ok=True)

    def plot_kernel_heatmaps(exp_id, use_tight_layout=False):
        """绘制核范数和内积热力图的公共函数"""
        C = all_kernels_inner_product[exp_id].shape[1]
        plot_values_ip = numpy.zeros((10, C))
        plot_values_norm = numpy.zeros((10, C))
        fig, axes = plt.subplots(2, 1, figsize=(C, 10*2))
        
        for i in range(10): 
            plot_values_norm[i, :] = all_kernels_norm[exp_id][i * num_epochs]
            plot_values_ip[i, :] = all_kernels_inner_product[exp_id][i * num_epochs] / plot_values_norm[i, :]
        
        ax_norm = sns.heatmap(
            data=plot_values_norm * 10, 
            annot=True, 
            fmt=".4f", 
            cmap="coolwarm", 
            square=False,
            ax=axes[1])
        ax_norm.set_title(f"Kernels Norm Heat Map for {exp_id} (scaled by 10)")
        ax_norm.set_xlabel("Kernel Index")
        ax_norm.set_ylabel("Epoch")

        ax_ip = sns.heatmap(
            data=plot_values_ip, 
            annot=True, 
            fmt=".3f", 
            cmap="coolwarm", 
            vmin=0, 
            vmax=1, 
            square=False,
            ax=axes[0])
        ax_ip.set_title(f"Inner Product Heat Map for {exp_id}")
        ax_ip.set_xlabel("Kernel Index")
        ax_ip.set_ylabel("Epoch")

        if use_tight_layout:
            plt.tight_layout()
        plt.savefig(os.path.join(plot_path, f"{exp_id}.png"))
        plt.close()

    # 第一次绘制（不使用tight_layout）
    for exp_id in all_kernels_inner_product.keys():
        plot_kernel_heatmaps(exp_id, use_tight_layout=False)

    # 第二次绘制（使用tight_layout）
    for exp_id in all_kernels_inner_product.keys():
        plot_kernel_heatmaps(exp_id, use_tight_layout=True)

    # for exp_id in all_kernels_norm.keys():
    #     C = all_kernels_norm[exp_id].shape[1]
    #     plot_values_norm = numpy.zeros((10, C))
    #     fig, ax = plt.subplots(figsize=(C, 10))
    #     for i in range(10):
    #         plot_values[i, :] = all_kernels_norm[exp_id][i * num_epochs]
    #     ax = sns.heatmap(data=plot_values, annot=True, fmt=".3f", cmap="coolwarm", vmin=0, vmax=1, square=False, ax=ax)
    #     ax.set_title(f"Kernels Norm Heat Map for {exp_id}")
    #     ax.set_xlabel("Kernel Index")
    #     ax.set_ylabel("Epoch")
    #     plt.savefig(os.path.join(plot_path, f"{exp_id}_kernels_norm_heat_map.png"))
    #     plt.close()
    # Initialize model
    # model = ConvAE(config_file="../config.json")

    # Train model
    # train(model, data_loader, num_epochs=20, learning_rate=1e-3)

    # Feature visualization
    # traced_features = model.trace_feature()
    # feature_visualization(model, traced_features)

    # Report accuracy
    # report_accuracy(model)