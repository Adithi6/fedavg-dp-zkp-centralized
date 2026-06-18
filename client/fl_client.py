import logging
import time
import sys
import json
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model.cnn import LeNet
from crypto import zkp_utils
from utils.weights import (
    apply_weight_arrays,
    weights_to_bytes,
    model_to_weight_arrays,
)


def build_model(
    model_name: str,
    device: str,
    input_channels: int,
    num_classes: int,
    input_height: int,
    input_width: int,
    conv1_channels: int,
    conv2_channels: int,
) -> nn.Module:
    """
    Build model.
    """
    model_name = model_name.lower()

    if model_name == "lenet":
        model = LeNet(
            input_channels=input_channels,
            num_classes=num_classes,
            input_height=input_height,
            input_width=input_width,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
        )
        return model.to(device)

    raise ValueError(f"Unsupported model: {model_name}")


def _estimate_payload_size_kb(payload: dict) -> float:
    """
    Estimate payload size more accurately than sys.getsizeof(dict).

    Counts:
        - bytes objects by len()
        - strings by encoded length
        - dict/list objects by JSON size when possible
    """
    total_bytes = 0

    for value in payload.values():
        if isinstance(value, bytes):
            total_bytes += len(value)
        elif isinstance(value, str):
            total_bytes += len(value.encode("utf-8"))
        else:
            try:
                total_bytes += len(json.dumps(value).encode("utf-8"))
            except Exception:
                total_bytes += sys.getsizeof(value)

    return total_bytes / 1024.0


class FederatedClient:
    """
    Federated client for Method 3:

        Centralized FedAvg + DP + Circom zk-SNARK / ZKP validation

    Flow:
        1. Receive global model weights.
        2. Train locally.
        3. Apply client-level DP:
            update delta -> clip -> add Gaussian noise.
        4. Serialize model update.
        5. Generate Schnorr-style proof / identity proof.
        6. Generate Circom zk-SNARK proof over sampled update values.
        7. Send update + proof payload directly to central server.

    DP rule for fair comparison:
        if auto_noise=True:
            noise_std = base_noise / epsilon

    This must match:
        - FedAvg + DP baseline
        - FedAvg + DP + Dilithium
        - FedAvg + DP + ZKP
    """

    def __init__(
        self,
        client_id: str,
        dataloader: DataLoader,
        device: str,
        weight_dtype: str,
        learning_rate: float,
        model_name: str,
        input_channels: int,
        num_classes: int,
        input_height: int,
        input_width: int,
        conv1_channels: int,
        conv2_channels: int,
        dp_config: Optional[dict] = None,
        zkp_config: Optional[dict] = None,
    ):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.weight_dtype = weight_dtype
        self.learning_rate = learning_rate
        self.model_name = model_name

        if dp_config is None:
            dp_config = {}

        if zkp_config is None:
            zkp_config = {}

        # ------------------------------------------------------------
        # DP Configuration
        # ------------------------------------------------------------
        self.dp_enabled = bool(dp_config.get("enabled", True))
        self.dp_clip_norm = float(dp_config.get("clip_norm", 0.5))

        self.epsilon = float(dp_config.get("epsilon", 0.9))
        self.delta = float(dp_config.get("delta", 1e-5))

        self.auto_noise = bool(dp_config.get("auto_noise", True))
        self.base_noise = float(dp_config.get("base_noise", 0.05))

        if self.auto_noise:
            self.dp_noise_std = self.base_noise / max(self.epsilon, 1e-8)
        else:
            self.dp_noise_std = float(dp_config.get("noise_std", 0.01))

        self.optimizer_name = str(dp_config.get("optimizer", "adam")).lower()

        # ------------------------------------------------------------
        # ZKP Configuration
        # ------------------------------------------------------------
        self.zkp_enabled = bool(zkp_config.get("enabled", True))
        self.proof_system = str(zkp_config.get("proof_system", "circom_snark"))
        self.verify_before_aggregation = bool(
            zkp_config.get("verify_before_aggregation", True)
        )

        self.snark_sample_size = int(zkp_config.get("sample_size", 10))
        self.snark_scale = int(zkp_config.get("scale", 1000))
        self.snark_threshold = int(zkp_config.get("threshold", 100000000))

        # ------------------------------------------------------------
        # ZKP Keys
        # ------------------------------------------------------------
        self.pk, self.sk, keygen_ms = zkp_utils.keygen()

        # ------------------------------------------------------------
        # Model
        # ------------------------------------------------------------
        self.model = build_model(
            model_name=self.model_name,
            device=self.device,
            input_channels=input_channels,
            num_classes=num_classes,
            input_height=input_height,
            input_width=input_width,
            conv1_channels=conv1_channels,
            conv2_channels=conv2_channels,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()

        logging.info(
            f"[{self.client_id}] initialized | "
            f"approach=Centralized FedAvg+DP+ZKP | "
            f"model={self.model_name} | "
            f"optimizer={self.optimizer_name} | "
            f"learning_rate={self.learning_rate} | "
            f"weight_dtype={self.weight_dtype}"
        )

        logging.info(
            f"[{self.client_id}] DP settings | "
            f"enabled={self.dp_enabled} | "
            f"clip_norm={self.dp_clip_norm} | "
            f"epsilon={self.epsilon} | "
            f"delta={self.delta} | "
            f"auto_noise={self.auto_noise} | "
            f"base_noise={self.base_noise} | "
            f"calculated_noise_std={self.dp_noise_std:.4f}"
        )

        logging.info(
            f"[{self.client_id}] ZKP settings | "
            f"enabled={self.zkp_enabled} | "
            f"proof_system={self.proof_system} | "
            f"sample_size={self.snark_sample_size} | "
            f"scale={self.snark_scale} | "
            f"threshold={self.snark_threshold} | "
            f"keygen_time={keygen_ms:.3f} ms"
        )

    def _build_optimizer(self):
        """
        Build optimizer.

        Keep optimizer same across all three methods for fair comparison.
        """
        if self.optimizer_name == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
            )

        if self.optimizer_name == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=1e-4,
            )

        raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def _reset_optimizer(self):
        """
        Reset optimizer every FL round after syncing global weights.
        """
        self.optimizer = self._build_optimizer()

    def _raw_model(self):
        return self.model

    def get_raw_model(self):
        """
        Use this in main.py / node.py when model is needed.
        """
        return self.model

    def _save_initial_trainable_state(self):
        """
        Save model parameters before local training.
        """
        return {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def _apply_client_level_dp(self, initial_state: dict):
        """
        Apply client-level DP-style update clipping and Gaussian noise.

        delta = local_model - global_model
        clipped_delta = delta * min(1, C / ||delta||_2)
        private_delta = clipped_delta + Gaussian noise
        final_model = global_model + private_delta
        """
        total_norm_sq = 0.0

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            delta = param.data - initial_state[name]
            total_norm_sq += torch.sum(delta ** 2).item()

        total_norm = total_norm_sq ** 0.5

        clip_factor = min(
            1.0,
            self.dp_clip_norm / (total_norm + 1e-12),
        )

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                delta = param.data - initial_state[name]
                clipped_delta = delta * clip_factor

                if self.dp_enabled and self.dp_noise_std > 0:
                    noise = torch.normal(
                        mean=0.0,
                        std=self.dp_noise_std,
                        size=clipped_delta.shape,
                        device=clipped_delta.device,
                        dtype=clipped_delta.dtype,
                    )
                else:
                    noise = torch.zeros_like(clipped_delta)

                param.data.copy_(initial_state[name] + clipped_delta + noise)

        return total_norm, clip_factor

    def _local_accuracy(self):
        """
        Local sanity-check accuracy, not global test accuracy.
        """
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in self.dataloader:
                x = x.to(self.device)
                y = y.to(self.device)

                outputs = self.model(x)
                predictions = outputs.argmax(dim=1)

                correct += (predictions == y).sum().item()
                total += y.size(0)

        return correct / total if total > 0 else 0.0

    def local_train(self, global_weight_arrays=None, epochs: int = 1):
        """
        Perform local client training and then apply client-level DP.
        """
        start_time = time.time()

        if global_weight_arrays is not None:
            apply_weight_arrays(self._raw_model(), global_weight_arrays)

        if epochs == 0:
            logging.info(
                f"[{self.client_id}] weights synchronized without DP training"
            )
            return None, self.delta

        initial_state = self._save_initial_trainable_state()

        self._reset_optimizer()
        self.model.train()

        total_loss = 0.0
        total_batches = 0

        for epoch in range(epochs):
            for batch_idx, (x, y) in enumerate(self.dataloader):
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                loss.backward()

                # Stability clipping, separate from DP update clipping.
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )

                self.optimizer.step()

                total_loss += loss.item()
                total_batches += 1

                if epoch == 0 and batch_idx == 0:
                    pred = torch.argmax(outputs, dim=1)
                    logging.info(
                        f"[{self.client_id}] sample prediction check | "
                        f"pred={pred[0].item()} | actual={y[0].item()}"
                    )

        if self.dp_enabled:
            update_norm, clip_factor = self._apply_client_level_dp(initial_state)
        else:
            update_norm = 0.0
            clip_factor = 1.0

        local_acc = self._local_accuracy()

        exec_ms = (time.time() - start_time) * 1000
        avg_loss = total_loss / total_batches if total_batches > 0 else 0.0

        logging.info(
            f"[{self.client_id}] local train completed | "
            f"loss={avg_loss:.4f} | "
            f"local_acc={local_acc * 100:.2f}% | "
            f"dp_enabled={self.dp_enabled} | "
            f"update_norm={update_norm:.4f} | "
            f"clip_factor={clip_factor:.4f} | "
            f"epsilon={self.epsilon} | "
            f"noise_std={self.dp_noise_std:.4f} | "
            f"time={exec_ms:.2f}ms"
        )

        return None, self.delta

    def _sample_update_values_for_snark(self) -> list[int]:
        """
        Sample model values for Circom UpdateNorm circuit.

        Current prototype:
            - flattens model weights
            - takes first sample_size values
            - scales by integer factor
            - uses absolute positive integers for Circom

        Paper limitation:
            This validates sampled values, not the entire model update.
        """
        arrays = model_to_weight_arrays(self._raw_model())
        flat = np.concatenate([arr.flatten() for arr in arrays])

        sample_size = min(self.snark_sample_size, flat.size)

        sampled_weights = (
            np.abs(flat[:sample_size] * self.snark_scale)
            .astype(int)
            .tolist()
        )

        # If circuit expects fixed size, pad with zeros.
        if sample_size < self.snark_sample_size:
            sampled_weights.extend([0] * (self.snark_sample_size - sample_size))

        return sampled_weights

    def prepare_update(self):
        """
        Prepare update with ZKP proof.

        DP is already applied in local_train().
        """
        prep_start = time.time()

        update_bytes = weights_to_bytes(
            self._raw_model(),
            self.weight_dtype,
        )

        zkp_start = time.time()

        if self.zkp_enabled:
            # --------------------------------------------------------
            # Schnorr-style proof / identity proof
            # --------------------------------------------------------
            schnorr_proof = zkp_utils.generate_proof(
                self.sk,
                update_bytes,
                self.client_id,
            )

            # --------------------------------------------------------
            # Circom zk-SNARK proof for sampled update norm constraint
            # --------------------------------------------------------
            sampled_weights = self._sample_update_values_for_snark()

            snark_proof, snark_public, snark_ms = zkp_utils.generate_snark(
                sampled_weights,
                threshold=self.snark_threshold,
            )
        else:
            schnorr_proof = None
            snark_proof = None
            snark_public = None
            snark_ms = 0.0
            sampled_weights = []

        zkp_end = time.time()

        payload = {
            "client_id": self.client_id,
            "update_bytes": update_bytes,

            # ZKP fields
            "zkp": schnorr_proof,
            "snark_proof": snark_proof,
            "snark_public": snark_public,
            "public_key": self.pk,
            "zkp_enabled": self.zkp_enabled,
            "proof_system": self.proof_system,
            "sample_size": self.snark_sample_size,
            "snark_threshold": self.snark_threshold,

            # Timing
            "proof_gen_ms": float((zkp_end - zkp_start) * 1000),
            "snark_ms": float(snark_ms),
            "total_prep_ms": float((time.time() - prep_start) * 1000),

            # Metadata
            "num_samples": len(self.dataloader.dataset),

            # DP metadata
            "dp_enabled": self.dp_enabled,
            "dp_clip_norm": self.dp_clip_norm,
            "dp_noise_std": self.dp_noise_std,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "auto_noise": self.auto_noise,
            "base_noise": self.base_noise,
        }

        payload_size_kb = _estimate_payload_size_kb(payload)

        payload["payload_size_kb"] = float(payload_size_kb)

        prep_end = time.time()

        logging.info(
            f"[{self.client_id}] ZKP update prepared | "
            f"epsilon={self.epsilon} | "
            f"noise_std={self.dp_noise_std:.4f} | "
            f"update_size={len(update_bytes) / 1024:.2f} KB | "
            f"proof_system={self.proof_system} | "
            f"sample_size={self.snark_sample_size} | "
            f"proof_gen_time={(zkp_end - zkp_start) * 1000:.2f} ms | "
            f"snark_time={snark_ms:.2f} ms | "
            f"payload_size={payload_size_kb:.2f} KB | "
            f"total_prep_time={(prep_end - prep_start) * 1000:.2f} ms"
        )

        return payload