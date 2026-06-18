import logging
import yaml
import time
import copy
import torch
import numpy as np

from data.loader import make_client_loaders
from client.fl_client import FederatedClient
from utils.weights import model_to_weight_arrays, apply_weight_arrays, bytes_to_weight_arrays
from crypto import zkp_utils


def setup_logging(config):
    logging.basicConfig(
        level=getattr(logging, config["logging"]["log_level"].upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(config["logging"]["log_file"]),
            logging.StreamHandler(),
        ],
    )


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


class CentralServer:
    def __init__(self, template_model, device, weight_dtype):
        self.global_model = template_model.to(device)
        self.device = device
        self.weight_dtype = weight_dtype
        logging.info("Central Server initialized")

    def get_global_weights(self):
        return model_to_weight_arrays(self.global_model)

    def verify_zkp(self, message: dict) -> tuple[bool, float]:
        required_fields = [
            "client_id",
            "update_bytes",
            "zkp",
            "snark_proof",
            "snark_public",
            "public_key",
        ]

        for field in required_fields:
            if field not in message:
                logging.warning(
                    f"Message from {message.get('client_id', 'unknown')} "
                    f"missing field: {field}"
                )
                return False, 0.0

        try:
            # 1. Verify Schnorr Signature
            is_valid_sig, sig_ms = zkp_utils.verify_proof(
                public_key=message["public_key"],
                update_bytes=message["update_bytes"],
                client_id=message["client_id"],
                proof=message["zkp"],
            )
            if not is_valid_sig:
                logging.warning(f"Invalid Schnorr signature from {message['client_id']}")
                return False, sig_ms
                
            # 2. Verify zk-SNARK
            is_valid_snark, snark_ms = zkp_utils.verify_snark(
                proof=message["snark_proof"],
                public_signals=message["snark_public"]
            )
            
            if not is_valid_snark:
                logging.warning(f"Invalid zk-SNARK proof from {message['client_id']}")
                return False, sig_ms + snark_ms
                
            total_ms = float(sig_ms + snark_ms)
            
        except Exception as e:
            logging.error(f"ZKP verification error: {e}")
            return False, 0.0

        return True, total_ms

    def aggregate_updates(self, client_payloads: list[dict]):
        start_time = time.time()
        
        weight_sets = []
        accepted_clients = []
        rejected_clients = []
        sample_counts = []
        
        total_verify_ms = 0.0

        for payload in client_payloads:
            client_id = payload.get("client_id", "unknown")
            num_samples = payload.get("num_samples", 1)
            
            # ZKP Verification
            is_valid, verify_ms = self.verify_zkp(payload)
            total_verify_ms += verify_ms
            
            logging.info(
                f"[{client_id}] ZKP Verification: "
                f"verify_ms={verify_ms:.3f} ms | "
                f"[{'ACCEPTED' if is_valid else 'REJECTED'}]"
            )

            if not is_valid:
                rejected_clients.append(client_id)
                continue

            # Decode bytes to arrays
            try:
                arrays = bytes_to_weight_arrays(
                    payload["update_bytes"],
                    self.global_model,
                    dtype_name=self.weight_dtype,
                )
            except Exception as e:
                rejected_clients.append(client_id)
                logging.warning(f"Failed to decode update from {client_id} | error={e}")
                continue

            weight_sets.append(arrays)
            accepted_clients.append(client_id)
            sample_counts.append(num_samples)

        if not weight_sets:
            logging.warning("[Server] No valid updates after verification. Skipping aggregation.")
            return accepted_clients, rejected_clients, total_verify_ms

        # Weighted FedAvg
        total_samples = sum(sample_counts)
        averaged = []
        for i in range(len(weight_sets[0])):
            layer_weighted_sum = np.zeros_like(weight_sets[0][i])
            for j in range(len(weight_sets)):
                layer_weighted_sum += weight_sets[j][i] * sample_counts[j]
            averaged.append(layer_weighted_sum / total_samples)

        apply_weight_arrays(self.global_model, averaged)

        end_time = time.time()
        logging.info(
            f"[Server] Centralized FedAvg aggregation completed | "
            f"Accepted: {len(accepted_clients)} | "
            f"Rejected: {len(rejected_clients)} | "
            f"Time: {end_time - start_time:.4f}s"
        )
        
        return accepted_clients, rejected_clients, total_verify_ms


def main():
    config = load_config()
    setup_logging(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    N_CLIENTS = config["experiment"]["n_clients"]
    N_ROUNDS = config["experiment"]["n_rounds"]
    LOCAL_EPOCHS = config["experiment"]["local_epochs"]

    LEARNING_RATE = config["training"]["learning_rate"]

    MODEL = config["model"]
    DATA = config["data"]
    WEIGHTS = config["weights"]

    DP_CONFIG = config.get("dp", {})
    ZKP_CONFIG = config.get("zkp", {})

    logging.info(
        "DP settings loaded from config | "
        f"enabled={DP_CONFIG.get('enabled', True)} | "
        f"epsilon={DP_CONFIG.get('epsilon', 1.0)} | "
        f"delta={DP_CONFIG.get('delta', 1e-5)} | "
        f"clip_norm={DP_CONFIG.get('clip_norm', 0.5)} | "
        f"auto_noise={DP_CONFIG.get('auto_noise', True)} | "
        f"base_noise={DP_CONFIG.get('base_noise', 0.05)}"
    )

    logging.info(
        "Architecture: Centralized FedAvg + Differential Privacy + zk-SNARK validation"
    )

    client_loaders, test_loader = make_client_loaders(
        n_clients=N_CLIENTS,
        batch_size=DATA["batch_size"],
        alpha=DATA["alpha"],
        dataset_name=DATA["dataset_name"],
        partition_by=DATA["partition_by"],
        min_partition_size=DATA["min_partition_size"],
        self_balancing=DATA["self_balancing"],
        seed=DATA["seed"],
        test_batch_size=DATA["test_batch_size"],
        normalize_mean=DATA["normalize_mean"],
        normalize_std=DATA["normalize_std"],
    )

    clients = []

    for i in range(N_CLIENTS):
        client = FederatedClient(
            client_id=f"client_{i}",
            dataloader=client_loaders[i],
            device=device,
            weight_dtype=WEIGHTS["dtype"],
            learning_rate=LEARNING_RATE,
            model_name=MODEL["name"],
            input_channels=MODEL["input_channels"],
            num_classes=MODEL["num_classes"],
            input_height=MODEL["input_height"],
            input_width=MODEL["input_width"],
            conv1_channels=MODEL["conv1_channels"],
            conv2_channels=MODEL["conv2_channels"],
            dp_config=DP_CONFIG,
            zkp_config=ZKP_CONFIG,
        )
        clients.append(client)

    # Initialize Central Server
    template_model = copy.deepcopy(clients[0].get_raw_model())
    server = CentralServer(template_model, device, WEIGHTS["dtype"])

    start_time = time.time()

    for r in range(1, N_ROUNDS + 1):
        logging.info("=" * 60)
        logging.info(f"Round {r}/{N_ROUNDS}")
        logging.info("=" * 60)

        # 1. Server broadcasts global weights
        global_weights = server.get_global_weights()

        client_payloads = []

        # 2. Clients train locally and generate proofs
        for client in clients:
            client.local_train(global_weights, epochs=LOCAL_EPOCHS)
            payload = client.prepare_update()
            client_payloads.append(payload)

        # 3. Server aggregates updates after ZKP verification
        accepted, rejected, verify_ms = server.aggregate_updates(client_payloads)
        
        logging.info(f"Round {r} total ZKP verification time: {verify_ms:.3f} ms")
        logging.info(f"Round {r} accepted updates: {len(accepted)}")
        logging.info(f"Round {r} rejected updates: {len(rejected)}")

        # 4. Evaluate global model
        accuracy = evaluate_model(server.global_model, test_loader, device)

        logging.info(f"Round {r} global test accuracy: {accuracy * 100:.2f}%")

    end_time = time.time()
    logging.info(f"Total time = {end_time - start_time:.2f}s")


if __name__ == "__main__":
    main()