# Centralized Federated Learning with Differential Privacy and Zero-Knowledge Proof Validation

## Overview

This project implements a centralized Federated Learning (FL) framework that combines Differential Privacy (DP) with Zero-Knowledge Proof (ZKP) based model update validation.

Clients train locally on private data and apply Differential Privacy through gradient clipping and Gaussian noise addition. Before sending updates to the central server, each client generates a Zero-Knowledge Proof demonstrating that its model update satisfies a predefined constraint without revealing the underlying model parameters or local training data.

The central server verifies the proofs before aggregation. Only updates with valid proofs are included in the Federated Averaging (FedAvg) process, preventing invalid or malicious updates from influencing the global model.

The implementation uses zk-SNARKs generated through Circom and snarkjs and is evaluated on the MNIST dataset using a LeNet-style Convolutional Neural Network (CNN) under a non-IID data distribution.

---

## Features

* Centralized Federated Learning architecture
* Federated Averaging (FedAvg)
* Differential Privacy (DP)

  * Gradient clipping
  * Gaussian noise injection
  * Configurable privacy budget (ε)
* Zero-Knowledge Proof (ZKP) validation
* zk-SNARK proof generation and verification
* Circom-based arithmetic circuits
* Server-side proof verification
* Rejection of invalid model updates
* Non-IID client data partitioning using Dirichlet distribution
* LeNet-style CNN model

---

## Security Workflow

### Client Side

1. Receive global model from the server.
2. Perform local training.
3. Apply Differential Privacy:

   * Gradient clipping
   * Gaussian noise addition
4. Generate a model update.
5. Construct the ZKP witness.
6. Generate a zk-SNARK proof.
7. Send:

   * DP-protected model update
   * Zero-Knowledge Proof
   * Public inputs
   * Client identifier

### Server Side

1. Receive updates from all clients.
2. Verify each zk-SNARK proof.
3. Reject invalid submissions.
4. Aggregate only verified updates using FedAvg.
5. Update and redistribute the global model.

---

## Zero-Knowledge Proof Validation

The project uses a Circom circuit to verify a mathematical constraint on the client update.

### Proof System

* zk-SNARK
* Circom
* snarkjs
* Groth16 proving system

### Validation Process

* Client generates proof locally.
* Server verifies proof before aggregation.
* Invalid proofs are rejected.
* Valid proofs are aggregated using FedAvg.

This allows model updates to be validated without exposing the underlying private training data.

---

## System Configuration

### Dataset

* MNIST
* Non-IID Dirichlet partitioning (α = 0.5)

### Model

* LeNet-style CNN
* Group Normalization

### Federated Learning

* Number of Clients: 10
* Communication Rounds: 150
* Local Epochs: 5
* Aggregation Method: FedAvg

### Differential Privacy

* Epsilon (ε): 1.0
* Delta (δ): 1e-5
* Clip Norm: 0.5

### ZKP Configuration

* Proof System: Circom zk-SNARK
* Verification Before Aggregation: Enabled
* Sample Size: 10
* Scale Factor: 1000
* Threshold: 100000000

---

## Project Structure

```text
client/
    fl_client.py

crypto/
    zkp_utils.py

data/
    loader.py

model/
    cnn.py

utils/
    weights.py

zkp/
    circuits/
        update_norm.circom
    package.json
    package-lock.json

config.yaml
main.py
README.md
```

---

## Installation

### Python Dependencies

```bash
pip install torch torchvision datasets pyyaml numpy pandas
```

### ZKP Dependencies

```bash
npm install
```

Install:

* Node.js
* Circom
* snarkjs

---


## Running the Project

```bash
python main.py
```

## Note

The repository already contains the required zk-SNARK runtime artifacts:

- update_norm.wasm
- update_norm_0001.zkey
- verification_key.json

Therefore, the circuit does not need to be recompiled before running the project.

The Powers of Tau files and Circom compiler are not included in the repository because they are only required when regenerating proving and verification keys.

During execution, the system will:

* Train local client models
* Apply Differential Privacy
* Generate zk-SNARK proofs
* Verify proofs at the server
* Reject invalid updates
* Aggregate verified updates using FedAvg
* Evaluate global model accuracy

---

## Experimental Setup

| Parameter         | Value    |
| ----------------- | -------- |
| Dataset           | MNIST    |
| Clients           | 10       |
| Rounds            | 150      |
| Local Epochs      | 5        |
| Alpha             | 0.5      |
| Epsilon           | 1.0      |
| Delta             | 1e-5     |
| Clip Norm         | 0.5      |
| Proof System      | zk-SNARK |
| Circuit Framework | Circom   |
| Verification Tool | snarkjs  |
| Optimizer         | Adam     |
| Learning Rate     | 0.001    |

---

## Research Context

This repository represents Approach 3 of a privacy-preserving Federated Learning study.

### Approach 1

FedAvg + Differential Privacy

### Approach 2

FedAvg + Differential Privacy + Dilithium Authentication

### Approach 3 (This Repository)

FedAvg + Differential Privacy + Zero-Knowledge Proof Validation

The objective of Approach 3 is to provide privacy-preserving validation of client updates by ensuring that only updates satisfying predefined constraints are accepted for aggregation, without revealing the underlying private model parameters or training data.

---

## Author

Adithi

B.Tech Computer Science and Engineering

NMAM Institute of Technology (NMAMIT)

Research Internship – NITK Surathkal
