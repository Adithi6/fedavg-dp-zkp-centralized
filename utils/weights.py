import logging
import hashlib
import numpy as np
import torch
import torch.nn as nn


def weights_to_bytes(model: nn.Module, dtype_name: str) -> bytes:
    dtype_name = dtype_name.lower()

    if dtype_name == "float32":
        target_dtype = np.float32
    elif dtype_name == "float64":
        target_dtype = np.float64
    else:
        raise ValueError(f"Unsupported weight dtype: {dtype_name}")

    arrays = [p.data.cpu().numpy().flatten() for p in model.parameters()]
    flat = np.concatenate(arrays).astype(target_dtype)
    data = flat.tobytes()

    logging.debug(
        f"Converted model weights to bytes | size={len(data)} bytes | dtype={dtype_name}"
    )
    return data


def bytes_to_weight_arrays(
    data: bytes,
    template_model: nn.Module,
    dtype_name: str,
) -> list[np.ndarray]:
    dtype_name = dtype_name.lower()

    if dtype_name == "float32":
        source_dtype = np.float32
    elif dtype_name == "float64":
        source_dtype = np.float64
    else:
        raise ValueError(f"Unsupported weight dtype: {dtype_name}")

    flat = np.frombuffer(data, dtype=source_dtype).copy()
    shapes = [tuple(p.shape) for p in template_model.parameters()]

    arrays = []
    idx = 0
    for shape in shapes:
        n = int(np.prod(shape))
        arrays.append(flat[idx: idx + n].reshape(shape))
        idx += n

    logging.debug(
        f"Reconstructed weight arrays | total_elements={len(flat)} | dtype={dtype_name}"
    )
    return arrays


def apply_weight_arrays(model: nn.Module, arrays: list[np.ndarray]):
    for param, arr in zip(model.parameters(), arrays):
        param.data = torch.from_numpy(arr).to(param.device)


def model_to_weight_arrays(model: nn.Module) -> list[np.ndarray]:
    arrays = [p.data.cpu().numpy().copy() for p in model.parameters()]
    logging.debug("Converted model to weight arrays")
    return arrays


def hash_weights(model: nn.Module, dtype_name: str, hash_algorithm: str) -> bytes:
    weight_bytes = weights_to_bytes(model, dtype_name=dtype_name)

    hash_algorithm = hash_algorithm.lower()

    if hash_algorithm == "sha256":
        return hashlib.sha256(weight_bytes).digest()

    if hash_algorithm == "sha512":
        return hashlib.sha512(weight_bytes).digest()

    raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")