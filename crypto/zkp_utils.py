import hashlib
import secrets
import time
import os
import json
import subprocess
import tempfile


P = 208351617316091241234326746312124448251235562226470491514186331217050270460481
G = 2


def _get_zkp_dir():
    """
    Works on both Windows and Colab/Linux.
    crypto/zkp_utils.py -> project root -> zkp/
    """
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_file_dir)
    return os.path.join(project_root, "zkp")


def _npx_cmd():
    """
    Windows uses npx.cmd, Linux/Colab uses npx.
    """
    return "npx.cmd" if os.name == "nt" else "npx"


def _hash_to_int(*values) -> int:
    h = hashlib.sha256()
    for v in values:
        if isinstance(v, int):
            h.update(str(v).encode())
        elif isinstance(v, bytes):
            h.update(v)
        else:
            h.update(str(v).encode())
    return int.from_bytes(h.digest(), "big") % (P - 1)


def keygen():
    start = time.time()
    secret_key = secrets.randbelow(P - 2) + 1
    public_key = pow(G, secret_key, P)
    keygen_ms = (time.time() - start) * 1000
    return public_key, secret_key, keygen_ms


def generate_proof(secret_key: int, update_bytes: bytes, client_id: str):
    start = time.time()

    update_hash = hashlib.sha256(update_bytes).hexdigest()

    r = secrets.randbelow(P - 2) + 1
    commitment = pow(G, r, P)

    challenge = _hash_to_int(client_id, update_hash, commitment)

    response = (r + challenge * secret_key) % (P - 1)

    proof_ms = (time.time() - start) * 1000

    return {
        "update_hash": update_hash,
        "commitment": commitment,
        "challenge": challenge,
        "response": response,
        "proof_ms": proof_ms,
    }


def verify_proof(public_key: int, update_bytes: bytes, client_id: str, proof: dict):
    start = time.time()

    expected_hash = hashlib.sha256(update_bytes).hexdigest()

    if proof["update_hash"] != expected_hash:
        return False, 0.0

    expected_challenge = _hash_to_int(
        client_id,
        proof["update_hash"],
        proof["commitment"],
    )

    if proof["challenge"] != expected_challenge:
        return False, 0.0

    left = pow(G, proof["response"], P)
    right = (proof["commitment"] * pow(public_key, proof["challenge"], P)) % P

    verify_ms = (time.time() - start) * 1000

    return left == right, verify_ms


def generate_snark(values: list[int], threshold: int):
    start = time.time()

    input_data = {
        "values": values,
        "threshold": threshold,
    }

    zkp_dir = _get_zkp_dir()
    build_dir = os.path.join(zkp_dir, "build")

    if not os.path.isdir(zkp_dir):
        raise FileNotFoundError(f"ZKP directory not found: {zkp_dir}")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.json")
        proof_path = os.path.join(tmpdir, "proof.json")
        public_path = os.path.join(tmpdir, "public.json")

        with open(input_path, "w") as f:
            json.dump(input_data, f)

        cmd = [
            _npx_cmd(),
            "snarkjs",
            "groth16",
            "fullprove",
            input_path,
            os.path.join(build_dir, "update_norm_js", "update_norm.wasm"),
            os.path.join(build_dir, "update_norm_0001.zkey"),
            proof_path,
            public_path,
        ]

        subprocess.run(
            cmd,
            check=True,
            cwd=zkp_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        with open(proof_path, "r") as f:
            proof = json.load(f)

        with open(public_path, "r") as f:
            public_signals = json.load(f)

    snark_ms = (time.time() - start) * 1000
    return proof, public_signals, snark_ms


def verify_snark(proof: dict, public_signals: list):
    start = time.time()

    zkp_dir = _get_zkp_dir()
    build_dir = os.path.join(zkp_dir, "build")

    if not os.path.isdir(zkp_dir):
        raise FileNotFoundError(f"ZKP directory not found: {zkp_dir}")

    with tempfile.TemporaryDirectory() as tmpdir:
        proof_path = os.path.join(tmpdir, "proof.json")
        public_path = os.path.join(tmpdir, "public.json")

        with open(proof_path, "w") as f:
            json.dump(proof, f)

        with open(public_path, "w") as f:
            json.dump(public_signals, f)

        cmd = [
            _npx_cmd(),
            "snarkjs",
            "groth16",
            "verify",
            os.path.join(build_dir, "verification_key.json"),
            public_path,
            proof_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            cwd=zkp_dir,
        )

        is_valid = "OK" in result.stdout.decode()

    verify_ms = (time.time() - start) * 1000
    return is_valid, verify_ms
