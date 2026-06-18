import logging
import random

from crypto import zkp_utils


class GossipProtocol:
    def __init__(self, fanout: int, max_hops: int):
        self.fanout = fanout
        self.max_hops = max_hops

        # track (origin, forwarder) to avoid loops
        self._seen_forward: set[tuple[str, str]] = set()

        # store metrics
        self.gossip_timings: list[dict] = []

    def reset_round(self):
        self._seen_forward.clear()
        self.gossip_timings.clear()
        logging.info("Gossip round state reset")

    # -------------------------------
    # ZKP VERIFICATION
    # -------------------------------
    def _verify_zkp(self, message: dict) -> tuple[bool, float]:
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
                return False, sig_ms
                
            # 2. Verify zk-SNARK
            is_valid_snark, snark_ms = zkp_utils.verify_snark(
                proof=message["snark_proof"],
                public_signals=message["snark_public"]
            )
            
            if not is_valid_snark:
                return False, sig_ms + snark_ms
                
            total_ms = float(sig_ms + snark_ms)
            
        except Exception as e:
            logging.error(f"ZKP verification error: {e}")
            return False, 0.0

        return True, total_ms

    # -------------------------------
    # GOSSIP SPREAD
    # -------------------------------
    def spread(
        self,
        origin_node,
        all_nodes,
        message: dict,
        hop: int = 0,
    ):
        origin_client_id = message["client_id"]
        state_id = (origin_client_id, origin_node.client_id)

        # avoid duplicate forwarding
        if state_id in self._seen_forward:
            logging.debug(
                f"Skipping duplicate forward from {origin_node.client_id} "
                f"for {origin_client_id}"
            )
            return

        if hop >= self.max_hops:
            logging.debug(f"Max hops reached for {origin_client_id}")
            return

        self._seen_forward.add(state_id)

        peers = [n for n in all_nodes if n.client_id != origin_node.client_id]
        if not peers:
            return

        targets = random.sample(peers, min(self.fanout, len(peers)))

        for target in targets:
            # -------- ZKP VERIFICATION --------
            is_valid, verify_ms = self._verify_zkp(message)

            self.gossip_timings.append({
                "from": origin_node.client_id,
                "to": target.client_id,
                "origin": origin_client_id,
                "hop": hop + 1,
                "verify_ms": round(verify_ms, 3),
                "accepted": is_valid,
            })

            logging.info(
                f"[gossip] {origin_node.client_id} -> {target.client_id} "
                f"hop={hop + 1} zkp_verify={verify_ms:.3f} ms "
                f"[{'OK' if is_valid else 'REJECTED'}]"
            )

            # -------- REJECT INVALID --------
            if not is_valid:
                logging.warning(
                    f"[gossip] rejected invalid ZKP update from {origin_client_id} "
                    f"at {target.client_id}"
                )
                continue

            # -------- ACCEPT & FORWARD --------
            target.receive_gossip(message)

            self.spread(
                target,
                all_nodes,
                message,
                hop=hop + 1,
            )

    # -------------------------------
    # RUN ROUND
    # -------------------------------
    def run_round(self, nodes):
        self.reset_round()

        for node in nodes:
            if node.own_submission is None:
                raise RuntimeError(
                    f"{node.client_id} has no submission — call prepare_update() first"
                )

            logging.info(f"[gossip] spreading ZKP-verified update from {node.client_id}")

            self.spread(
                origin_node=node,
                all_nodes=nodes,
                message=node.own_submission,
                hop=0,
            )

    # -------------------------------
    # SUMMARY
    # -------------------------------
    def print_gossip_summary(self):
        if not self.gossip_timings:
            logging.info("No gossip records available")
            return

        logging.info("-" * 80)
        logging.info(f"Gossip log (fanout={self.fanout}, max_hops={self.max_hops})")
        logging.info("-" * 80)
        logging.info(
            f"{'Origin':<12} {'From':<12} {'To':<12} "
            f"{'Hop':<5} {'Verify(ms)':<12} Accepted"
        )
        logging.info("-" * 80)

        for t in self.gossip_timings:
            logging.info(
                f"{t['origin']:<12} {t['from']:<12} {t['to']:<12} "
                f"{t['hop']:<5} {t['verify_ms']:<12} {t['accepted']}"
            )

        accepted = [t for t in self.gossip_timings if t["accepted"]]

        logging.info(f"Total gossip hops: {len(self.gossip_timings)}")

        if accepted:
            avg_verify = sum(t["verify_ms"] for t in accepted) / len(accepted)
            logging.info(f"Average ZKP verify time: {avg_verify:.3f} ms")