"""
Blockchain Hash Logger.
Provides tamper-proof audit trail for verification results.
Uses a local blockchain-like chain for demo purposes.
"""
import hashlib
import json
import time
from typing import Optional


class BlockchainLogger:
    """
    Simple blockchain implementation for audit logging.
    In production, this would interface with Ethereum/Polygon.
    """

    def __init__(self):
        self.chain = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        """Create the first block in the chain."""
        genesis = {
            'index': 0,
            'timestamp': time.time(),
            'data': {'type': 'genesis', 'message': 'VeriShield Blockchain Audit Trail Initialized'},
            'previous_hash': '0' * 64,
            'nonce': 0,
        }
        genesis['hash'] = self._compute_hash(genesis)
        self.chain.append(genesis)

    async def log_verification(
        self,
        verification_id: str,
        content_hash: str,
        result_hash: str,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Log a verification result to the blockchain."""
        block_data = {
            'type': 'verification',
            'verification_id': verification_id,
            'content_hash': content_hash,
            'result_hash': result_hash,
            'timestamp': time.time(),
            'metadata': metadata or {},
        }

        block = {
            'index': len(self.chain),
            'timestamp': time.time(),
            'data': block_data,
            'previous_hash': self.chain[-1]['hash'],
            'nonce': 0,
        }

        # Simple proof-of-work (low difficulty for demo)
        block = self._mine_block(block, difficulty=2)
        self.chain.append(block)

        return {
            'block_number': block['index'],
            'tx_hash': block['hash'],
            'timestamp': block['timestamp'],
            'chain_length': len(self.chain),
        }

    async def verify_integrity(self) -> dict:
        """Verify the integrity of the entire chain."""
        issues = []
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            # Check hash
            computed_hash = self._compute_hash({k: v for k, v in current.items() if k != 'hash'})
            if current['hash'] != computed_hash:
                issues.append({
                    'block': i,
                    'type': 'hash_mismatch',
                    'expected': computed_hash,
                    'actual': current['hash'],
                })

            # Check chain linkage
            if current['previous_hash'] != previous['hash']:
                issues.append({
                    'block': i,
                    'type': 'chain_break',
                    'expected_previous': previous['hash'],
                    'actual_previous': current['previous_hash'],
                })

        return {
            'valid': len(issues) == 0,
            'chain_length': len(self.chain),
            'issues': issues,
        }

    async def get_verification_record(self, verification_id: str) -> Optional[dict]:
        """Find a specific verification record in the chain."""
        for block in self.chain:
            if block['data'].get('verification_id') == verification_id:
                return {
                    'block_number': block['index'],
                    'tx_hash': block['hash'],
                    'timestamp': block['timestamp'],
                    'data': block['data'],
                }
        return None

    def _compute_hash(self, block: dict) -> str:
        """Compute SHA-256 hash of a block."""
        block_string = json.dumps(block, sort_keys=True, default=str)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def _mine_block(self, block: dict, difficulty: int = 2) -> dict:
        """Simple proof-of-work mining."""
        prefix = '0' * difficulty
        while True:
            block['nonce'] += 1
            computed = self._compute_hash(block)
            if computed.startswith(prefix):
                block['hash'] = computed
                return block
            if block['nonce'] > 10000:  # Safety limit
                block['hash'] = self._compute_hash(block)
                return block
