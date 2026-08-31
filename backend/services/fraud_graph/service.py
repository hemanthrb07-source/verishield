"""
Fraud Graph Service.
Stores and queries fraud relationship networks.
Detects repeated identity usage and suspicious clusters.
"""
from typing import Optional
from collections import defaultdict
import hashlib
import json


class FraudGraphService:
    """
    In-memory graph for fraud relationship tracking.
    In production, backed by Neo4j; here we use a lightweight in-memory graph
    that can persist to SQLite/PostgreSQL.
    """

    def __init__(self):
        self.nodes = {}  # id -> {type, value, metadata, risk_score}
        self.edges = []  # [{source, target, relationship, weight}]
        self.adjacency = defaultdict(list)  # node_id -> [edge_indices]

    async def add_verification_to_graph(
        self,
        verification_id: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        device_info: Optional[dict] = None,
        face_embedding: Optional[list] = None,
    ) -> dict:
        """
        Add a verification event to the fraud graph.
        Returns graph risk assessment.
        """
        added_nodes = []
        added_edges = []

        # Add user node
        if user_id:
            node_id = self._make_node_id('user', user_id)
            self._upsert_node(node_id, 'user', user_id, {'verification_id': verification_id})
            added_nodes.append(node_id)

        # Add IP node
        if ip_address:
            node_id = self._make_node_id('ip', ip_address)
            self._upsert_node(node_id, 'ip', ip_address, {'verification_id': verification_id})
            added_nodes.append(node_id)

        # Add device node
        if device_info:
            device_fingerprint = self._compute_device_fingerprint(device_info)
            node_id = self._make_node_id('device', device_fingerprint)
            self._upsert_node(node_id, 'device', device_fingerprint, {
                'verification_id': verification_id,
                'device_info': device_info,
            })
            added_nodes.append(node_id)

        # Add face embedding node (hash it for privacy)
        if face_embedding:
            emb_hash = hashlib.sha256(json.dumps(face_embedding[:10]).encode()).hexdigest()[:16]
            node_id = self._make_node_id('face', emb_hash)
            self._upsert_node(node_id, 'face', emb_hash, {'verification_id': verification_id})
            added_nodes.append(node_id)

        # Create edges between related nodes
        for i in range(len(added_nodes)):
            for j in range(i + 1, len(added_nodes)):
                edge = {
                    'source': added_nodes[i],
                    'target': added_nodes[j],
                    'relationship': 'co_occurred',
                    'weight': 1.0,
                }
                self.edges.append(edge)
                edge_idx = len(self.edges) - 1
                self.adjacency[added_nodes[i]].append(edge_idx)
                self.adjacency[added_nodes[j]].append(edge_idx)
                added_edges.append(edge)

        # Compute risk score based on graph analysis
        risk = self._compute_graph_risk(added_nodes)

        return {
            'nodes_added': len(added_nodes),
            'edges_added': len(added_edges),
            'risk_score': risk['risk_score'],
            'suspicious_patterns': risk['patterns'],
            'cluster_info': risk['clusters'],
        }

    async def lookup_risk(self, user_id: Optional[str] = None,
                          ip_address: Optional[str] = None) -> float:
        """Look up risk score for known entities."""
        risk_factors = []

        if user_id:
            node_id = self._make_node_id('user', user_id)
            if node_id in self.nodes:
                node = self.nodes[node_id]
                risk_factors.append(node.get('risk_score', 0))

                # Count connections
                connections = len(self.adjacency.get(node_id, []))
                if connections > 10:
                    risk_factors.append(min(connections / 50, 1.0))

        if ip_address:
            node_id = self._make_node_id('ip', ip_address)
            if node_id in self.nodes:
                # Find all users from this IP
                users_from_ip = set()
                for edge_idx in self.adjacency.get(node_id, []):
                    edge = self.edges[edge_idx]
                    other = edge['target'] if edge['source'] == node_id else edge['source']
                    other_node = self.nodes.get(other, {})
                    if other_node.get('type') == 'user':
                        users_from_ip.add(other_node.get('value'))

                if len(users_from_ip) > 3:
                    risk_factors.append(min(len(users_from_ip) / 10, 1.0))

        return max(risk_factors) if risk_factors else 0.0

    async def get_graph_data(self) -> dict:
        """Get graph data for visualization."""
        nodes = []
        for node_id, node_data in self.nodes.items():
            nodes.append({
                'id': node_id,
                'type': node_data['type'],
                'label': node_data['value'][:20],
                'risk_score': node_data.get('risk_score', 0),
            })

        edges = []
        for edge in self.edges:
            edges.append({
                'source': edge['source'],
                'target': edge['target'],
                'relationship': edge['relationship'],
                'weight': edge['weight'],
            })

        suspicious = self._find_suspicious_clusters()

        return {
            'nodes': nodes,
            'edges': edges,
            'suspicious_clusters': suspicious,
            'stats': {
                'total_nodes': len(self.nodes),
                'total_edges': len(self.edges),
            },
        }

    async def find_suspicious_clusters(self) -> list[dict]:
        """Find clusters of suspicious activity."""
        return self._find_suspicious_clusters()

    # ── Internal Methods ──

    def _make_node_id(self, node_type: str, value: str) -> str:
        return f"{node_type}:{value}"

    def _upsert_node(self, node_id: str, node_type: str, value: str, metadata: dict):
        if node_id in self.nodes:
            # Update metadata and increment interaction count
            existing = self.nodes[node_id]
            count = existing.get('metadata', {}).get('interaction_count', 1)
            existing['metadata']['interaction_count'] = count + 1
            existing['metadata']['last_seen'] = metadata.get('verification_id')
        else:
            self.nodes[node_id] = {
                'type': node_type,
                'value': value,
                'metadata': {**metadata, 'interaction_count': 1},
                'risk_score': 0.0,
            }

    def _compute_device_fingerprint(self, device_info: dict) -> str:
        """Compute a fingerprint from device info."""
        parts = [
            str(device_info.get('user_agent', '')),
            str(device_info.get('screen_resolution', '')),
            str(device_info.get('timezone', '')),
            str(device_info.get('language', '')),
        ]
        return hashlib.md5('|'.join(parts).encode()).hexdigest()[:16]

    def _compute_graph_risk(self, node_ids: list[str]) -> dict:
        """Compute risk score based on graph topology."""
        risk_score = 0.0
        patterns = []
        clusters = []

        for node_id in node_ids:
            node = self.nodes.get(node_id, {})
            connections = len(self.adjacency.get(node_id, []))
            interaction_count = node.get('metadata', {}).get('interaction_count', 1)

            # Risk increases with repeated usage
            if interaction_count > 1:
                usage_risk = min((interaction_count - 1) * 0.15, 0.6)
                risk_score = max(risk_score, usage_risk)
                patterns.append({
                    'type': 'repeated_usage',
                    'entity': node_id,
                    'count': interaction_count,
                    'risk_contribution': usage_risk,
                })

            # Risk increases with high connectivity
            if connections > 5:
                conn_risk = min((connections - 5) * 0.1, 0.5)
                risk_score = max(risk_score, conn_risk)
                patterns.append({
                    'type': 'high_connectivity',
                    'entity': node_id,
                    'connections': connections,
                    'risk_contribution': conn_risk,
                })

        # Check for IP sharing across multiple users
        ip_nodes = [n for n in node_ids if n.startswith('ip:')]
        for ip_node in ip_nodes:
            users = set()
            for edge_idx in self.adjacency.get(ip_node, []):
                edge = self.edges[edge_idx]
                other = edge['target'] if edge['source'] == ip_node else edge['source']
                if other.startswith('user:'):
                    users.add(other)
            if len(users) > 2:
                ip_risk = min(len(users) * 0.2, 0.8)
                risk_score = max(risk_score, ip_risk)
                clusters.append({
                    'type': 'shared_ip',
                    'ip': ip_node,
                    'users': list(users),
                    'risk_score': ip_risk,
                })
                patterns.append({
                    'type': 'multiple_users_same_ip',
                    'ip': ip_node.split(':')[1],
                    'user_count': len(users),
                    'risk_contribution': ip_risk,
                })

        return {
            'risk_score': min(risk_score, 1.0),
            'patterns': patterns,
            'clusters': clusters,
        }

    def _find_suspicious_clusters(self) -> list[dict]:
        """Find all suspicious clusters in the graph."""
        clusters = []

        # Group nodes by type
        by_type = defaultdict(list)
        for node_id, node in self.nodes.items():
            by_type[node['type']].append((node_id, node))

        # Find IPs with multiple users
        for ip_id, ip_node in by_type.get('ip', []):
            connected_users = set()
            for edge_idx in self.adjacency.get(ip_id, []):
                edge = self.edges[edge_idx]
                other = edge['target'] if edge['source'] == ip_id else edge['source']
                if other.startswith('user:'):
                    connected_users.add(other.split(':')[1])

            if len(connected_users) > 1:
                clusters.append({
                    'type': 'shared_ip',
                    'description': f'IP shared by {len(connected_users)} users',
                    'entities': list(connected_users),
                    'risk_score': min(len(connected_users) * 0.2, 0.8),
                })

        # Find users with multiple devices
        for user_id, user_node in by_type.get('user', []):
            connected_devices = set()
            for edge_idx in self.adjacency.get(user_id, []):
                edge = self.edges[edge_idx]
                other = edge['target'] if edge['source'] == user_id else edge['source']
                if other.startswith('device:'):
                    connected_devices.add(other.split(':')[1])

            if len(connected_devices) > 2:
                clusters.append({
                    'type': 'multiple_devices',
                    'description': f'User with {len(connected_devices)} devices',
                    'user': user_id.split(':')[1],
                    'risk_score': min(len(connected_devices) * 0.15, 0.6),
                })

        return clusters
