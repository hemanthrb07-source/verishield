"""
Persistent verification store using SQLite.
Replaces the in-memory dict so history survives server restarts.
"""
import sqlite3
import json
import os
import threading
from typing import Optional, Dict, Any, List


class VerificationStore:
    """Thread-safe SQLite store for verification records."""

    def __init__(self, db_path: str = "verishield.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS verifications (
                id TEXT PRIMARY KEY,
                status TEXT,
                file_type TEXT,
                file_name TEXT,
                file_hash TEXT,
                trust_score REAL,
                risk_level TEXT,
                confidence REAL,
                reasons TEXT,
                detailed_results TEXT,
                processing_time_ms INTEGER,
                blockchain_tx_hash TEXT,
                user_id TEXT,
                created_at TEXT,
                raw_data TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON verifications(status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_risk_level ON verifications(risk_level)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON verifications(created_at)
        """)
        conn.commit()

    def put(self, verification_id: str, data: Dict[str, Any]):
        """Insert or update a verification record."""
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO verifications
            (id, status, file_type, file_name, file_hash, trust_score,
             risk_level, confidence, reasons, detailed_results,
             processing_time_ms, blockchain_tx_hash, user_id, created_at, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            verification_id,
            data.get('status'),
            data.get('file_type'),
            data.get('file_name'),
            data.get('file_hash'),
            data.get('trust_score'),
            data.get('risk_level'),
            data.get('confidence'),
            json.dumps(data.get('reasons', []), default=str),
            json.dumps(data.get('detailed_results', {}), default=str),
            data.get('processing_time_ms'),
            data.get('blockchain_tx_hash'),
            data.get('user_id'),
            data.get('created_at'),
            json.dumps(data, default=str),
        ))
        conn.commit()

    def update(self, verification_id: str, updates: Dict[str, Any]):
        """Update specific fields of a verification record."""
        conn = self._get_conn()

        # Load existing raw_data and merge
        row = conn.execute(
            "SELECT raw_data FROM verifications WHERE id = ?",
            (verification_id,)
        ).fetchone()

        if row:
            existing = json.loads(row['raw_data'])
            existing.update(updates)
            merged = existing
        else:
            merged = updates

        # Update individual columns
        set_clauses = []
        values = []
        for key in ['status', 'trust_score', 'risk_level', 'confidence',
                     'processing_time_ms', 'blockchain_tx_hash']:
            if key in updates:
                set_clauses.append(f"{key} = ?")
                values.append(updates[key])
        if 'reasons' in updates:
            set_clauses.append("reasons = ?")
            values.append(json.dumps(updates['reasons'], default=str))
        if 'detailed_results' in updates:
            set_clauses.append("detailed_results = ?")
            values.append(json.dumps(updates['detailed_results'], default=str))

        if set_clauses:
            values.append(verification_id)
            conn.execute(
                f"UPDATE verifications SET {', '.join(set_clauses)} WHERE id = ?",
                values
            )

        # Always update raw_data
        conn.execute(
            "UPDATE verifications SET raw_data = ? WHERE id = ?",
            (json.dumps(merged, default=str), verification_id)
        )
        conn.commit()

    def get(self, verification_id: str) -> Optional[Dict[str, Any]]:
        """Get a single verification record."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT raw_data FROM verifications WHERE id = ?",
            (verification_id,)
        ).fetchone()
        if row:
            return json.loads(row['raw_data'])
        return None

    def list(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        risk_level: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List verification records with filtering and pagination."""
        conn = self._get_conn()

        where_clauses = []
        params = []
        if status:
            where_clauses.append("status = ?")
            params.append(status)
        if risk_level:
            where_clauses.append("risk_level = ?")
            params.append(risk_level)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Get total count
        count_row = conn.execute(
            f"SELECT COUNT(*) as cnt FROM verifications {where_sql}",
            params
        ).fetchone()
        total = count_row['cnt']

        # Get items
        rows = conn.execute(
            f"SELECT raw_data FROM verifications {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params + [limit, offset]
        ).fetchall()

        items = [json.loads(row['raw_data']) for row in rows]

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": items,
        }

    def count(self) -> int:
        """Count total records."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM verifications").fetchone()
        return row['cnt']

    def stats(self) -> Dict[str, Any]:
        """Get aggregated stats."""
        conn = self._get_conn()
        row = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN risk_level IN ('HIGH', 'CRITICAL') THEN 1 ELSE 0 END) as high_risk,
                AVG(trust_score) as avg_score
            FROM verifications
        """).fetchone()
        return {
            "total_verifications": row['total'],
            "completed": row['completed'] or 0,
            "high_risk_detected": row['high_risk'] or 0,
            "avg_trust_score": round(row['avg_score'] or 0, 1),
        }

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def __setitem__(self, key: str, value: Dict[str, Any]):
        self.put(key, value)

    def __getitem__(self, key: str) -> Dict[str, Any]:
        val = self.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def values(self):
        """Return all values (for compatibility)."""
        conn = self._get_conn()
        rows = conn.execute("SELECT raw_data FROM verifications ORDER BY created_at DESC").fetchall()
        return [json.loads(row['raw_data']) for row in rows]

    def update_existing(self, verification_id: str, updates: Dict[str, Any]):
        """Update an existing record (dict-style .update())."""
        self.update(verification_id, updates)


# Singleton instance
DB_PATH = os.environ.get("VERISHIELD_DB", "verishield.db")
verification_store = VerificationStore(db_path=DB_PATH)
