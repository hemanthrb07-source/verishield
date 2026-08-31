const API_BASE = '';

export interface VerificationResult {
  verification_id: string;
  status: string;
  file_type: string;
  trust_score: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  confidence: number;
  reasons: string[];
  detailed_results: any;
  processing_time_ms: number;
  blockchain_tx_hash?: string;
  created_at: string;
}

export interface BatchResult {
  batch_id: string;
  total_items: number;
  results: VerificationResult[];
  summary: {
    high_risk_count: number;
    avg_trust_score: number;
  };
}

export interface GraphData {
  nodes: { id: string; type: string; label: string; risk_score: number }[];
  edges: { source: string; target: string; relationship: string; weight: number }[];
  suspicious_clusters: any[];
  stats: { total_nodes: number; total_edges: number };
}

export interface SystemStats {
  total_verifications: number;
  completed: number;
  high_risk_detected: number;
  avg_trust_score: number;
  graph_nodes: number;
  blockchain_blocks: number;
}

class ApiService {
  private async request(path: string, options: RequestInit = {}): Promise<any> {
    const res = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: {
        ...options.headers,
      },
    });
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(error.detail || `Request failed: ${res.status}`);
    }
    return res.json();
  }

  async getStats(): Promise<SystemStats> {
    return this.request('/api/stats');
  }

  async getHealth(): Promise<any> {
    return this.request('/health');
  }

  async verifyDocument(file: File, referenceFile?: File, userId?: string): Promise<VerificationResult> {
    const formData = new FormData();
    formData.append('file', file);
    if (referenceFile) formData.append('reference_file', referenceFile);
    if (userId) formData.append('user_id', userId);
    return this.request('/verify/document', { method: 'POST', body: formData });
  }

  async verifyDeepfake(file: File, userId?: string): Promise<VerificationResult> {
    const formData = new FormData();
    formData.append('file', file);
    if (userId) formData.append('user_id', userId);
    return this.request('/verify/deepfake', { method: 'POST', body: formData });
  }

  async verifyFace(
    file: File,
    referenceFile?: File,
    userId?: string
  ): Promise<VerificationResult> {
    const formData = new FormData();
    formData.append('file', file);
    if (referenceFile) formData.append('reference_file', referenceFile);
    if (userId) formData.append('user_id', userId);
    return this.request('/verify/face', { method: 'POST', body: formData });
  }

  async verifyFull(
    file: File,
    referenceFile?: File,
    userId?: string
  ): Promise<VerificationResult> {
    const formData = new FormData();
    formData.append('file', file);
    if (referenceFile) formData.append('reference_file', referenceFile);
    if (userId) formData.append('user_id', userId);
    return this.request('/verify/full', { method: 'POST', body: formData });
  }

  async batchVerify(files: File[], userId?: string): Promise<BatchResult> {
    const formData = new FormData();
    files.forEach((f) => formData.append('files', f));
    if (userId) formData.append('user_id', userId);
    return this.request('/verify/batch', { method: 'POST', body: formData });
  }

  async getGraphData(): Promise<GraphData> {
    return this.request('/graph/data');
  }

  async getVerifications(params?: {
    limit?: number;
    offset?: number;
    status?: string;
    risk_level?: string;
  }): Promise<{ total: number; items: any[] }> {
    const query = new URLSearchParams();
    if (params?.limit) query.set('limit', String(params.limit));
    if (params?.offset) query.set('offset', String(params.offset));
    if (params?.status) query.set('status', params.status);
    if (params?.risk_level) query.set('risk_level', params.risk_level);
    return this.request(`/verifications?${query.toString()}`);
  }

  async getVerification(id: string): Promise<any> {
    return this.request(`/verifications/${id}`);
  }

  async verifyBlockchain(): Promise<any> {
    return this.request('/blockchain/verify');
  }

  async getBlockchainRecord(verificationId: string): Promise<any> {
    return this.request(`/blockchain/record/${verificationId}`);
  }

  async verifyLiveness(file: File, userId?: string): Promise<VerificationResult> {
    const formData = new FormData();
    formData.append('file', file);
    if (userId) formData.append('user_id', userId);
    return this.request('/verify/liveness', { method: 'POST', body: formData });
  }
}

export const api = new ApiService();
