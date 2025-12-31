// GodsEye API - Frontend TypeScript calls
// Replace API_BASE_URL with your deployed Railway URL

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

interface ProcessResponse {
  status: string;
  product_id: string;
  data_source: string;
  run_id?: string;
  analysis_id?: string;
  message?: string;
  error?: string;
}

interface StatusResponse {
  product_id: string;
  data_source: string;
  status: string;
  analysis_id?: string;
  dna_blueprint?: any;
  created_at?: string;
  updated_at?: string;
}

// Google analysis
export async function analyzeGoogle(productId: string): Promise<ProcessResponse> {
  const response = await fetch(`${API_BASE_URL}/process`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      product_id: productId,
      source: 'google'
    })
  });

  if (!response.ok) {
    throw new Error(`Google analysis failed: ${response.statusText}`);
  }

  return response.json();
}

// Perplexity analysis
export async function analyzePerplexity(productId: string): Promise<ProcessResponse> {
  const response = await fetch(`${API_BASE_URL}/process`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      product_id: productId,
      source: 'perplexity'
    })
  });

  if (!response.ok) {
    throw new Error(`Perplexity analysis failed: ${response.statusText}`);
  }

  return response.json();
}

// Check status/results
export async function getStatus(productId: string): Promise<StatusResponse> {
  const response = await fetch(`${API_BASE_URL}/status`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      product_id: productId
    })
  });

  if (!response.ok) {
    throw new Error(`Status check failed: ${response.statusText}`);
  }

  return response.json();
}

// Health check
export async function healthCheck(): Promise<{ status: string; timestamp: string }> {
  const response = await fetch(`${API_BASE_URL}/health`);
  return response.json();
}
