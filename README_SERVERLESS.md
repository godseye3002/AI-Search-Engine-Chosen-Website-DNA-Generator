# GodsEye DNA Pipeline - Serverless Ready

FastAPI-based serverless deployment for the GodsEye DNA analysis pipeline.

## Quick Start

### 1. Environment Setup

Copy `.env.example` to `.env` and configure:

```bash
# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# AI Model Configuration
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key

# Server Configuration
PORT=8000
LOG_LEVEL=INFO
```

### 2. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
python serverless_api.py
```

API will be available at `http://localhost:8000`

### 3. Docker Deployment

```bash
# Build image
docker build -t godseye-dna-pipeline .

# Run container
docker run -p 8000:8000 --env-file .env godseye-dna-pipeline
```

### 4. Serverless Deployment

#### Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

#### Railway

```bash
# Deploy to Railway
railway login
railway new
railway up
```

## API Endpoints

### Process Single Product
```http
POST /process
Content-Type: application/json

{
  "product_id": "02f92e70-7b53-45b6-bdef-7ef36d8fc578",
  "source": "google"
}
```

### Process Batch
```http
POST /process-batch
Content-Type: application/json

{
  "source": "google",
  "limit": 10
}
```

### Check Status
```http
POST /status
Content-Type: application/json

{
  "product_id": "02f92e70-7b53-45b6-bdef-7ef36d8fc578"
}
```

### Health Check
```http
GET /health
```

### Statistics
```http
GET /stats
```

## Response Examples

### Process Response
```json
{
  "status": "completed",
  "product_id": "02f92e70-7b53-45b6-bdef-7ef36d8fc578",
  "data_source": "google",
  "run_id": "dna_google_02f92e70-7b53-45b6-bdef-7ef36d8fc578_1766816712",
  "analysis_id": "c8c097a9-8496-4cf1-a010-82614a2af715",
  "final_output_path": "outputs/stage_3_results/run_dna_google_02f92e70-7b53-45b6-bdef-7ef36d8fc578_1766816712/final_aggregation.json"
}
```

### Status Response
```json
{
  "product_id": "02f92e70-7b53-45b6-bdef-7ef36d8fc578",
  "data_source": "google",
  "status": "completed",
  "analysis_id": "c8c097a9-8496-4cf1-a010-82614a2af715",
  "dna_blueprint": {
    "query": "...",
    "master_blueprint": {
      "top_performers": [...],
      "content_gaps": [...],
      "recommendations": [...]
    }
  },
  "created_at": "2025-12-27T11:55:18.366Z",
  "updated_at": "2025-12-27T11:55:18.366Z"
}
```

## Architecture

```
Frontend (Web/Mobile) → FastAPI Server → Database Orchestrator → DNA Pipeline → Supabase
```

- **FastAPI Server**: Lightweight, async, auto-generated docs
- **Docker**: Containerized for consistent deployment
- **Environment Variables**: All secrets externalized
- **Health Checks**: Built-in monitoring endpoints

## Monitoring

- Health endpoint: `GET /health`
- Logs: Structured JSON logging
- Error handling: Comprehensive HTTP status codes
- Timeouts: Configurable per stage

## Production Considerations

1. **Rate Limiting**: Add middleware for API rate limiting
2. **Authentication**: Add API key middleware if needed
3. **Queue System**: For high volume, add Redis/RabbitMQ
4. **Caching**: Cache frequent requests to Supabase
5. **Monitoring**: Add Prometheus metrics

## Frontend Integration

### JavaScript/TypeScript Example

```typescript
interface ProcessRequest {
  product_id: string;
  source: 'google' | 'perplexity';
}

interface ProcessResponse {
  status: string;
  product_id: string;
  data_source: string;
  run_id?: string;
  analysis_id?: string;
  final_output_path?: string;
  error?: string;
}

class GodsEyeAPI {
  private baseURL: string;

  constructor(baseURL: string = 'https://your-app.vercel.app') {
    this.baseURL = baseURL;
  }

  async processProduct(request: ProcessRequest): Promise<ProcessResponse> {
    const response = await fetch(`${this.baseURL}/process`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request)
    });
    return response.json();
  }

  async checkStatus(productId: string): Promise<any> {
    const response = await fetch(`${this.baseURL}/status`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ product_id: productId })
    });
    return response.json();
  }
}

// Usage
const api = new GodsEyeAPI();
api.processProduct({
  product_id: '02f92e70-7b53-45b6-bdef-7ef36d8fc578',
  source: 'google'
}).then(result => {
  console.log('Processing result:', result);
});
```

## Deployment Checklist

- [ ] Environment variables configured
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Docker image built
- [ ] Health endpoint responding
- [ ] Test API calls successful
- [ ] Monitoring configured
- [ ] Frontend integration tested
