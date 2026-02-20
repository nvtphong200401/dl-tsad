# RAG System for Historical Pattern Retrieval

## Position in Pipeline

This RAG system enhances **Step 3: Scoring via LLM Reasoning** in the 4-step pipeline (Phase 3).

**Pipeline Context:**
- **Step 1** (Preprocessing): Generates forecasts
- **Step 2** (Detection): Extracts statistical evidence
- **Step 3** (Scoring): **LLM reasoning ← RAG enhances this step** with historical context
- **Step 4** (Post-Processing): Parses outputs

The RAG system retrieves similar historical patterns and injects them into the LLM prompt, improving reasoning consistency and confidence calibration.

## Overview

This document specifies a **Retrieval-Augmented Generation (RAG) system** for time series anomaly detection. The system stores historical anomaly patterns and retrieves similar cases to enhance LLM reasoning with relevant examples.

## Motivation

### Why RAG for Time Series Anomaly Detection?

1. **Few-Shot Learning**: Provide LLM with relevant examples without retraining
2. **Consistency**: Similar patterns receive similar explanations
3. **Knowledge Accumulation**: System improves as more patterns are added
4. **Domain Adaptation**: Inject domain-specific knowledge via examples
5. **Confidence Calibration**: Historical outcomes inform confidence estimates

### Advantages Over Fine-Tuning

| Aspect | Fine-Tuning | RAG |
|--------|-------------|-----|
| **Update Speed** | Slow (hours) | Instant (add to DB) |
| **Interpretability** | Black-box | Transparent (see retrieved examples) |
| **Cost** | High (GPU training) | Low (vector DB only) |
| **Flexibility** | Static knowledge | Dynamic, queryable knowledge |
| **Domain Adaptation** | Requires retraining | Add domain examples |

## Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────┐
│ INPUT: Time Series Window + Statistical Evidence       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 1: Create Query Embedding                         │
│ • Encode evidence profile (10+ metrics)                │
│ • Optionally: encode time series features              │
│ • Generate query vector (dim: 384-1536)                │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Retrieve Similar Patterns                      │
│ • Query vector database (ChromaDB, FAISS, Pinecone)    │
│ • Use cosine similarity or L2 distance                 │
│ • Retrieve top-k most similar cases (k=3-5)            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Format Context for LLM                         │
│ • Extract: pattern description, evidence, label, reason│
│ • Format as few-shot examples                          │
│ • Inject into LLM prompt                               │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ OUTPUT: Enhanced LLM Prompt with Historical Context    │
└─────────────────────────────────────────────────────────┘
```

### Vector Database Schema

Each entry in the vector database represents a historical pattern:

```python
{
    # Identifiers
    "id": "case_12345",
    "timestamp": "2024-01-15T10:30:00Z",
    "dataset": "yahoo_s5",
    "category": "network_traffic",

    # Time Series Data
    "time_series_values": [1.2, 1.3, 5.8, 1.4, ...],  # The actual window
    "time_series_length": 100,
    "window_start": "2024-01-15T10:00:00Z",
    "window_end": "2024-01-15T11:40:00Z",

    # Statistical Evidence (used for embedding)
    "evidence": {
        "mae": 2.34,
        "z_score": 3.8,
        "volatility_ratio": 5.2,
        "quantile_violation": true,
        # ... all 10+ metrics
    },

    # Ground Truth
    "label": "anomaly",  # or "normal"
    "anomaly_ranges": [[42, 48]],  # Indices of anomalous points
    "confidence": 0.92,

    # Explanation
    "reasoning": "Sensor failure caused sharp spike. Z-score=3.8, volatility spike=5.2x baseline.",
    "anomaly_type": "point",  # point, contextual, collective
    "root_cause": "sensor_failure",  # Optional domain label

    # Embeddings (stored for similarity search)
    "evidence_embedding": [0.12, -0.34, 0.56, ...],  # Dim: 384
    "time_series_embedding": [0.23, 0.45, -0.12, ...],  # Optional

    # Metadata
    "created_at": "2024-01-15T12:00:00Z",
    "created_by": "llm_agent",  # or "human_annotator"
    "verified": true,  # Human-verified label
    "quality_score": 0.95  # Confidence in this example
}
```

## Embedding Strategies

### Strategy 1: Evidence-Based Embedding (Recommended)

**Rationale**: Statistical evidence captures anomaly characteristics better than raw time series.

**Implementation**:
```python
def create_evidence_embedding(evidence):
    """Convert evidence dict to vector."""

    # Option A: Simple feature vector
    features = [
        evidence.get('mae', 0),
        evidence.get('z_score', 0),
        evidence.get('volatility_ratio', 1.0),
        evidence.get('quantile_violation', 0),
        evidence.get('kl_divergence', 0),
        evidence.get('acf_max_diff', 0),
        evidence.get('surprise_score', 0),
        evidence.get('grubbs_statistic', 0),
        evidence.get('cusum_max', 0),
        evidence.get('wasserstein_distance', 0),
        # ... all metrics
    ]

    # Normalize features
    features = normalize_features(features)

    return np.array(features)
```

**Option B: Use embedding model**:
```python
from sentence_transformers import SentenceTransformer

def create_evidence_embedding_with_model(evidence):
    """Use transformer to embed evidence description."""

    # Convert evidence to natural language
    description = format_evidence_as_text(evidence)
    # e.g., "MAE: 2.34 (high), Z-score: 3.8 (extreme), Volatility: 5.2x baseline"

    # Embed using sentence transformer
    model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dim
    embedding = model.encode(description)

    return embedding
```

### Strategy 2: Time Series Embedding (Optional)

**Use Case**: Retrieve patterns with similar shapes, not just similar evidence.

**Implementation**:
```python
def create_time_series_embedding(time_series):
    """Embed time series using foundation model representations."""

    # Option A: Statistical features (Catch22, tsfresh)
    from tsfresh import extract_features
    features = extract_features(time_series)

    # Option B: Autoencoder embedding
    # (Use pre-trained time series autoencoder)
    encoder = load_pretrained_encoder()
    embedding = encoder.encode(time_series)

    # Option C: Foundation model hidden states
    # (Extract intermediate representations from TimesFM/Chronos)
    foundation_model = load_timesfm()
    hidden_states = foundation_model.get_hidden_states(time_series)
    embedding = hidden_states.mean(axis=0)  # Pool over time

    return embedding
```

### Strategy 3: Hybrid Embedding (Best)

**Combine evidence and time series embeddings**:
```python
def create_hybrid_embedding(evidence, time_series):
    """Combine evidence and time series embeddings."""

    evidence_emb = create_evidence_embedding_with_model(evidence)
    ts_emb = create_time_series_embedding(time_series)

    # Weighted concatenation
    alpha = 0.7  # Weight for evidence (more important)
    beta = 0.3   # Weight for time series shape

    hybrid = np.concatenate([
        alpha * evidence_emb,
        beta * ts_emb
    ])

    return hybrid
```

## Vector Database Selection

### Option 1: ChromaDB (Recommended for Prototyping)

**Pros**:
- Easy to use, minimal setup
- Embedded (no separate server)
- Good for < 1M vectors
- Open-source

**Cons**:
- Not scalable to billions of vectors
- Limited filtering capabilities

**Setup**:
```python
import chromadb
from chromadb.config import Settings

# Initialize client
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

# Create collection
collection = client.create_collection(
    name="anomaly_patterns",
    metadata={"description": "Historical anomaly patterns for RAG"}
)
```

### Option 2: FAISS (Recommended for Scale)

**Pros**:
- Fast similarity search (billion-scale)
- CPU and GPU support
- Industry-standard (Facebook AI)

**Cons**:
- More complex setup
- Need external DB for metadata

**Setup**:
```python
import faiss
import numpy as np

# Create index
dimension = 384  # Embedding dimension
index = faiss.IndexFlatL2(dimension)  # Simple L2 index

# For large scale, use IVF (Inverted File Index)
# nlist = 100  # Number of clusters
# quantizer = faiss.IndexFlatL2(dimension)
# index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
# index.train(training_embeddings)  # Need training

# Add vectors
embeddings = np.array([embedding1, embedding2, ...]).astype('float32')
index.add(embeddings)

# Save index
faiss.write_index(index, "anomaly_patterns.index")
```

### Option 3: Pinecone (Managed Service)

**Pros**:
- Fully managed, no infrastructure
- Serverless, scales automatically
- Built-in filtering and metadata

**Cons**:
- Costs money (free tier: 1M vectors)
- External dependency

**Setup**:
```python
import pinecone

pinecone.init(api_key="YOUR_API_KEY", environment="us-west1-gcp")

# Create index
pinecone.create_index(
    name="anomaly-patterns",
    dimension=384,
    metric="cosine"
)

index = pinecone.Index("anomaly-patterns")
```

### Recommendation

- **Prototyping**: ChromaDB (easy, embedded)
- **Production (local)**: FAISS (fast, scalable, free)
- **Production (cloud)**: Pinecone (managed, no ops)

## RAG System Implementation

### Core RAG Class

```python
class RAGSystem:
    def __init__(self, db_type='chromadb', db_path='./rag_db'):
        self.db_type = db_type
        self.db_path = db_path
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize database
        if db_type == 'chromadb':
            self.db = self._init_chromadb()
        elif db_type == 'faiss':
            self.db = self._init_faiss()
        else:
            raise ValueError(f"Unsupported db_type: {db_type}")

    def _init_chromadb(self):
        import chromadb
        client = chromadb.PersistentClient(path=self.db_path)
        collection = client.get_or_create_collection("anomaly_patterns")
        return collection

    def add_pattern(self, pattern):
        """
        Add a historical pattern to the database.

        Args:
            pattern: dict with keys [id, time_series_values, evidence, label, reasoning, ...]
        """

        # Create embedding
        evidence_text = self._format_evidence_as_text(pattern['evidence'])
        embedding = self.embedding_model.encode(evidence_text).tolist()

        # Add to database
        self.db.add(
            ids=[pattern['id']],
            embeddings=[embedding],
            metadatas=[{
                'dataset': pattern.get('dataset', 'unknown'),
                'label': pattern['label'],
                'confidence': pattern.get('confidence', 1.0),
                'anomaly_type': pattern.get('anomaly_type', 'unknown')
            }],
            documents=[pattern['reasoning']]  # Store reasoning as document
        )

        # Store full pattern separately (for retrieval)
        self._store_full_pattern(pattern)

    def retrieve(self, query_evidence, top_k=3, filter_by_label=None):
        """
        Retrieve similar historical patterns.

        Args:
            query_evidence: dict of statistical evidence
            top_k: number of similar cases to retrieve
            filter_by_label: optional filter ('anomaly', 'normal', or None)

        Returns:
            list of dicts with similar patterns
        """

        # Create query embedding
        query_text = self._format_evidence_as_text(query_evidence)
        query_embedding = self.embedding_model.encode(query_text).tolist()

        # Query database
        where_filter = {'label': filter_by_label} if filter_by_label else None
        results = self.db.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )

        # Retrieve full patterns
        similar_patterns = []
        for pattern_id in results['ids'][0]:
            full_pattern = self._load_full_pattern(pattern_id)
            similar_patterns.append(full_pattern)

        return similar_patterns

    def _format_evidence_as_text(self, evidence):
        """Convert evidence dict to natural language."""
        parts = []
        if 'mae' in evidence:
            parts.append(f"MAE: {evidence['mae']:.2f}")
        if 'z_score' in evidence:
            parts.append(f"Z-score: {evidence['z_score']:.2f}")
        if 'volatility_ratio' in evidence:
            parts.append(f"Volatility: {evidence['volatility_ratio']:.1f}x")
        # ... add all metrics
        return ", ".join(parts)

    def _store_full_pattern(self, pattern):
        """Store full pattern as JSON for later retrieval."""
        import json
        path = f"{self.db_path}/patterns/{pattern['id']}.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(pattern, f)

    def _load_full_pattern(self, pattern_id):
        """Load full pattern from storage."""
        import json
        path = f"{self.db_path}/patterns/{pattern_id}.json"
        with open(path, 'r') as f:
            return json.load(f)
```

### Populating the Database

#### Option 1: From Labeled Dataset

```python
def populate_from_dataset(rag_system, dataset, evidence_extractor):
    """Populate RAG database from labeled anomaly detection dataset."""

    for i, (time_series, labels) in enumerate(dataset):
        # Extract evidence
        train_data = time_series[:train_size]
        test_data = time_series[train_size:]
        evidence = evidence_extractor.extract(train_data, test_data, forecast=None)

        # Determine label
        has_anomaly = labels.sum() > 0
        label = 'anomaly' if has_anomaly else 'normal'

        # Create pattern entry
        pattern = {
            'id': f"dataset_{dataset.name}_{i}",
            'time_series_values': test_data.tolist(),
            'evidence': evidence,
            'label': label,
            'reasoning': f"Ground truth label: {label}",
            'dataset': dataset.name,
            'confidence': 1.0,
            'verified': True
        }

        rag_system.add_pattern(pattern)

    print(f"Added {len(dataset)} patterns to RAG database")
```

#### Option 2: From LLM Outputs

```python
def populate_from_llm_outputs(rag_system, llm_outputs, ground_truth):
    """Add LLM predictions to RAG database (continuous learning)."""

    for output, gt in zip(llm_outputs, ground_truth):
        # Only add high-quality predictions (correct and confident)
        is_correct = (output['predicted_label'] == gt['true_label'])
        is_confident = output['confidence'] > 0.8

        if is_correct and is_confident:
            pattern = {
                'id': f"llm_output_{output['id']}",
                'time_series_values': output['time_series'],
                'evidence': output['evidence'],
                'label': output['predicted_label'],
                'reasoning': output['reasoning'],
                'confidence': output['confidence'],
                'created_by': 'llm_agent',
                'verified': True  # Verified against ground truth
            }

            rag_system.add_pattern(pattern)
```

#### Option 3: Human-Curated Examples

```python
def add_curated_examples(rag_system):
    """Add expert-curated examples for specific anomaly types."""

    examples = [
        {
            'id': 'expert_001',
            'description': 'Sensor failure spike',
            'evidence': {'z_score': 4.0, 'volatility_ratio': 6.0, 'mae': 3.5},
            'label': 'anomaly',
            'reasoning': 'Hardware sensor malfunction causes sharp spike followed by return to baseline. Characteristic high Z-score and extreme volatility.',
            'anomaly_type': 'point',
            'root_cause': 'sensor_failure'
        },
        {
            'id': 'expert_002',
            'description': 'Network outage',
            'evidence': {'z_score': -5.0, 'trend_break': True, 'mae': 2.8},
            'label': 'anomaly',
            'reasoning': 'Network connectivity loss results in sudden drop to zero or near-zero values. Look for negative Z-scores and trend breaks.',
            'anomaly_type': 'contextual',
            'root_cause': 'network_outage'
        },
        # ... more curated examples
    ]

    for example in examples:
        rag_system.add_pattern(example)
```

## Integration with LLM Agent

### Context Formatting

```python
def format_rag_context(similar_patterns):
    """Format retrieved patterns for LLM prompt."""

    if not similar_patterns:
        return "No similar historical patterns found."

    lines = ["## Historical Context (Retrieved from Database)\n"]
    lines.append("Similar patterns detected in the past:\n")

    for i, pattern in enumerate(similar_patterns, 1):
        lines.append(f"### Case #{pattern['id']} ({pattern.get('dataset', 'unknown')})")

        # Pattern description
        if 'description' in pattern:
            lines.append(f"- Pattern: {pattern['description']}")

        # Evidence summary
        evidence_summary = summarize_evidence(pattern['evidence'])
        lines.append(f"- Evidence: {evidence_summary}")

        # Label
        lines.append(f"- Label: {pattern['label']} (confidence: {pattern.get('confidence', 1.0):.2f})")

        # Reasoning
        lines.append(f"- Explanation: {pattern['reasoning']}")

        lines.append("")  # Blank line

    return "\n".join(lines)

def summarize_evidence(evidence):
    """Create concise summary of evidence."""
    parts = []
    if evidence.get('z_score', 0) > 3:
        parts.append(f"Z-score={evidence['z_score']:.1f}")
    if evidence.get('volatility_ratio', 1) > 2:
        parts.append(f"Volatility spike={evidence['volatility_ratio']:.1f}x")
    if evidence.get('quantile_violation'):
        parts.append("Quantile violation")
    # ... summarize key metrics
    return ", ".join(parts)
```

### Usage in LLM Agent

```python
class LLMAnomalyAgent:
    def __init__(self, backend, rag_system=None):
        self.backend = backend
        self.rag_system = rag_system

    def analyze_window(self, time_series, evidence, metadata=None):
        # Retrieve similar patterns
        rag_context = ""
        if self.rag_system:
            similar_patterns = self.rag_system.retrieve(
                query_evidence=evidence,
                top_k=3,
                filter_by_label=None  # Retrieve both anomalies and normal cases
            )
            rag_context = format_rag_context(similar_patterns)

        # Build prompt with RAG context
        prompt = build_prompt(time_series, evidence, rag_context)

        # Generate completion
        output = self.backend.generate(prompt)

        return output
```

## Continuous Learning

### Feedback Loop

```python
def continuous_learning_loop(rag_system, llm_agent, new_data, ground_truth):
    """Continuously improve RAG database with new examples."""

    for time_series, labels in zip(new_data, ground_truth):
        # Get LLM prediction
        evidence = extract_evidence(time_series)
        output = llm_agent.analyze_window(time_series, evidence)

        # Evaluate correctness
        predicted_labels = convert_to_binary_labels(output)
        is_correct = evaluate_prediction(predicted_labels, labels)

        # Add to RAG if high quality
        if is_correct and output['confidence'] > 0.8:
            pattern = create_pattern_from_output(time_series, evidence, output, labels)
            rag_system.add_pattern(pattern)
            print(f"Added new pattern: {pattern['id']}")
```

### Quality Control

```python
def audit_rag_database(rag_system, validation_set):
    """Remove low-quality patterns from database."""

    low_quality_ids = []

    for pattern in rag_system.get_all_patterns():
        # Check if pattern is still useful
        if pattern.get('confidence', 1.0) < 0.5:
            low_quality_ids.append(pattern['id'])

        # Check if reasoning is too generic
        if is_generic_reasoning(pattern['reasoning']):
            low_quality_ids.append(pattern['id'])

    # Remove low-quality patterns
    for pattern_id in low_quality_ids:
        rag_system.delete_pattern(pattern_id)

    print(f"Removed {len(low_quality_ids)} low-quality patterns")
```

## Evaluation

### RAG Impact Analysis

**Ablation study**:
```python
def evaluate_rag_impact(llm_agent_with_rag, llm_agent_without_rag, test_set):
    """Compare performance with and without RAG."""

    results_with_rag = evaluate(llm_agent_with_rag, test_set)
    results_without_rag = evaluate(llm_agent_without_rag, test_set)

    improvement = {
        'f1_delta': results_with_rag['f1'] - results_without_rag['f1'],
        'precision_delta': results_with_rag['precision'] - results_without_rag['precision'],
        'recall_delta': results_with_rag['recall'] - results_without_rag['recall']
    }

    return improvement
```

### Retrieval Quality

**Metrics**:
- **Retrieval accuracy**: Are retrieved patterns actually similar?
- **Diversity**: Do retrieved patterns cover different anomaly types?
- **Relevance**: Do retrieved patterns help LLM reasoning?

```python
def evaluate_retrieval_quality(rag_system, test_queries):
    """Evaluate retrieval quality."""

    relevance_scores = []

    for query in test_queries:
        retrieved = rag_system.retrieve(query['evidence'], top_k=5)

        # Human annotator scores relevance (1-5)
        for pattern in retrieved:
            score = human_rate_relevance(query, pattern)
            relevance_scores.append(score)

    return {
        'mean_relevance': np.mean(relevance_scores),
        'precision_at_k': np.mean([s >= 3 for s in relevance_scores])  # 3+ is relevant
    }
```

## Configuration

```yaml
rag_system:
  database:
    type: "chromadb"         # chromadb, faiss, pinecone
    path: "./rag_db"         # Local path for chromadb/faiss
    collection_name: "anomaly_patterns"

  embedding:
    model: "all-MiniLM-L6-v2"  # Sentence transformer model
    strategy: "evidence_based"  # evidence_based, time_series, hybrid
    dimension: 384              # Embedding dimension

  retrieval:
    top_k: 3                    # Number of similar cases to retrieve
    similarity_metric: "cosine" # cosine, l2, dot_product
    min_similarity: 0.7         # Minimum similarity threshold

  population:
    auto_add_llm_outputs: true  # Add high-quality LLM predictions
    min_confidence: 0.8         # Min confidence to auto-add
    require_verification: false # Require human verification

  quality_control:
    audit_frequency: 100        # Audit every N additions
    min_pattern_quality: 0.5    # Remove patterns below this score
    remove_generic: true        # Remove patterns with generic reasoning
```

## Next Steps

1. Implement `RAGSystem` class with ChromaDB backend
2. Populate database with labeled datasets (UCR, Yahoo, NAB)
3. Add curated examples for common anomaly types
4. Integrate with `LLMAnomalyAgent`
5. Evaluate RAG impact on detection performance
6. Implement continuous learning loop

---

**Status**: Specification complete, ready for implementation
**Last Updated**: 2026-02-17
**Dependencies**: ChromaDB or FAISS, sentence-transformers, numpy
