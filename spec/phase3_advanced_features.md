# Phase 3: Advanced Features & Integration (Weeks 5-6)

## Overview

This phase adds advanced features to enhance the foundation model + LLM pipeline: RAG system for historical context, prompt optimization for better LLM reasoning, integration of pre-trained models, and system-wide optimization.

**Timeline**: 2 weeks
**Status**: 📋 Planned
**Prerequisites**: Phase 2 (Core Implementation) completed

---

## Objectives

1. ✅ Implement RAG system for historical pattern retrieval
2. ✅ Optimize prompts for better LLM reasoning
3. ✅ Integrate pre-trained models (AER, Transformer) as optional evidence
4. ✅ Add cost optimization strategies
5. ✅ Improve system reliability and error handling
6. ✅ Create comprehensive documentation and examples

**Goal**: By end of Phase 3, have a production-ready system with historical learning, optimized prompts, and cost-effective operation.

---

## Phase Breakdown

### Week 1: RAG System Implementation

**Deliverable**: Working RAG system that retrieves similar historical patterns

#### Tasks

**1.1 Set Up Vector Database**

**Install Dependencies**:
```bash
pip install chromadb  # Vector database
pip install sentence-transformers  # For embeddings
# Alternative: pip install faiss-cpu  # For FAISS
```

**File**: `src/rag/__init__.py`
```python
from .vector_store import VectorStore
from .retrieval_engine import RetrievalEngine
from .pattern_encoder import PatternEncoder

__all__ = ['VectorStore', 'RetrievalEngine', 'PatternEncoder']
```

**1.2 Implement Vector Store**

**File**: `src/rag/vector_store.py`
```python
import chromadb
from typing import Dict, List, Optional
import numpy as np

class VectorStore:
    """Vector database for storing historical anomaly patterns."""

    def __init__(self, config: Dict):
        self.collection_name = config.get('collection_name', 'anomaly_patterns')
        self.persist_directory = config.get('persist_directory', './data/vector_db')

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_pattern(
        self,
        pattern_id: str,
        embedding: np.ndarray,
        metadata: Dict,
        time_series: Optional[np.ndarray] = None
    ):
        """
        Add anomaly pattern to database.

        Args:
            pattern_id: Unique identifier
            embedding: Vector representation of evidence profile
            metadata: {
                'evidence': dict,
                'label': bool (anomaly or not),
                'reasoning': str,
                'dataset': str,
                'timestamp': str
            }
            time_series: Optional time series data
        """
        self.collection.add(
            ids=[pattern_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata]
        )

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        min_similarity: float = 0.7
    ) -> List[Dict]:
        """
        Retrieve similar patterns.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            min_similarity: Minimum cosine similarity threshold

        Returns:
            List of similar patterns with metadata
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        # Filter by similarity threshold
        similar_patterns = []
        for i, (id_, metadata, distance) in enumerate(zip(
            results['ids'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            similarity = 1 - distance  # Convert distance to similarity
            if similarity >= min_similarity:
                similar_patterns.append({
                    'id': id_,
                    'similarity': similarity,
                    'metadata': metadata
                })

        return similar_patterns

    def count(self) -> int:
        """Return number of patterns in database."""
        return self.collection.count()
```

**1.3 Implement Pattern Encoder**

**File**: `src/rag/pattern_encoder.py`
```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict

class PatternEncoder:
    """Encode evidence profiles into vector embeddings."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize encoder.

        Args:
            model_name: SentenceTransformer model
                - 'all-MiniLM-L6-v2': Fast, 384 dims
                - 'all-mpnet-base-v2': Better quality, 768 dims
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode_evidence(self, evidence: Dict) -> np.ndarray:
        """
        Convert evidence dictionary to embedding.

        Args:
            evidence: Statistical evidence dict

        Returns:
            Vector embedding
        """
        # Create text representation of evidence
        evidence_text = self._evidence_to_text(evidence)

        # Encode with SentenceTransformer
        embedding = self.model.encode(evidence_text, convert_to_numpy=True)

        return embedding

    def _evidence_to_text(self, evidence: Dict) -> str:
        """Convert evidence dict to text for encoding."""
        parts = []

        # Forecast-based
        if 'mae' in evidence:
            parts.append(f"MAE: {evidence['mae']:.2f}")
        if 'mae_anomalous' in evidence:
            parts.append(f"MAE anomalous: {evidence['mae_anomalous']}")

        # Statistical tests
        if 'z_score' in evidence:
            parts.append(f"Z-score: {evidence['z_score']:.2f}")
        if 'extreme_z_count' in evidence:
            parts.append(f"Extreme Z count: {evidence['extreme_z_count']}")

        # Distribution
        if 'kl_divergence' in evidence:
            parts.append(f"KL divergence: {evidence['kl_divergence']:.2f}")

        # Pattern
        if 'volatility_ratio' in evidence:
            parts.append(f"Volatility ratio: {evidence['volatility_ratio']:.2f}")
        if 'autocorr_break' in evidence:
            parts.append(f"Autocorrelation break: {evidence['autocorr_break']}")

        return ". ".join(parts)
```

**1.4 Implement Retrieval Engine**

**File**: `src/rag/retrieval_engine.py`
```python
from typing import Dict, List
from .vector_store import VectorStore
from .pattern_encoder import PatternEncoder

class RetrievalEngine:
    """RAG retrieval engine for historical patterns."""

    def __init__(self, config: Dict):
        self.config = config
        self.vector_store = VectorStore(config.get('vector_store', {}))
        self.encoder = PatternEncoder(
            model_name=config.get('encoder_model', 'all-MiniLM-L6-v2')
        )

    def retrieve_similar_patterns(
        self,
        evidence: Dict,
        top_k: int = 3,
        min_similarity: float = 0.7
    ) -> List[Dict]:
        """
        Retrieve similar historical patterns.

        Args:
            evidence: Current evidence dict
            top_k: Number of similar patterns to retrieve
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar patterns with metadata
        """
        # Encode current evidence
        query_embedding = self.encoder.encode_evidence(evidence)

        # Query vector database
        similar_patterns = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            min_similarity=min_similarity
        )

        return similar_patterns

    def add_pattern(
        self,
        pattern_id: str,
        evidence: Dict,
        label: bool,
        reasoning: str = "",
        dataset: str = "",
    ):
        """
        Add new pattern to database.

        Args:
            pattern_id: Unique ID
            evidence: Evidence dict
            label: True if anomaly, False if normal
            reasoning: LLM explanation
            dataset: Dataset name
        """
        # Encode evidence
        embedding = self.encoder.encode_evidence(evidence)

        # Prepare metadata
        metadata = {
            'label': label,
            'reasoning': reasoning,
            'dataset': dataset,
            # Store compact evidence representation
            'evidence_summary': self._summarize_evidence(evidence)
        }

        # Add to vector store
        self.vector_store.add_pattern(
            pattern_id=pattern_id,
            embedding=embedding,
            metadata=metadata
        )

    def _summarize_evidence(self, evidence: Dict) -> str:
        """Create compact string summary of evidence."""
        summary_parts = []
        if 'mae' in evidence:
            summary_parts.append(f"MAE={evidence['mae']:.2f}")
        if 'z_score' in evidence:
            summary_parts.append(f"Z={evidence['z_score']:.2f}")
        if 'volatility_ratio' in evidence:
            summary_parts.append(f"Vol={evidence['volatility_ratio']:.1f}x")
        return ", ".join(summary_parts)
```

**1.5 Integrate RAG with LLM Agent**

**File**: `src/llm/llm_agent.py` (update)
```python
class LLMAnomalyAgent:
    def __init__(self, backend, rag_system=None):
        self.backend = backend
        self.rag_system = rag_system  # RetrievalEngine instance

    def analyze_window(self, time_series, evidence, metadata=None):
        """Analyze with RAG context."""

        # Retrieve similar patterns if RAG enabled
        rag_context = ""
        if self.rag_system:
            similar_patterns = self.rag_system.retrieve_similar_patterns(
                evidence=evidence,
                top_k=3,
                min_similarity=0.7
            )

            # Format RAG context
            if similar_patterns:
                rag_context = self._format_rag_context(similar_patterns)

        # Build prompt with RAG context
        user_prompt = USER_PROMPT_TEMPLATE.format(
            time_series_formatted=format_time_series(time_series),
            evidence_formatted=format_evidence(evidence),
            rag_context=rag_context  # Inject RAG context
        )

        # Generate LLM response
        llm_output = self.backend.generate(SYSTEM_PROMPT, user_prompt)

        # Parse output
        result = self.parse_output(llm_output)

        return result

    def _format_rag_context(self, similar_patterns):
        """Format retrieved patterns for prompt."""
        if not similar_patterns:
            return "No similar historical patterns found."

        lines = ["## Historical Context (Similar Patterns)\n"]
        for i, pattern in enumerate(similar_patterns, 1):
            meta = pattern['metadata']
            lines.append(f"### Pattern #{i} (Similarity: {pattern['similarity']:.2f})")
            lines.append(f"- Evidence: {meta.get('evidence_summary', 'N/A')}")
            lines.append(f"- Label: {'Anomaly' if meta.get('label') else 'Normal'}")
            lines.append(f"- Reasoning: {meta.get('reasoning', 'N/A')}")
            lines.append(f"- Dataset: {meta.get('dataset', 'Unknown')}\n")

        return "\n".join(lines)
```

**1.6 Update Configuration**

**File**: `configs/pipelines/phase3_with_rag.yaml`
```yaml
step3_scoring:
  type: "LLMReasoningScoring"
  backend: "gemini"
  model: "gemini-1.5-pro"

  # RAG configuration
  use_rag: true
  rag_config:
    vector_store:
      collection_name: "anomaly_patterns"
      persist_directory: "./data/vector_db"
    encoder_model: "all-MiniLM-L6-v2"
    top_k: 3
    min_similarity: 0.7
```

**Week 1 Success Criteria**:
- [ ] Vector database stores and retrieves patterns
- [ ] Pattern encoder converts evidence to embeddings
- [ ] Retrieval engine finds similar patterns
- [ ] RAG context injected into LLM prompts
- [ ] Tests show RAG improves consistency

---

### Week 2: Optimization & Integration

**Deliverable**: Optimized system with cost controls and pre-trained model integration

#### Tasks

**2.1 Prompt Optimization**

**File**: `src/llm/prompt_optimizer.py`
```python
class PromptOptimizer:
    """Optimize prompts for better LLM performance."""

    @staticmethod
    def optimize_evidence_formatting(evidence: Dict) -> str:
        """Format evidence with emphasis on important signals."""

        # Categorize evidence by importance
        critical = []
        important = []
        supporting = []

        # Critical: Extreme signals
        if evidence.get('mae_anomalous'):
            critical.append(f"⚠️ MAE: {evidence['mae']:.2f} (ANOMALOUS)")
        if abs(evidence.get('z_score', 0)) > 3:
            critical.append(f"⚠️ Z-Score: {evidence['z_score']:.2f} (EXTREME)")
        if evidence.get('any_extreme_violation'):
            critical.append("⚠️ Extreme quantile violation detected")

        # Important: Strong signals
        if evidence.get('volatility_ratio', 1) > 3:
            important.append(f"Volatility spike: {evidence['volatility_ratio']:.1f}x baseline")
        if evidence.get('kl_divergence', 0) > 0.5:
            important.append(f"High distributional shift: {evidence['kl_divergence']:.2f}")

        # Supporting: Other metrics
        # ... add remaining metrics

        # Format with hierarchy
        formatted = []
        if critical:
            formatted.append("🚨 CRITICAL SIGNALS:")
            formatted.extend([f"  • {s}" for s in critical])
        if important:
            formatted.append("\n📊 IMPORTANT SIGNALS:")
            formatted.extend([f"  • {s}" for s in important])
        if supporting:
            formatted.append("\n📈 SUPPORTING SIGNALS:")
            formatted.extend([f"  • {s}" for s in supporting])

        return "\n".join(formatted)
```

**2.2 Cost Optimization**

**File**: `src/llm/cost_optimizer.py`
```python
class CostOptimizer:
    """Strategies to reduce LLM API costs."""

    def __init__(self, config: Dict):
        self.config = config
        self.cache = {}  # Simple response cache

    def should_use_llm(self, evidence: Dict) -> bool:
        """
        Decide if LLM is needed or if statistical threshold suffices.

        Use LLM only for uncertain cases.
        """
        # Clear cases: use statistical threshold
        extreme_signals = sum([
            evidence.get('mae_anomalous', False),
            abs(evidence.get('z_score', 0)) > 3,
            evidence.get('any_extreme_violation', False),
            evidence.get('volatility_ratio', 1) > 5
        ])

        # If 3+ extreme signals, it's clearly anomalous (no LLM needed)
        if extreme_signals >= 3:
            return False

        # If no extreme signals, likely normal (no LLM needed)
        if extreme_signals == 0:
            return False

        # Uncertain case (1-2 signals): use LLM
        return True

    def get_cheapest_model(self, task_complexity: str) -> str:
        """Select cheapest adequate model."""
        if task_complexity == 'simple':
            return 'gemini-2.0-flash'  # $0.15/M tokens
        elif task_complexity == 'moderate':
            return 'gpt-4o-mini'  # $0.30/M tokens
        else:
            return 'gemini-1.5-pro'  # $3.50/M tokens

    def check_cache(self, evidence: Dict) -> Optional[Dict]:
        """Check if similar evidence was seen before."""
        # Create cache key from evidence
        cache_key = self._evidence_to_key(evidence)

        return self.cache.get(cache_key)

    def add_to_cache(self, evidence: Dict, result: Dict):
        """Cache LLM response."""
        cache_key = self._evidence_to_key(evidence)
        self.cache[cache_key] = result
```

**2.3 Integrate Pre-trained Models**

**File**: `src/evidence/pretrained_models.py`
```python
import torch
from src.models.aer import AER
from src.models.anomaly_transformer import AnomalyTransformer

class PretrainedModelEvidence:
    """Extract evidence from pre-trained Phase 2 models."""

    def __init__(self, config: Dict):
        self.config = config
        self.models = {}

        # Load AER if configured
        if config.get('use_aer', False):
            self.models['aer'] = self._load_aer(
                config.get('aer_weights_path')
            )

        # Load Anomaly Transformer if configured
        if config.get('use_transformer', False):
            self.models['transformer'] = self._load_transformer(
                config.get('transformer_weights_path')
            )

    def _load_aer(self, weights_path: str):
        """Load pre-trained AER model."""
        model = AER(
            window_size=100,
            input_dim=1,
            hidden_dim=64,
            num_layers=2
        )
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        model.eval()
        return model

    def extract(self, window: np.ndarray) -> Dict:
        """Run inference on pre-trained models."""
        evidence = {}

        # AER score
        if 'aer' in self.models:
            with torch.no_grad():
                window_tensor = torch.FloatTensor(window).unsqueeze(0)
                aer_score = self.models['aer'](window_tensor).item()
                evidence['aer_score'] = aer_score
                evidence['aer_anomalous'] = aer_score > 0.5

        # Transformer score
        if 'transformer' in self.models:
            with torch.no_grad():
                window_tensor = torch.FloatTensor(window).unsqueeze(0)
                transformer_score = self.models['transformer'](window_tensor).item()
                evidence['transformer_score'] = transformer_score
                evidence['transformer_anomalous'] = transformer_score > 0.5

        return evidence
```

**Integrate into evidence extractor** (`src/evidence/evidence_extractor.py`):
```python
from .pretrained_models import PretrainedModelEvidence

class StatisticalEvidenceExtractor:
    def __init__(self, config: Dict):
        # ... existing code ...

        # Add pre-trained models if configured
        if config.get('use_pretrained_models', False):
            self.pretrained_extractor = PretrainedModelEvidence(
                config.get('pretrained_config', {})
            )
```

**2.4 Error Handling & Reliability**

**File**: `src/utils/error_handling.py`
```python
import time
from typing import Callable, Any

def retry_with_exponential_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0
) -> Any:
    """Retry function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = initial_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            time.sleep(delay)

class FallbackHandler:
    """Handle LLM failures gracefully."""

    @staticmethod
    def statistical_fallback(evidence: Dict) -> Dict:
        """Fallback to statistical threshold if LLM fails."""
        # Simple rule-based decision
        anomaly_score = 0

        if evidence.get('mae_anomalous'):
            anomaly_score += 0.3
        if abs(evidence.get('z_score', 0)) > 3:
            anomaly_score += 0.3
        if evidence.get('any_extreme_violation'):
            anomaly_score += 0.2
        if evidence.get('volatility_ratio', 1) > 3:
            anomaly_score += 0.2

        is_anomaly = anomaly_score > 0.5

        return {
            'anomalies': [{'start': 0, 'end': 100, 'confidence': anomaly_score}] if is_anomaly else [],
            'overall_assessment': f'Statistical fallback: score={anomaly_score:.2f}',
            'fallback': True
        }
```

**2.5 Documentation & Examples**

**File**: `notebooks/phase3_advanced_features.ipynb`
```python
# Jupyter notebook demonstrating:
# 1. RAG system usage
# 2. Cost optimization strategies
# 3. Pre-trained model integration
# 4. Error handling
```

**Week 2 Success Criteria**:
- [ ] Prompt optimization improves LLM quality
- [ ] Cost optimizer reduces API costs by 30%+
- [ ] Pre-trained models integrated successfully
- [ ] Error handling prevents pipeline crashes
- [ ] Documentation complete with examples

---

## Phase 3 Completion Checklist

### RAG System
- [ ] Vector database implemented (ChromaDB or FAISS)
- [ ] Pattern encoder converts evidence to embeddings
- [ ] Retrieval engine finds similar patterns
- [ ] RAG context improves LLM consistency (measured)
- [ ] Can add new patterns dynamically

### Optimization
- [ ] Prompt optimization implemented
- [ ] Cost optimizer reduces unnecessary LLM calls
- [ ] Response caching implemented
- [ ] Model selection based on complexity
- [ ] Cost tracking and reporting

### Integration
- [ ] Pre-trained models (AER, Transformer) integrated
- [ ] Works with or without pre-trained weights
- [ ] Fallback to statistical baseline if LLM fails
- [ ] Retry logic for API failures
- [ ] Comprehensive error handling

### Documentation
- [ ] RAG system usage guide
- [ ] Cost optimization best practices
- [ ] Pre-trained model integration guide
- [ ] API reference updated
- [ ] Example notebooks created

---

## Expected Outcomes

By end of Phase 3, you should have:

1. **RAG-Enhanced System**: LLM reasoning improves with historical context
2. **Cost-Effective Operation**: 30-50% reduction in API costs through optimization
3. **Hybrid Evidence**: Combines foundation models, statistics, and pre-trained models
4. **Reliable System**: Handles failures gracefully with fallbacks
5. **Production-Ready**: Comprehensive error handling and monitoring

**Performance Target**: F1 > 0.75 with explainable outputs

---

## Next Phase

**Phase 4: Evaluation & Optimization**
- Comprehensive benchmarking on multiple datasets
- Ablation studies to identify critical components
- Performance optimization
- Production deployment preparation

---

**Status**: Planned
**Last Updated**: 2026-02-17
**Prerequisites**: Phase 2 complete, vector database installed
