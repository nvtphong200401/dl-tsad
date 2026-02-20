# LLM Reasoning Pipeline

## Position in Pipeline

This framework implements **Step 3: Scoring via LLM Reasoning** in the 4-step pipeline (Phase 3).

**Pipeline Context:**
- **Step 1** (Preprocessing): Generates forecasts via foundation models
- **Step 2** (Detection): Extracts 10+ statistical evidence metrics
- **Step 3** (Scoring): **← THIS FRAMEWORK** - LLM aggregates evidence into scores + explanations
- **Step 4** (Post-Processing): Parses LLM outputs and generates final predictions

**Key Insight**: LLM reasoning IS a scoring method that replaces heuristic pooling (max/average) with intelligent aggregation. Both Phase 2 and Phase 3 scoring methods convert multiple signals into actionable scores—Phase 2 uses heuristics, Phase 3 uses contextual understanding.

**Conceptual Shift:**
- **Phase 2 Scoring**: Multiple window scores → Max/Average pooling → Point-wise scores (heuristic)
- **Phase 3 Scoring**: Statistical evidence + Time series → LLM reasoning → Anomaly ranges + Explanations (intelligent)

## Overview

This document specifies the **LLM reasoning layer** for time series anomaly detection. The LLM acts as an intelligent judge that:
1. Receives time series data + statistical evidence (from Step 2)
2. Reasons about anomalies using contextual understanding
3. Provides explainable decisions with cited evidence
4. Outputs structured anomaly labels + confidence + explanations

This approach is inspired by **LLM-TSAD's AnoAgent** but enhanced with statistical grounding from foundation models.

## Design Principles

1. **Evidence-Grounded**: LLM must cite statistical signals, not hallucinate
2. **Structured Output**: JSON format for easy parsing
3. **Explainable**: Provide human-readable reasoning for each decision
4. **Contextual**: Use RAG to inject relevant historical patterns
5. **Model-Agnostic**: Support multiple LLM backends (GPT-4, Gemini, Claude)

## Supported LLM Models

### Tier 1: Production Models
- **GPT-4 Turbo (OpenAI)**: Best reasoning quality, expensive ($10-30/M tokens)
- **Gemini 1.5 Pro (Google)**: Long context (1M tokens), cost-effective ($3.50-7/M tokens)
- **Claude 3 Opus (Anthropic)**: Strong reasoning, moderate cost ($15-75/M tokens)

### Tier 2: Fast/Cheap Models
- **Gemini 2.0 Flash (Google)**: Fast, cheap ($0.15-0.60/M tokens), good for prototyping
- **GPT-4o Mini (OpenAI)**: Lightweight GPT-4 variant ($0.15-0.60/M tokens)
- **Claude 3.5 Sonnet (Anthropic)**: Balanced speed and quality ($3-15/M tokens)

### Selection Strategy
- **Development**: Use Gemini 2.0 Flash for rapid iteration
- **Evaluation**: Use GPT-4 Turbo or Gemini 1.5 Pro for best results
- **Production**: Depends on latency/cost requirements

## Prompt Engineering

### Prompt Structure

```
[SYSTEM ROLE]
You are an expert time series analyst specializing in anomaly detection...

[TIME SERIES DATA]
Timestamp | Value
2024-01-01 00:00:00 | 1.23
2024-01-01 00:01:00 | 1.25
...

[STATISTICAL EVIDENCE]
Foundation Model Forecast:
- Mean Absolute Error (MAE): 2.34 (95th percentile: ANOMALOUS)
- Quantile Violation: Value exceeds P99 (99th percentile)
- Surprise Score: 8.5 (high unlikelihood)

Statistical Tests:
- Z-Score: 3.8 (exceeds 3-sigma threshold: ANOMALOUS)
- Grubbs Test: Outlier detected at position 45
- CUSUM: Change point detected at position 42

Distribution Analysis:
- KL Divergence: 0.45 (high distributional shift)
- Volatility Ratio: 5.2x baseline (extreme spike)

Pattern Analysis:
- Autocorrelation Break: Periodicity changed from 24h to irregular
- Trend Break: Level shift detected at position 40

[HISTORICAL CONTEXT] (from RAG)
Similar patterns detected in:
- Case #127 (2023-11-15): Sensor failure with volatility spike (5.1x)
- Case #203 (2024-01-03): Network outage with sudden level drop

[TASK]
Based on the evidence above, determine:
1. Are there anomalies in this time series window?
2. If yes, specify exact timestamp ranges
3. Provide confidence score (0-1) for each anomaly
4. Explain reasoning, citing specific evidence

Output format: JSON
```

### Detailed Prompt Template

```python
SYSTEM_PROMPT = """You are an expert time series analyst with deep knowledge of statistical methods and anomaly detection. Your task is to analyze time series data and identify anomalies based on:
1. Visual patterns in the time series
2. Statistical evidence from forecasting models and hypothesis tests
3. Historical patterns from similar cases

Guidelines:
- Base your reasoning on the statistical evidence provided
- Cite specific metrics when explaining anomalies (e.g., "Z-score of 3.8 exceeds threshold")
- Consider multiple lines of evidence (convergent signals are stronger)
- Distinguish between noise and true anomalies
- Provide confidence scores based on evidence strength
- If evidence is weak or contradictory, express uncertainty

Output your analysis as valid JSON with this structure:
{
  "anomalies": [
    {
      "start_timestamp": "2024-01-01 00:42:00",
      "end_timestamp": "2024-01-01 00:48:00",
      "confidence": 0.92,
      "severity": "high",
      "reasoning": "Extreme Z-score (3.8) combined with quantile violation (actual > P99) and 5.2x volatility spike. Pattern matches historical case #127 (sensor failure).",
      "evidence_cited": ["z_score", "quantile_violation", "volatility_spike", "historical_pattern"]
    }
  ],
  "overall_assessment": "High confidence anomaly detected due to convergent statistical signals and historical pattern match.",
  "uncertainty_factors": ["Limited historical context for this specific pattern"]
}
"""

USER_PROMPT_TEMPLATE = """
# Time Series Analysis Request

## Time Series Data
{time_series_formatted}

## Statistical Evidence

### Foundation Model Forecast
{forecast_evidence}

### Statistical Tests
{statistical_test_evidence}

### Distribution Analysis
{distribution_evidence}

### Pattern Analysis
{pattern_evidence}

{optional_pretrained_evidence}

## Historical Context (RAG)
{rag_context}

## Analysis Request
Please analyze this time series window and identify any anomalies. Provide structured output as specified in your system instructions.
"""
```

### Evidence Formatting

Transform evidence dictionary into human-readable format:

```python
def format_forecast_evidence(evidence):
    lines = ["Foundation Model Forecast:"]

    # MAE
    if 'mae' in evidence:
        mae_status = "ANOMALOUS" if evidence.get('mae_anomalous') else "NORMAL"
        lines.append(f"- Mean Absolute Error (MAE): {evidence['mae']:.2f} "
                    f"({evidence.get('mae_percentile', 0):.1f}th percentile: {mae_status})")

    # Quantile violations
    if 'quantile_violations' in evidence:
        violations = evidence['quantile_violations']
        if violations.get('above_p99'):
            lines.append("- Quantile Violation: Value EXCEEDS P99 (99th percentile) - ANOMALOUS")
        elif violations.get('below_p01'):
            lines.append("- Quantile Violation: Value BELOW P01 (1st percentile) - ANOMALOUS")

    # Surprise score
    if 'surprise_score' in evidence:
        lines.append(f"- Surprise Score: {evidence['surprise_score']:.2f} "
                    "(high = unlikely under forecast distribution)")

    return "\n".join(lines)

def format_statistical_tests(evidence):
    lines = ["Statistical Tests:"]

    # Z-score
    if 'z_score' in evidence:
        z = evidence['z_score']
        status = "ANOMALOUS" if abs(z) > 3 else "NORMAL"
        lines.append(f"- Z-Score: {z:.2f} (threshold: 3.0, status: {status})")

    # Grubbs test
    if 'grubbs_outlier' in evidence:
        if evidence['grubbs_outlier']:
            idx = evidence.get('outlier_index', 'unknown')
            lines.append(f"- Grubbs Test: Outlier detected at position {idx}")

    # CUSUM
    if 'cusum_change_points' in evidence:
        cps = evidence['cusum_change_points']
        if len(cps) > 0:
            lines.append(f"- CUSUM: Change points detected at positions {cps}")

    return "\n".join(lines)

# Similar functions for distribution_evidence, pattern_evidence
```

### Time Series Formatting

Present time series data in a clear, parseable format:

```python
def format_time_series(timestamps, values):
    """Format time series for LLM prompt."""

    # Option 1: Tabular format (preferred for short windows)
    if len(values) <= 100:
        lines = ["Timestamp | Value"]
        lines.append("-" * 40)
        for ts, val in zip(timestamps, values):
            lines.append(f"{ts} | {val:.4f}")
        return "\n".join(lines)

    # Option 2: Compact format (for longer windows)
    else:
        # Show first N, middle N, last N
        n = 20
        lines = ["Time Series (showing first 20, middle 20, last 20 points):"]
        lines.append(f"Start: {timestamps[0]} to {timestamps[n-1]}")
        for i in range(n):
            lines.append(f"  t{i}: {values[i]:.4f}")

        lines.append(f"... ({len(values) - 3*n} points omitted) ...")

        mid = len(values) // 2
        lines.append(f"Middle: {timestamps[mid]} to {timestamps[mid+n-1]}")
        for i in range(mid, mid+n):
            lines.append(f"  t{i}: {values[i]:.4f}")

        lines.append(f"End: {timestamps[-n]} to {timestamps[-1]}")
        for i in range(-n, 0):
            lines.append(f"  t{len(values)+i}: {values[i]:.4f}")

        return "\n".join(lines)
```

## LLM Integration

### API Wrapper

Support multiple LLM backends with a unified interface:

```python
from abc import ABC, abstractmethod

class LLMBackend(ABC):
    @abstractmethod
    def generate(self, system_prompt, user_prompt, temperature=0.0):
        """Generate completion from LLM."""
        pass

class OpenAIBackend(LLMBackend):
    def __init__(self, api_key, model="gpt-4-turbo"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system_prompt, user_prompt, temperature=0.0):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            response_format={"type": "json_object"}  # Force JSON output
        )
        return response.choices[0].message.content

class GeminiBackend(LLMBackend):
    def __init__(self, api_key, model="gemini-1.5-pro"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def generate(self, system_prompt, user_prompt, temperature=0.0):
        # Gemini doesn't have system prompts, prepend to user prompt
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self.model.generate_content(
            full_prompt,
            generation_config={"temperature": temperature}
        )
        return response.text

class ClaudeBackend(LLMBackend):
    def __init__(self, api_key, model="claude-3-opus-20240229"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system_prompt, user_prompt, temperature=0.0):
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
            max_tokens=4096
        )
        return response.content[0].text
```

### LLM Agent Class

Main class that orchestrates prompting and output parsing:

```python
class LLMAnomalyAgent:
    def __init__(self, backend, rag_system=None):
        self.backend = backend
        self.rag_system = rag_system

    def analyze_window(self, time_series, evidence, metadata=None):
        """
        Analyze a time series window using LLM reasoning.

        Args:
            time_series: dict with 'timestamps' and 'values'
            evidence: dict from StatisticalEvidenceExtractor
            metadata: optional context (dataset name, domain, etc.)

        Returns:
            dict with anomaly labels, confidence, reasoning
        """

        # Step 1: Format evidence for prompt
        forecast_evidence = format_forecast_evidence(evidence)
        statistical_tests = format_statistical_tests(evidence)
        distribution_evidence = format_distribution_evidence(evidence)
        pattern_evidence = format_pattern_evidence(evidence)

        # Step 2: Retrieve historical context via RAG
        rag_context = ""
        if self.rag_system:
            similar_cases = self.rag_system.retrieve(evidence, top_k=3)
            rag_context = format_rag_context(similar_cases)

        # Step 3: Format time series
        ts_formatted = format_time_series(
            time_series['timestamps'],
            time_series['values']
        )

        # Step 4: Build prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            time_series_formatted=ts_formatted,
            forecast_evidence=forecast_evidence,
            statistical_test_evidence=statistical_tests,
            distribution_evidence=distribution_evidence,
            pattern_evidence=pattern_evidence,
            optional_pretrained_evidence=format_pretrained_evidence(evidence),
            rag_context=rag_context
        )

        # Step 5: Generate completion
        llm_output = self.backend.generate(SYSTEM_PROMPT, user_prompt, temperature=0.0)

        # Step 6: Parse output
        result = self.parse_output(llm_output)

        return result

    def parse_output(self, llm_output):
        """Parse LLM JSON output into structured format."""
        import json

        try:
            parsed = json.loads(llm_output)

            # Validate structure
            assert 'anomalies' in parsed, "Missing 'anomalies' field"
            assert isinstance(parsed['anomalies'], list), "'anomalies' must be a list"

            # Validate each anomaly
            for anom in parsed['anomalies']:
                assert 'start_timestamp' in anom, "Missing 'start_timestamp'"
                assert 'end_timestamp' in anom, "Missing 'end_timestamp'"
                assert 'confidence' in anom, "Missing 'confidence'"
                assert 'reasoning' in anom, "Missing 'reasoning'"

            return parsed

        except (json.JSONDecodeError, AssertionError) as e:
            # Fallback: try to extract info from text
            print(f"Failed to parse LLM output: {e}")
            return self.fallback_parsing(llm_output)

    def fallback_parsing(self, llm_output):
        """Fallback parser if JSON parsing fails."""
        # Simple heuristic: look for keywords
        has_anomaly = any(word in llm_output.lower() for word in ['anomaly', 'anomalous', 'outlier'])

        return {
            'anomalies': [],
            'overall_assessment': llm_output,
            'parse_error': True,
            'fallback_anomaly_detected': has_anomaly
        }
```

## Output Parsing

### Expected Output Format

```json
{
  "anomalies": [
    {
      "start_timestamp": "2024-01-01 00:42:00",
      "end_timestamp": "2024-01-01 00:48:00",
      "start_index": 42,
      "end_index": 48,
      "confidence": 0.92,
      "severity": "high",
      "reasoning": "Extreme Z-score (3.8) combined with quantile violation (actual > P99) and 5.2x volatility spike. Pattern matches historical case #127 (sensor failure).",
      "evidence_cited": ["z_score", "quantile_violation", "volatility_spike", "historical_pattern"],
      "anomaly_type": "point"
    }
  ],
  "overall_assessment": "High confidence anomaly detected due to convergent statistical signals and historical pattern match.",
  "uncertainty_factors": ["Limited historical context for this specific pattern"],
  "confidence_summary": {
    "mean": 0.92,
    "min": 0.92,
    "max": 0.92
  }
}
```

### Conversion to Binary Labels

Convert LLM output to binary labels for evaluation:

```python
def convert_to_binary_labels(llm_output, time_series_length):
    """Convert LLM anomaly ranges to binary labels."""

    labels = np.zeros(time_series_length, dtype=int)

    for anom in llm_output['anomalies']:
        start = anom.get('start_index', 0)
        end = anom.get('end_index', time_series_length)

        # Apply confidence threshold
        if anom['confidence'] >= 0.5:
            labels[start:end+1] = 1

    return labels
```

## Error Handling

### Common Issues and Solutions

1. **JSON Parsing Failure**
   - Solution: Use `response_format={"type": "json_object"}` (OpenAI)
   - Fallback: Regex extraction or rule-based parsing

2. **Missing Evidence Citations**
   - Solution: Add explicit instruction "You must cite specific evidence"
   - Validation: Check that `evidence_cited` list is non-empty

3. **Hallucinated Timestamps**
   - Solution: Provide clear index ranges in prompt
   - Validation: Ensure indices are within bounds

4. **Overconfident Predictions**
   - Solution: Emphasize uncertainty in system prompt
   - Calibration: Apply post-hoc confidence calibration

5. **Rate Limits**
   - Solution: Implement exponential backoff and retry logic
   - Alternative: Batch multiple windows into single prompt

### Retry Logic

```python
import time

def generate_with_retry(backend, system_prompt, user_prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return backend.generate(system_prompt, user_prompt)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retry {attempt + 1}/{max_retries} after {wait_time}s due to: {e}")
                time.sleep(wait_time)
            else:
                raise
```

## Evaluation

### Qualitative Evaluation

**Human review checklist**:
- [ ] Reasoning is coherent and logical
- [ ] Evidence is correctly cited
- [ ] Confidence scores are calibrated
- [ ] Explanations are helpful for understanding

**Example evaluation**:
```python
def evaluate_explanation_quality(llm_output, human_labels):
    """Manually score explanation quality."""

    scores = {
        'coherence': 0,      # 1-5: Is reasoning logical?
        'evidence_use': 0,   # 1-5: Does it cite evidence correctly?
        'calibration': 0,    # 1-5: Is confidence aligned with correctness?
        'helpfulness': 0     # 1-5: Is explanation useful?
    }

    # Human annotator fills in scores
    return scores
```

### Quantitative Evaluation

**Agreement with statistical baseline**:
```python
def evaluate_llm_vs_statistical(llm_labels, statistical_labels):
    """Compare LLM decisions to statistical threshold baseline."""

    agreement = np.mean(llm_labels == statistical_labels)
    kappa = cohen_kappa_score(llm_labels, statistical_labels)

    return {'agreement': agreement, 'kappa': kappa}
```

**Ablation studies**:
- LLM without RAG
- LLM without evidence (pure time series)
- LLM with partial evidence (e.g., forecast only)

## Cost Optimization

### Strategies

1. **Model Selection**: Use cheaper models (Gemini Flash) for prototyping
2. **Batching**: Combine multiple windows into one prompt (if context allows)
3. **Caching**: Cache responses for identical inputs
4. **Selective Usage**: Only use LLM for high-uncertainty cases (statistical baseline for clear cases)

### Cost Estimation

```python
def estimate_cost(num_windows, model='gpt-4-turbo'):
    """Estimate total cost for analyzing dataset."""

    # Token counts (approximate)
    tokens_per_window = 1500  # Prompt + completion
    total_tokens = num_windows * tokens_per_window

    # Pricing (per million tokens)
    pricing = {
        'gpt-4-turbo': 10.0,  # Input: $10, Output: $30 (average)
        'gemini-1.5-pro': 3.5,
        'gemini-2.0-flash': 0.15,
        'claude-3-opus': 15.0
    }

    cost = (total_tokens / 1_000_000) * pricing.get(model, 10.0)
    return {'total_tokens': total_tokens, 'estimated_cost_usd': cost}
```

## Integration with RAG System

See `spec/rag_system_design.md` for details on historical pattern retrieval.

**Summary**:
- RAG retrieves similar patterns from vector database
- Retrieved cases include: (time series snippet, evidence, label, explanation)
- LLM uses these as few-shot examples to improve reasoning

**Example RAG context injection**:
```
## Historical Context (RAG)

Similar patterns detected in the past:

### Case #127 (2023-11-15): Sensor Failure
- Pattern: Sharp spike followed by return to baseline
- Evidence: Z-score=3.9, Volatility spike=5.1x
- Label: Anomaly (confirmed)
- Explanation: Hardware sensor malfunction caused brief erroneous readings

### Case #203 (2024-01-03): Network Outage
- Pattern: Sudden drop to zero, then recovery
- Evidence: Z-score=-4.2, Trend break detected
- Label: Anomaly (confirmed)
- Explanation: Network connectivity loss resulted in missing data reported as zeros
```

## Module Structure

```
src/llm_reasoning/
├── __init__.py
├── llm_agent.py              # Main LLMAnomalyAgent class
├── backends.py               # LLM backend wrappers (OpenAI, Gemini, Claude)
├── prompt_builder.py         # Prompt formatting functions
├── output_parser.py          # JSON parsing and validation
└── evaluation.py             # Explanation quality metrics
```

## Configuration

```yaml
llm_reasoning:
  backend: "gemini"           # openai, gemini, claude
  model: "gemini-1.5-pro"     # Model name
  temperature: 0.0            # Deterministic output
  max_retries: 3              # Retry on failure
  timeout: 30                 # Seconds

  prompt_config:
    include_rag: true         # Use historical context
    max_rag_examples: 3       # Top-k similar cases
    evidence_format: "detailed"  # or "compact"

  output_config:
    confidence_threshold: 0.5  # Min confidence to label as anomaly
    require_evidence_citation: true
    validate_json: true

  cost_optimization:
    use_cache: true
    batch_windows: false      # Combine multiple windows (experimental)
```

## Next Steps

1. Implement `LLMAnomalyAgent` class
2. Test prompt engineering on sample data
3. Evaluate explanation quality (human review)
4. Tune confidence thresholds
5. Integrate with RAG system
6. Compare to statistical baseline

---

**Status**: Specification complete, ready for implementation
**Last Updated**: 2026-02-17
**Dependencies**: OpenAI SDK, Google AI SDK, Anthropic SDK, RAG system
