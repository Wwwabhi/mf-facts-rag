"""Self-contained evaluation script for Google Colab and local execution.

Run locally:
    python -m evaluation.run_eval --dataset test_dataset.json

Run in Google Colab:
    !python evaluation/run_eval.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path so imports work seamlessly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# Colab Secrets support: read from google.colab.userdata if available
def get_secret(key: str, default: str = "") -> str:
    """Retrieve secret from Google Colab userdata if present, else environment."""
    try:
        from google.colab import userdata

        val = userdata.get(key)
        if val:
            return str(val).strip()
    except Exception:
        pass
    return os.getenv(key, default).strip()


# ==============================================================================
# ⚙️ CONFIGURATION SECTION (Colab & Local Configuration)
# Modify the parameters below or provide them via environment variables / flags.
# ==============================================================================

# 1. Pipeline Execution Mode:
#    - "direct_pipeline": Executes your exact Python RAG pipeline in-process
#    - "http_api": Sends requests to an external live HTTP REST API endpoint
DEFAULT_EVALUATION_MODE = os.getenv("EVALUATION_MODE", "direct_pipeline")

# 2. HTTP API Endpoint Settings (Used ONLY if EVALUATION_MODE == "http_api")
DEFAULT_API_ENDPOINT_URL = os.getenv("API_ENDPOINT_URL", "http://localhost:8000/api/chat")
DEFAULT_API_AUTH_TOKEN = get_secret("API_AUTH_TOKEN", "")

# Request schema: maps question string to request payload dict
DEFAULT_REQUEST_MAPPING = lambda question: {"question": question}

# Response schema: maps internal keys to fields/JSON paths in the API response JSON
DEFAULT_RESPONSE_MAPPING = {
    "answer": "answer",
    "retrieval_context": "retrieval_context",  # list of chunk strings or text
    "source_url": "source_url",
    "source_date": "source_date",
}

# 3. Model Configuration
DEFAULT_GROQ_API_KEY = get_secret("GROQ_API_KEY", "")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

# 4. Evaluation Judge Configuration
#    Supported providers: "groq" or "openai"
DEFAULT_JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "groq")
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", DEFAULT_GROQ_MODEL)
DEFAULT_OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")

# Pacing delay between judge calls to respect free-tier API rate limits
DEFAULT_JUDGE_DELAY = float(os.getenv("JUDGE_DELAY", "0.5"))

# 5. Dataset and Output Locations
DEFAULT_DATASET_PATH = os.getenv("DATASET_PATH", "test_dataset.json")
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "evaluation/outputs")

# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeatable RAG Evaluation Pipeline")
    parser.add_argument(
        "--mode",
        choices=["direct_pipeline", "http_api"],
        default=DEFAULT_EVALUATION_MODE,
        help="Execution mode (direct_pipeline or http_api)",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_PATH,
        help="Path to test_dataset.json",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save CSV, JSON summary, and PNG visualizations",
    )
    parser.add_argument(
        "--judge-provider",
        choices=["groq", "openai", "none"],
        default=DEFAULT_JUDGE_PROVIDER,
        help="LLM provider for DeepEval judge (groq, openai, or none for heuristics)",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Model name for evaluation judge",
    )
    parser.add_argument(
        "--groq-model",
        default=DEFAULT_GROQ_MODEL,
        help="Model name for target chatbot generation",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_ENDPOINT_URL,
        help="URL of live HTTP API (when mode=http_api)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_JUDGE_DELAY,
        help="Delay in seconds between judge requests to avoid rate limits",
    )
    return parser.parse_args()


def run_evaluation(
    mode: str = DEFAULT_EVALUATION_MODE,
    dataset_path: str = DEFAULT_DATASET_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    judge_provider: str = DEFAULT_JUDGE_PROVIDER,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    groq_model: str = DEFAULT_GROQ_MODEL,
    groq_api_key: str = DEFAULT_GROQ_API_KEY,
    openai_api_key: str = DEFAULT_OPENAI_API_KEY,
    api_url: str = DEFAULT_API_ENDPOINT_URL,
    api_auth_token: str = DEFAULT_API_AUTH_TOKEN,
    limit: int | None = None,
    judge_delay: float = DEFAULT_JUDGE_DELAY,
) -> None:
    from evaluation.dataset_validator import load_and_validate_dataset
    from evaluation.deep_eval_judge import GroqJudge, OpenAIJudge
    from evaluation.evaluator import EvaluatedSampleResult, RAGEvaluator
    from evaluation.pipeline_runner import DirectPipelineRunner, HttpApiRunner
    from evaluation.reporting import generate_reports, print_cli_summary

    print("=" * 70)
    print("🚀 INITIALIZING MUTUAL FUND RAG EVALUATION PIPELINE")
    print("=" * 70)
    print(f"• Execution Mode:     {mode}")
    print(f"• Dataset Source:     {dataset_path}")
    print(f"• Output Directory:   {output_dir}")
    print(f"• Target Chatbot LLM: {groq_model}")
    print(f"• Judge Provider:     {judge_provider} ({judge_model})")
    print("=" * 70)

    # 1. Dataset Loading and Validation
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        # Fallback check inside evaluation/
        alt_path = PROJECT_ROOT / "evaluation" / "test_dataset.json"
        if alt_path.exists():
            dataset_file = alt_path

    validation_result = load_and_validate_dataset(dataset_file)
    print(f"✓ Loaded {validation_result.total_loaded} records from {dataset_file.name}")
    print(f"✓ Valid samples: {len(validation_result.valid_samples)}")

    if validation_result.malformed_entries:
        print(f"⚠️  Found {len(validation_result.malformed_entries)} malformed entries:")
        for issue in validation_result.malformed_entries:
            print(f"   - Index {issue.index}: {'; '.join(issue.errors)}")
        print("   Malformed entries will be skipped from evaluation.")

    samples_to_run = validation_result.valid_samples
    if limit is not None and limit > 0:
        samples_to_run = samples_to_run[:limit]
        print(f"ℹ️  Limiting evaluation run to {len(samples_to_run)} samples.")

    # 2. Pipeline Runner Setup
    if mode == "direct_pipeline":
        # Ensure Groq API Key is present in environment
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        os.environ["GROQ_MODEL"] = groq_model
        runner = DirectPipelineRunner(top_k=5, model_name=groq_model)
    elif mode == "http_api":
        headers = {"Content-Type": "application/json"}
        if api_auth_token:
            headers["Authorization"] = f"Bearer {api_auth_token}"
        runner = HttpApiRunner(
            endpoint_url=api_url,
            headers=headers,
            request_mapping=DEFAULT_REQUEST_MAPPING,
            response_mapping=DEFAULT_RESPONSE_MAPPING,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 3. DeepEval Judge Setup
    judge_instance = None
    if judge_provider == "groq":
        active_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        if not active_key:
            print("⚠️  Warning: GROQ_API_KEY is not set. DeepEval judge will use heuristic fallback.")
        else:
            judge_instance = GroqJudge(
                model_name=judge_model,
                api_key=active_key,
                rate_limit_delay_seconds=judge_delay,
            )
    elif judge_provider == "openai":
        active_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if not active_key:
            print("⚠️  Warning: OPENAI_API_KEY is not set. DeepEval judge will use heuristic fallback.")
        else:
            judge_instance = OpenAIJudge(
                model_name=judge_model or "gpt-4o-mini",
                api_key=active_key,
            )

    evaluator = RAGEvaluator(judge_model=judge_instance)

    # 4. Execute and Evaluate
    evaluated_results: list[EvaluatedSampleResult] = []
    total = len(samples_to_run)

    print("\n🔍 Running Evaluation Test Cases...")
    for idx, sample in enumerate(samples_to_run, start=1):
        print(f"[{idx:02d}/{total:02d}] {sample.id} ({sample.category}): {sample.question[:55]}...", end="", flush=True)

        # Run question through chatbot
        exec_result = runner.run_sample(
            sample_id=sample.id,
            question=sample.question,
            ground_truth=sample.ground_truth,
            category=sample.category,
        )

        # Evaluate metrics
        eval_result = evaluator.evaluate_sample(sample, exec_result)
        evaluated_results.append(eval_result)

        # Quick inline feedback
        rel = f"Rel={eval_result.answer_relevancy_score:.2f}" if eval_result.answer_relevancy_score is not None else "Rel=N/A"
        faith = f"Faith={eval_result.faithfulness_score:.2f}" if eval_result.faithfulness_score is not None else "Faith=N/A"
        lat = f"{eval_result.total_latency_ms:.0f}ms"
        print(f" -> {exec_result.status} ({rel}, {faith}, {lat})")

    # 5. Generate Outputs
    judge_display = f"{judge_provider} ({judge_model})" if judge_instance else "heuristic_fallback"
    csv_file, json_file, plot_file = generate_reports(
        results=evaluated_results,
        output_dir=output_dir,
        model_name=groq_model,
        judge_name=judge_display,
        dataset_name=dataset_file.name,
    )

    # 6. Read Summary and Print CLI Overview
    import json

    with open(json_file, "r", encoding="utf-8") as f:
        summary_dict = json.load(f)

    print_cli_summary(summary_dict)

    print("📁 Generated Artifacts:")
    print(f"  • Detailed CSV:      {csv_file}")
    print(f"  • Aggregate JSON:    {json_file}")
    print(f"  • Visualizations:    {plot_file}")

    # 7. Google Colab Download Helper
    if "google.colab" in sys.modules:
        print("\n⬇️  Google Colab environment detected!")
        print("To download the outputs, run:")
        print("  from google.colab import files")
        print(f"  files.download('{csv_file}')")
        print(f"  files.download('{json_file}')")
        print(f"  files.download('{plot_file}')")


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        mode=args.mode,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        groq_model=args.groq_model,
        api_url=args.api_url,
        limit=args.limit,
        judge_delay=args.delay,
    )
