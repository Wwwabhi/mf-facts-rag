"""Single entry point for running the repeatable RAG evaluation workflow.

Run locally:
    python -m evaluation.run_all

Run with custom options:
    python -m evaluation.run_all --limit 5
    python -m evaluation.run_all --judge-provider openai --judge-model gpt-4o-mini
    python -m evaluation.run_all --mode http_api --api-url http://localhost:8000/api/chat

Outputs:
    Saves report.html, evaluation_results.csv, evaluation_summary.json, and
    evaluation_visualizations.png into a timestamped directory (e.g. evaluation/outputs/run_YYYYMMDD_HHMMSS/).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Silence unnecessary hub and tokenizer warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Load environment variables from .env
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")


def get_secret(key: str, default: str = "") -> str:
    """Retrieve secret from Google Colab userdata if available, else environment."""
    try:
        from google.colab import userdata

        val = userdata.get(key)
        if val:
            return str(val).strip()
    except Exception:
        pass
    return os.getenv(key, default).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single Entry Point for Repeatable RAG Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=os.getenv("DATASET_PATH", "test_dataset.json"),
        help="Path to evaluation test dataset JSON file",
    )
    parser.add_argument(
        "--output-base-dir",
        default=os.getenv("OUTPUT_DIR", "evaluation/outputs"),
        help="Base directory for saving evaluation runs",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Custom run directory name (default: run_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--mode",
        choices=["direct_pipeline", "http_api"],
        default=os.getenv("EVALUATION_MODE", "direct_pipeline"),
        help="Evaluation execution mode",
    )
    parser.add_argument(
        "--judge-provider",
        choices=["groq", "openai", "none"],
        default=os.getenv("JUDGE_PROVIDER", "groq"),
        help="LLM provider for evaluation judge (groq, openai, or none for heuristics)",
    )
    parser.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL", os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")),
        help="Model identifier for the evaluation judge",
    )
    parser.add_argument(
        "--groq-model",
        default=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
        help="Target RAG chatbot LLM model",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("API_ENDPOINT_URL", "http://localhost:8000/api/chat"),
        help="URL of live HTTP API (used when mode=http_api)",
    )
    parser.add_argument(
        "--api-token",
        default=get_secret("API_AUTH_TOKEN", ""),
        help="Authorization token for HTTP API (used when mode=http_api)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate (useful for smoke tests)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=float(os.getenv("JUDGE_DELAY", "0.5")),
        help="Delay in seconds between judge requests to respect API rate limits",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question in direct pipeline mode",
    )
    return parser.parse_args()


def run_all(
    dataset_path: str = "test_dataset.json",
    output_base_dir: str = "evaluation/outputs",
    run_name: str | None = None,
    mode: str = "direct_pipeline",
    judge_provider: str = "groq",
    judge_model: str = "qwen/qwen3.8-27b",
    groq_model: str = "qwen/qwen3.8-27b",
    groq_api_key: str | None = None,
    openai_api_key: str | None = None,
    api_url: str = "http://localhost:8000/api/chat",
    api_token: str = "",
    limit: int | None = None,
    delay: float = 0.5,
    top_k: int = 5,
) -> dict[str, Any]:
    """Runs the complete RAG evaluation workflow, saving detailed artifacts and human-readable HTML report."""
    from evaluation.dataset_validator import load_and_validate_dataset
    from evaluation.deep_eval_judge import GroqJudge, OpenAIJudge
    from evaluation.evaluator import EvaluatedSampleResult, RAGEvaluator
    from evaluation.pipeline_runner import DirectPipelineRunner, HttpApiRunner
    from evaluation.reporting import generate_reports, print_cli_summary

    # 1. Establish timestamped run directory
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = run_name or f"run_{timestamp_str}"
    out_dir = Path(output_base_dir).resolve() / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 76)
    print("🚀 MUTUAL FUND FACTS RAG — EVALUATION PIPELINE")
    print("=" * 76)
    print(f"• Execution Mode:       {mode}")
    print(f"• Target Chatbot LLM:   {groq_model}")
    print(f"• Evaluation Judge:     {judge_provider} ({judge_model})")
    print(f"• Rate-Limit Delay:     {delay:.1f}s")
    print(f"• Output Directory:     {out_dir}")
    print("=" * 76)

    # 2. Locate and Validate Dataset
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        alt_path = PROJECT_ROOT / "test_dataset.json"
        if alt_path.exists():
            dataset_file = alt_path
        else:
            alt_path2 = PROJECT_ROOT / "evaluation" / "test_dataset.json"
            if alt_path2.exists():
                dataset_file = alt_path2

    validation_result = load_and_validate_dataset(dataset_file)
    print(f"✓ Loaded {validation_result.total_loaded} samples from {dataset_file.name}")
    print(f"✓ Valid samples for evaluation: {len(validation_result.valid_samples)}")

    if validation_result.malformed_entries:
        print(f"⚠️  Found {len(validation_result.malformed_entries)} malformed entries (will be skipped):")
        for issue in validation_result.malformed_entries:
            print(f"   - Index {issue.index}: {'; '.join(issue.errors)}")

    samples = validation_result.valid_samples
    if limit is not None and limit > 0:
        samples = samples[:limit]
        print(f"ℹ️  Evaluation limited to first {len(samples)} samples.")

    # 3. Initialize Pipeline Runner
    active_groq_key = (groq_api_key or get_secret("GROQ_API_KEY", "")).strip()
    active_openai_key = (openai_api_key or get_secret("OPENAI_API_KEY", "")).strip()

    if active_groq_key:
        os.environ["GROQ_API_KEY"] = active_groq_key
    os.environ["GROQ_MODEL"] = groq_model

    if mode == "direct_pipeline":
        runner = DirectPipelineRunner(top_k=top_k, model_name=groq_model)
    elif mode == "http_api":
        headers = {"Content-Type": "application/json"}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        runner = HttpApiRunner(
            endpoint_url=api_url,
            headers=headers,
        )
    else:
        raise ValueError(f"Unsupported evaluation mode: '{mode}'")

    # 4. Initialize Evaluation Judge
    judge_instance: Any = None
    if judge_provider == "groq":
        if not active_groq_key:
            print("⚠️  Notice: GROQ_API_KEY not configured. Falling back to heuristic quality scoring.")
        else:
            try:
                judge_instance = GroqJudge(
                    model_name=judge_model,
                    api_key=active_groq_key,
                    rate_limit_delay_seconds=delay,
                )
            except Exception as exc:
                print(f"⚠️  Notice: Failed to initialize GroqJudge ({exc}). Using heuristic scoring.")
    elif judge_provider == "openai":
        if not active_openai_key:
            print("⚠️  Notice: OPENAI_API_KEY not configured. Falling back to heuristic quality scoring.")
        else:
            try:
                judge_instance = OpenAIJudge(
                    model_name=judge_model or "gpt-4o-mini",
                    api_key=active_openai_key,
                )
            except Exception as exc:
                print(f"⚠️  Notice: Failed to initialize OpenAIJudge ({exc}). Using heuristic scoring.")

    evaluator = RAGEvaluator(judge_model=judge_instance)

    # 5. Execute Evaluation Loop with Progress & Error Protection
    evaluated_results: list[EvaluatedSampleResult] = []
    total_count = len(samples)

    print(f"\n🔍 Executing Evaluation Test Cases (1 to {total_count})...\n")

    for idx, sample in enumerate(samples, start=1):
        q_snippet = sample.question[:50] + "..." if len(sample.question) > 50 else sample.question
        print(f"[{idx:02d}/{total_count:02d}] {sample.id} ({sample.category}): \"{q_snippet}\"", end="", flush=True)

        t_case_start = time.perf_counter()
        exec_result = None
        eval_result = None

        try:
            # Run query through RAG pipeline
            exec_result = runner.run_sample(
                sample_id=sample.id,
                question=sample.question,
                ground_truth=sample.ground_truth,
                category=sample.category,
            )

            # Evaluate metrics with DeepEval
            eval_result = evaluator.evaluate_sample(sample, exec_result)
            evaluated_results.append(eval_result)

            # Format concise status display
            rel_str = f"Rel={eval_result.answer_relevancy_score:.2f}" if eval_result.answer_relevancy_score is not None else "Rel=N/A"
            faith_str = f"Faith={eval_result.faithfulness_score:.2f}" if eval_result.faithfulness_score is not None else "Faith=N/A"
            rec_str = f"Rec={eval_result.contextual_recall_score:.2f}" if eval_result.contextual_recall_score is not None else "Rec=N/A"
            lat_str = f"{eval_result.total_latency_ms:.0f}ms"

            status_icon = "✓" if exec_result.status == "SUCCESS" else "⚠️"
            print(f" -> {status_icon} {exec_result.status} ({rel_str}, {faith_str}, {rec_str}, {lat_str})")

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t_case_start) * 1000.0
            print(f" -> ❌ ERROR: {exc} ({elapsed_ms:.0f}ms)")
            # Create graceful fallback error record so evaluation does not halt
            fallback_exec = exec_result or type("ExecResult", (), {
                "actual_output": "",
                "retrieval_context": [],
                "source_url": None,
                "source_date": None,
                "decision_allowed": False,
                "decision_category": "error",
                "retrieval_latency_ms": None,
                "generation_latency_ms": None,
                "total_latency_ms": elapsed_ms,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "cost_usd": None,
                "status": "ERROR",
                "error_message": str(exc),
                "retries": 0,
            })()
            fallback_eval = EvaluatedSampleResult(
                sample_id=sample.id,
                question=sample.question,
                category=sample.category,
                ground_truth=sample.ground_truth,
                actual_output="",
                retrieval_context=[],
                source_url=None,
                source_date=None,
                decision_allowed=False,
                decision_category="error",
                answer_relevancy_score=None,
                answer_relevancy_reason=f"Runtime evaluation exception: {exc}",
                faithfulness_score=None,
                faithfulness_reason=f"Runtime evaluation exception: {exc}",
                contextual_recall_score=None,
                contextual_recall_reason=f"Runtime evaluation exception: {exc}",
                retrieval_latency_ms=None,
                generation_latency_ms=None,
                total_latency_ms=elapsed_ms,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                cost_usd=None,
                status="ERROR",
                error_message=str(exc),
                retries=0,
            )
            evaluated_results.append(fallback_eval)

        # Inter-request delay between test cases to respect API RPM
        if delay > 0 and idx < total_count:
            time.sleep(delay)

    # 6. Generate Outputs (CSV, JSON, Visualizations, HTML Report)
    judge_display = f"{judge_provider} ({judge_model})" if judge_instance else "heuristic_fallback"
    csv_file, json_file, plot_file, html_file = generate_reports(
        results=evaluated_results,
        output_dir=out_dir,
        model_name=groq_model,
        judge_name=judge_display,
        dataset_name=dataset_file.name,
    )

    # Also maintain a convenience "latest" reference
    try:
        latest_file = Path(output_base_dir) / "latest_run.txt"
        with open(latest_file, "w", encoding="utf-8") as f:
            f.write(str(out_dir))
    except Exception:
        pass

    # 7. Print Terminal Summary
    import json

    with open(json_file, "r", encoding="utf-8") as f:
        summary_dict = json.load(f)

    print_cli_summary(summary_dict)

    # 8. Print Exact Artifact Paths
    rel_html = os.path.relpath(html_file, Path.cwd())
    print("📁 GENERATED ARTIFACTS IN TIMESTAMPED RUN DIRECTORY:")
    print(f"  • 📊 HTML Report:    file://{html_file}")
    print(f"                       ({rel_html})")
    print(f"  • 📁 Results CSV:    file://{csv_file}")
    print(f"  • 📄 Summary JSON:   file://{json_file}")
    print(f"  • 📈 Visualizations: file://{plot_file}")
    print("\n💡 Open the HTML report directly in your browser:")
    print(f"   open {rel_html}\n")

    return {
        "run_dir": out_dir,
        "html_report": html_file,
        "csv_results": csv_file,
        "summary_json": json_file,
        "visualizations": plot_file,
        "summary_data": summary_dict,
    }


def main() -> None:
    args = parse_args()
    run_all(
        dataset_path=args.dataset,
        output_base_dir=args.output_base_dir,
        run_name=args.run_name,
        mode=args.mode,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        groq_model=args.groq_model,
        api_url=args.api_url,
        api_token=args.api_token,
        limit=args.limit,
        delay=args.delay,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
