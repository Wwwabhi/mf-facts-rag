"""Report and visualization generator for RAG evaluation.

Generates:
  1. evaluation_results.csv: Per-question evaluation log with all details
  2. evaluation_summary.json: Aggregated metrics, category breakdown, latency percentiles,
     cost summary, and detailed failure reports
  3. evaluation_visualizations.png: High-resolution diagnostic charts
  4. Google Colab download helper
"""

from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.evaluator import EvaluatedSampleResult


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])


def _stats_for_list(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "p90": None, "p95": None}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "p90": round(_percentile(values, 90.0) or 0.0, 4),
        "p95": round(_percentile(values, 95.0) or 0.0, 4),
    }


def generate_reports(
    results: list[EvaluatedSampleResult],
    output_dir: str | Path,
    model_name: str,
    judge_name: str,
    dataset_name: str = "test_dataset.json",
) -> tuple[Path, Path, Path]:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "evaluation_results.csv"
    summary_path = out_dir / "evaluation_summary.json"
    plot_path = out_dir / "evaluation_visualizations.png"

    # 1. Write evaluation_results.csv
    if results:
        fieldnames = list(results[0].to_csv_dict().keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_csv_dict())

    # 2. Compute aggregate metrics
    relevancy_vals = [r.answer_relevancy_score for r in results if r.answer_relevancy_score is not None]
    faithfulness_vals = [r.faithfulness_score for r in results if r.faithfulness_score is not None]
    recall_vals = [r.contextual_recall_score for r in results if r.contextual_recall_score is not None]

    retrieval_lats = [r.retrieval_latency_ms for r in results if r.retrieval_latency_ms is not None]
    generation_lats = [r.generation_latency_ms for r in results if r.generation_latency_ms is not None]
    total_lats = [r.total_latency_ms for r in results]

    prompt_toks = sum(r.prompt_tokens or 0 for r in results)
    comp_toks = sum(r.completion_tokens or 0 for r in results)
    total_toks = sum(r.total_tokens or 0 for r in results)
    costs = [r.cost_usd for r in results if r.cost_usd is not None]
    total_cost = sum(costs) if costs else None

    # Category breakdown
    categories: set[str] = {r.category for r in results}
    category_summary: dict[str, Any] = {}
    for cat in sorted(categories):
        cat_items = [r for r in results if r.category == cat]
        cat_rel = [r.answer_relevancy_score for r in cat_items if r.answer_relevancy_score is not None]
        cat_faith = [r.faithfulness_score for r in cat_items if r.faithfulness_score is not None]
        cat_recall = [r.contextual_recall_score for r in cat_items if r.contextual_recall_score is not None]
        cat_lat = [r.total_latency_ms for r in cat_items]

        category_summary[cat] = {
            "total_questions": len(cat_items),
            "relevancy": _stats_for_list(cat_rel),
            "faithfulness": _stats_for_list(cat_faith),
            "contextual_recall": _stats_for_list(cat_recall),
            "total_latency_ms": _stats_for_list(cat_lat),
        }

    # Identify failures and low scores
    failed_items = [
        {
            "id": r.sample_id,
            "category": r.category,
            "question": r.question,
            "ground_truth": r.ground_truth,
            "actual_output": r.actual_output,
            "status": r.status,
            "error_message": r.error_message,
            "retrieved_context": r.retrieval_context,
            "reasons": {
                "relevancy": r.answer_relevancy_reason,
                "faithfulness": r.faithfulness_reason,
                "recall": r.contextual_recall_reason,
            },
        }
        for r in results
        if r.status != "SUCCESS"
        or (r.answer_relevancy_score is not None and r.answer_relevancy_score < 0.5)
        or (r.faithfulness_score is not None and r.faithfulness_score < 0.5)
    ]

    summary_data = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset_name,
            "total_samples": len(results),
            "successful_executions": sum(1 for r in results if r.status == "SUCCESS"),
            "failed_executions": sum(1 for r in results if r.status != "SUCCESS"),
            "target_model": model_name,
            "judge_model": judge_name,
        },
        "quality_metrics": {
            "answer_relevancy": _stats_for_list(relevancy_vals),
            "faithfulness": {
                **_stats_for_list(faithfulness_vals),
                "unanswerable_na_count": sum(1 for r in results if r.faithfulness_score is None),
            },
            "contextual_recall": {
                **_stats_for_list(recall_vals),
                "unanswerable_na_count": sum(1 for r in results if r.contextual_recall_score is None),
            },
        },
        "latency_metrics_ms": {
            "retrieval": _stats_for_list(retrieval_lats),
            "generation": _stats_for_list(generation_lats),
            "total": _stats_for_list(total_lats),
        },
        "token_usage_and_costs": {
            "total_prompt_tokens": prompt_toks,
            "total_completion_tokens": comp_toks,
            "total_tokens": total_toks,
            "estimated_cost_usd": round(total_cost, 6) if total_cost is not None else None,
            "cost_calculated": total_cost is not None,
        },
        "category_breakdown": category_summary,
        "failed_or_low_scoring_questions": failed_items,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 3. Generate Visualizations using matplotlib
    try:
        _render_visualizations(results, category_summary, summary_data, plot_path)
    except Exception as exc:
        print(f"Warning: Failed to render plots: {exc}")

    return csv_path, summary_path, plot_path


def _render_visualizations(
    results: list[EvaluatedSampleResult],
    category_summary: dict[str, Any],
    summary_data: dict[str, Any],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive headless backend
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#0f172a")

    for row in axes:
        for ax in row:
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#cbd5e1")
            ax.xaxis.label.set_color("#cbd5e1")
            ax.yaxis.label.set_color("#cbd5e1")
            ax.title.set_color("#f8fafc")
            for spine in ax.spines.values():
                spine.set_color("#334155")

    # Plot 1: Mean Quality Scores by Category
    ax1 = axes[0, 0]
    cats = list(category_summary.keys())
    if cats:
        x = np.arange(len(cats))
        width = 0.25
        rel_means = [category_summary[c]["relevancy"]["mean"] or 0.0 for c in cats]
        faith_means = [category_summary[c]["faithfulness"]["mean"] or 0.0 for c in cats]
        rec_means = [category_summary[c]["contextual_recall"]["mean"] or 0.0 for c in cats]

        ax1.bar(x - width, rel_means, width, label="Answer Relevancy", color="#38bdf8")
        ax1.bar(x, faith_means, width, label="Faithfulness", color="#34d399")
        ax1.bar(x + width, rec_means, width, label="Contextual Recall", color="#a78bfa")
        ax1.set_ylabel("Score (0.0 to 1.0)")
        ax1.set_title("Average Quality Scores by Question Category")
        ax1.set_xticks(x)
        ax1.set_xticklabels(cats, rotation=25, ha="right")
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis="y", linestyle="--", alpha=0.3, color="#64748b")
        ax1.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f8fafc")

    # Plot 2: Latency Breakdown (Retrieval vs Generation vs Total)
    ax2 = axes[0, 1]
    lats_to_plot = []
    lat_labels = []
    ret_vals = [r.retrieval_latency_ms for r in results if r.retrieval_latency_ms is not None]
    gen_vals = [r.generation_latency_ms for r in results if r.generation_latency_ms is not None]
    tot_vals = [r.total_latency_ms for r in results]

    if ret_vals:
        lats_to_plot.append(ret_vals)
        lat_labels.append("Retrieval")
    if gen_vals:
        lats_to_plot.append(gen_vals)
        lat_labels.append("Generation")
    if tot_vals:
        lats_to_plot.append(tot_vals)
        lat_labels.append("Total")

    if lats_to_plot:
        box = ax2.boxplot(lats_to_plot, patch_artist=True, tick_labels=lat_labels)
        colors = ["#38bdf8", "#f472b6", "#fbbf24"]
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in box["medians"]:
            median.set_color("#ffffff")
            median.set_linewidth(1.5)
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title("Latency Distribution (Retrieval vs Generation vs Total)")
        ax2.grid(axis="y", linestyle="--", alpha=0.3, color="#64748b")

    # Plot 3: Answer Relevancy vs Total Latency
    ax3 = axes[1, 0]
    cat_colors = {
        "factual": "#38bdf8",
        "multi_hop": "#a78bfa",
        "procedural": "#34d399",
        "unanswerable": "#fb923c",
        "cannot_verify": "#f43f5e",
        "investment_advice": "#eab308",
        "performance_returns": "#06b6d4",
    }
    for r in results:
        if r.answer_relevancy_score is not None:
            c = cat_colors.get(r.category.lower(), "#94a3b8")
            ax3.scatter(r.total_latency_ms, r.answer_relevancy_score, color=c, s=50, alpha=0.8, edgecolors="none")
    ax3.set_xlabel("Total Latency (ms)")
    ax3.set_ylabel("Answer Relevancy Score")
    ax3.set_title("Answer Relevancy vs. Total Latency")
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, linestyle="--", alpha=0.3, color="#64748b")

    # Plot 4: Metric Distributions & Cost Overview
    ax4 = axes[1, 1]
    tot_cost = summary_data["token_usage_and_costs"]["estimated_cost_usd"]
    tot_toks = summary_data["token_usage_and_costs"]["total_tokens"]
    succ = summary_data["metadata"]["successful_executions"]
    failed = summary_data["metadata"]["failed_executions"]

    cost_str = f"${tot_cost:.5f}" if tot_cost is not None else "N/A (pricing unconfigured)"

    summary_text = (
        f"EVALUATION SUMMARY OVERVIEW\n\n"
        f"• Target Model: {summary_data['metadata']['target_model']}\n"
        f"• Judge Engine: {summary_data['metadata']['judge_model']}\n"
        f"• Total Evaluated: {len(results)}\n"
        f"• Status: {succ} passed / {failed} failed\n\n"
        f"Aggregate Scores (Mean):\n"
        f"  - Relevancy: {summary_data['quality_metrics']['answer_relevancy']['mean'] or 'N/A'}\n"
        f"  - Faithfulness: {summary_data['quality_metrics']['faithfulness']['mean'] or 'N/A'}\n"
        f"  - Recall: {summary_data['quality_metrics']['contextual_recall']['mean'] or 'N/A'}\n\n"
        f"Performance & Cost:\n"
        f"  - Mean Total Latency: {summary_data['latency_metrics_ms']['total']['mean']} ms\n"
        f"  - Total Tokens: {tot_toks:,}\n"
        f"  - Estimated Run Cost: {cost_str}\n"
    )
    ax4.text(
        0.05,
        0.95,
        summary_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        color="#f8fafc",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=1", facecolor="#0f172a", edgecolor="#38bdf8", alpha=0.8),
    )
    ax4.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()


def print_cli_summary(summary_data: dict[str, Any]) -> None:
    """Prints a beautiful formatted summary to the console."""
    meta = summary_data["metadata"]
    qm = summary_data["quality_metrics"]
    lat = summary_data["latency_metrics_ms"]
    cost = summary_data["token_usage_and_costs"]

    print("\n" + "=" * 70)
    print(f"📊 RAG EVALUATION PIPELINE REPORT — {meta['timestamp']}")
    print("=" * 70)
    print(f"• Dataset:             {meta['dataset']} ({meta['total_samples']} samples)")
    print(f"• Target RAG Model:    {meta['target_model']}")
    print(f"• Evaluation Judge:    {meta['judge_model']}")
    print(f"• Executions:          {meta['successful_executions']} successful, {meta['failed_executions']} failed")

    print("\n📈 QUALITY METRICS (DeepEval):")
    print(f"  • Answer Relevancy:  Mean={qm['answer_relevancy']['mean']} | Median={qm['answer_relevancy']['median']} | Range=[{qm['answer_relevancy']['min']}, {qm['answer_relevancy']['max']}] (n={qm['answer_relevancy']['count']})")
    print(f"  • Faithfulness:      Mean={qm['faithfulness']['mean']} | Median={qm['faithfulness']['median']} | Range=[{qm['faithfulness']['min']}, {qm['faithfulness']['max']}] (n={qm['faithfulness']['count']}, N/A={qm['faithfulness']['unanswerable_na_count']})")
    print(f"  • Contextual Recall: Mean={qm['contextual_recall']['mean']} | Median={qm['contextual_recall']['median']} | Range=[{qm['contextual_recall']['min']}, {qm['contextual_recall']['max']}] (n={qm['contextual_recall']['count']}, N/A={qm['contextual_recall']['unanswerable_na_count']})")

    print("\n⚡ LATENCY BREAKDOWN (ms):")
    print(f"  • Retrieval:         Mean={lat['retrieval']['mean']} ms | p50={lat['retrieval']['median']} ms | p95={lat['retrieval']['p95']} ms")
    print(f"  • Generation:        Mean={lat['generation']['mean']} ms | p50={lat['generation']['median']} ms | p95={lat['generation']['p95']} ms")
    print(f"  • Total End-to-End:  Mean={lat['total']['mean']} ms | p50={lat['total']['median']} ms | p95={lat['total']['p95']} ms")

    cost_display = f"${cost['estimated_cost_usd']:.5f}" if cost["estimated_cost_usd"] is not None else "N/A (pricing not explicitly configured)"
    print("\n💰 TOKEN USAGE & ESTIMATED COST:")
    print(f"  • Prompt Tokens:     {cost['total_prompt_tokens']:,}")
    print(f"  • Completion Tokens: {cost['total_completion_tokens']:,}")
    print(f"  • Total Tokens:      {cost['total_tokens']:,}")
    print(f"  • Estimated Cost:    {cost_display}")

    failed = summary_data.get("failed_or_low_scoring_questions", [])
    if failed:
        print(f"\n⚠️  FLAGGED / LOW-SCORING SAMPLES ({len(failed)}):")
        for f in failed[:5]:
            print(f"  [{f['id']} - {f['category']}] Q: {f['question'][:60]}...")
            print(f"     Expected: {f['ground_truth'][:60]}...")
            print(f"     Actual:   {f['actual_output'][:60]}...")
    print("=" * 70 + "\n")
