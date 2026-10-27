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
import html
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
) -> tuple[Path, Path, Path, Path]:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "evaluation_results.csv"
    summary_path = out_dir / "evaluation_summary.json"
    plot_path = out_dir / "evaluation_visualizations.png"
    html_path = out_dir / "report.html"

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

    # 4. Generate Human-Readable report.html
    try:
        generate_html_report(
            results=results,
            summary_data=summary_data,
            output_path=html_path,
            chart_filename=plot_path.name,
        )
    except Exception as exc:
        print(f"Warning: Failed to generate HTML report: {exc}")

    return csv_path, summary_path, plot_path, html_path


def _badge_html(score: float | None, is_score: bool = True) -> str:
    if score is None:
        return '<span class="badge badge-na">N/A</span>'
    val_str = f"{score:.2f}"
    if score >= 0.8:
        return f'<span class="badge badge-high">{val_str}</span>'
    if score >= 0.5:
        return f'<span class="badge badge-med">{val_str}</span>'
    return f'<span class="badge badge-low">{val_str}</span>'


def generate_html_report(
    results: list[EvaluatedSampleResult],
    summary_data: dict[str, Any],
    output_path: str | Path,
    chart_filename: str = "evaluation_visualizations.png",
) -> Path:
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    meta = summary_data["metadata"]
    qm = summary_data["quality_metrics"]
    lat = summary_data["latency_metrics_ms"]
    cost = summary_data["token_usage_and_costs"]
    breakdown = summary_data["category_breakdown"]
    flagged = summary_data.get("failed_or_low_scoring_questions", [])

    cost_display = f"${cost['estimated_cost_usd']:.5f}" if cost["estimated_cost_usd"] is not None else "N/A"
    run_status_badge = (
        '<span class="status-pill status-pass">ALL TESTS PASSED</span>'
        if meta["failed_executions"] == 0 and len(flagged) == 0
        else f'<span class="status-pill status-warn">{len(flagged)} FLAGGED SAMPLES</span>'
    )

    # Category breakdown rows
    cat_rows = []
    for cat, stats in breakdown.items():
        rel_mean = f"{stats['relevancy']['mean']:.2f}" if stats['relevancy']['mean'] is not None else "N/A"
        faith_mean = f"{stats['faithfulness']['mean']:.2f}" if stats['faithfulness']['mean'] is not None else "N/A"
        rec_mean = f"{stats['contextual_recall']['mean']:.2f}" if stats['contextual_recall']['mean'] is not None else "N/A"
        lat_mean = f"{stats['total_latency_ms']['mean']:.0f} ms" if stats['total_latency_ms']['mean'] is not None else "N/A"
        cat_rows.append(
            f"""<tr>
                <td><span class="cat-pill cat-{html.escape(cat.lower())}">{html.escape(cat)}</span></td>
                <td><strong>{stats['total_questions']}</strong></td>
                <td>{rel_mean}</td>
                <td>{faith_mean}</td>
                <td>{rec_mean}</td>
                <td>{lat_mean}</td>
            </tr>"""
        )
    cat_table_html = "\n".join(cat_rows)

    # Flagged section
    flagged_html = ""
    if flagged:
        flagged_cards = []
        for item in flagged:
            flagged_cards.append(
                f"""<div class="flagged-card">
                    <div class="flagged-header">
                        <span class="flagged-id">[{html.escape(item['id'])}]</span>
                        <span class="cat-pill cat-{html.escape(item['category'].lower())}">{html.escape(item['category'])}</span>
                        <span class="badge badge-low">{html.escape(item['status'])}</span>
                    </div>
                    <p class="flagged-q"><strong>Question:</strong> {html.escape(item['question'])}</p>
                    <p class="flagged-gt"><strong>Expected:</strong> {html.escape(item['ground_truth'])}</p>
                    <p class="flagged-act"><strong>Actual:</strong> {html.escape(item['actual_output'])}</p>
                    <div class="flagged-reasons">
                        <div><strong>Relevancy:</strong> {html.escape(str(item['reasons'].get('relevancy') or 'None'))}</div>
                        <div><strong>Faithfulness:</strong> {html.escape(str(item['reasons'].get('faithfulness') or 'None'))}</div>
                    </div>
                </div>"""
            )
        flagged_html = f"""<section class="section">
            <h2 class="section-title text-danger">⚠️ Flagged / Low-Scoring Questions ({len(flagged)})</h2>
            <div class="flagged-grid">{''.join(flagged_cards)}</div>
        </section>"""
    else:
        flagged_html = """<section class="section">
            <div class="all-clean-banner">
                ✓ Perfect Run: Zero execution failures and all scored samples met the >= 0.50 quality threshold!
            </div>
        </section>"""

    # Interactive test cases table rows
    table_rows = []
    unique_categories = sorted({r.category for r in results})
    for idx, r in enumerate(results, start=1):
        is_flagged = r.status != "SUCCESS" or (r.answer_relevancy_score is not None and r.answer_relevancy_score < 0.5) or (r.faithfulness_score is not None and r.faithfulness_score < 0.5)
        flagged_attr = "true" if is_flagged else "false"
        status_badge = (
            '<span class="badge badge-high">SUCCESS</span>'
            if r.status == "SUCCESS"
            else f'<span class="badge badge-low">{html.escape(r.status)}</span>'
        )
        row_id = f"row-{idx}"

        # Source URL link if valid
        source_link_html = "None"
        if r.source_url:
            source_link_html = f'<a href="{html.escape(r.source_url)}" target="_blank" rel="noopener noreferrer" class="link">{html.escape(r.source_url)}</a>'

        # Retrieved contexts list
        contexts_html = ""
        if r.retrieval_context:
            ctx_items = "".join(f"<li>{html.escape(c)}</li>" for c in r.retrieval_context)
            contexts_html = f"<ul class='context-list'>{ctx_items}</ul>"
        else:
            contexts_html = "<span class='text-muted'>No context retrieved (refusal or unanswerable before retrieval)</span>"

        table_rows.append(
            f"""<tr class="main-row" data-category="{html.escape(r.category.lower())}" data-flagged="{flagged_attr}" onclick="toggleDetails('{row_id}')">
                <td><strong>{html.escape(r.sample_id)}</strong></td>
                <td><span class="cat-pill cat-{html.escape(r.category.lower())}">{html.escape(r.category)}</span></td>
                <td class="col-q" title="{html.escape(r.question)}">{html.escape(r.question)}</td>
                <td>{_badge_html(r.answer_relevancy_score)}</td>
                <td>{_badge_html(r.faithfulness_score)}</td>
                <td>{_badge_html(r.contextual_recall_score)}</td>
                <td>{r.total_latency_ms:.0f} ms</td>
                <td>{status_badge}</td>
                <td><button type="button" class="btn-details" onclick="event.stopPropagation(); toggleDetails('{row_id}')">Details ▾</button></td>
            </tr>
            <tr id="{row_id}" class="drawer-row" style="display: none;">
                <td colspan="9">
                    <div class="drawer-content">
                        <div class="drawer-grid">
                            <div class="drawer-panel">
                                <h4>Full Model Output</h4>
                                <div class="box box-output">{html.escape(r.actual_output or '(Empty)')}</div>
                                <h4>Expected Ground Truth</h4>
                                <div class="box box-gt">{html.escape(r.ground_truth)}</div>
                                <h4>Source Provenance</h4>
                                <p><strong>Source URL:</strong> {source_link_html}</p>
                                <p><strong>Source Date:</strong> {html.escape(r.source_date or 'N/A')}</p>
                            </div>
                            <div class="drawer-panel">
                                <h4>Evaluation Judge Reasoning</h4>
                                <div class="metric-reason">
                                    <strong>Answer Relevancy:</strong> {html.escape(r.answer_relevancy_reason or 'N/A')}
                                </div>
                                <div class="metric-reason">
                                    <strong>Faithfulness:</strong> {html.escape(r.faithfulness_reason or 'N/A')}
                                </div>
                                <div class="metric-reason">
                                    <strong>Contextual Recall:</strong> {html.escape(r.contextual_recall_reason or 'N/A')}
                                </div>
                                <h4>Execution Diagnostics</h4>
                                <p><strong>Retrieval Latency:</strong> {f"{r.retrieval_latency_ms:.1f} ms" if r.retrieval_latency_ms is not None else 'N/A'}</p>
                                <p><strong>Generation Latency:</strong> {f"{r.generation_latency_ms:.1f} ms" if r.generation_latency_ms is not None else 'N/A'}</p>
                                <p><strong>Tokens:</strong> {r.prompt_tokens or 0} prompt + {r.completion_tokens or 0} completion = {r.total_tokens or 0} total</p>
                                <p><strong>Estimated Cost:</strong> {f"${r.cost_usd:.6f}" if r.cost_usd is not None else 'N/A'}</p>
                            </div>
                        </div>
                        <div class="drawer-contexts">
                            <h4>Retrieved Context Chunks ({len(r.retrieval_context)})</h4>
                            {contexts_html}
                        </div>
                    </div>
                </td>
            </tr>"""
        )
    all_table_rows_html = "\n".join(table_rows)

    # Filter buttons
    filter_buttons = ['<button class="filter-pill active" onclick="setCategoryFilter(\'all\', this)">All (' + str(len(results)) + ')</button>']
    for cat in unique_categories:
        count = sum(1 for r in results if r.category == cat)
        filter_buttons.append(f'<button class="filter-pill" onclick="setCategoryFilter(\'{html.escape(cat.lower())}\', this)">{html.escape(cat.title())} ({count})</button>')
    if flagged:
        filter_buttons.append(f'<button class="filter-pill filter-flagged" onclick="setCategoryFilter(\'flagged\', this)">Flagged ({len(flagged)})</button>')
    filter_buttons_html = "\n".join(filter_buttons)

    # Full HTML Document
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mutual Fund Facts RAG — Evaluation Report</title>
    <style>
        :root {{
            --bg-body: #0b0f19;
            --bg-card: #131b2e;
            --bg-panel: #1a243b;
            --border: #23304c;
            --border-hover: #3b82f6;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --cyan: #38bdf8;
            --emerald: #10b981;
            --purple: #a78bfa;
            --amber: #f59e0b;
            --rose: #ef4444;
            --blue: #3b82f6;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-body);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.5;
            padding: 24px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            background: linear-gradient(135deg, #131b2e 0%, #172554 100%);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
        }}
        .header-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
            margin-bottom: 16px;
        }}
        h1 {{
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(90deg, #f8fafc, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .status-pill {{
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 0.5px;
        }}
        .status-pass {{ background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }}
        .status-warn {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }}
        .meta-strip {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            color: var(--text-muted);
            font-size: 14px;
            border-top: 1px solid rgba(255,255,255,0.1);
            padding-top: 16px;
        }}
        .meta-item strong {{ color: #e2e8f0; }}
        
        /* KPI Cards Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            position: relative;
            overflow: hidden;
            transition: transform 0.2s, border-color 0.2s;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            border-color: #3b82f6;
        }}
        .kpi-label {{
            font-size: 13px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }}
        .kpi-value {{
            font-size: 30px;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 8px;
        }}
        .val-cyan {{ color: var(--cyan); }}
        .val-emerald {{ color: var(--emerald); }}
        .val-purple {{ color: var(--purple); }}
        .val-amber {{ color: var(--amber); }}
        .val-blue {{ color: var(--blue); }}
        .kpi-sub {{
            font-size: 12px;
            color: var(--text-muted);
        }}

        /* Section Layout */
        .section {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .section-title {{
            font-size: 19px;
            font-weight: 700;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .text-danger {{ color: #f87171; }}

        /* Category Table */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            color: var(--text-muted);
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            background: rgba(255, 255, 255, 0.02);
        }}
        tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}
        .cat-pill {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 600;
            text-transform: capitalize;
        }}
        .cat-factual {{ background: rgba(56, 189, 248, 0.15); color: #38bdf8; }}
        .cat-multi_hop {{ background: rgba(167, 139, 250, 0.15); color: #c084fc; }}
        .cat-procedural {{ background: rgba(52, 211, 153, 0.15); color: #34d399; }}
        .cat-unanswerable {{ background: rgba(251, 146, 60, 0.15); color: #fb923c; }}

        /* Badges */
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 700;
        }}
        .badge-high {{ background: rgba(16, 185, 129, 0.2); color: #34d399; }}
        .badge-med {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; }}
        .badge-low {{ background: rgba(239, 68, 68, 0.2); color: #f87171; }}
        .badge-na {{ background: rgba(100, 116, 139, 0.2); color: #94a3b8; }}

        /* Chart */
        .chart-container {{
            background: #0f172a;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 12px;
            text-align: center;
        }}
        .chart-img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }}

        /* Flagged */
        .flagged-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 16px;
        }}
        .flagged-card {{
            background: rgba(239, 68, 68, 0.05);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 10px;
            padding: 16px;
        }}
        .flagged-header {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }}
        .flagged-id {{ font-weight: 700; color: #fca5a5; }}
        .flagged-q, .flagged-gt, .flagged-act {{ font-size: 13px; margin-bottom: 6px; }}
        .flagged-reasons {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid rgba(255,255,255,0.08);
        }}
        .all-clean-banner {{
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 10px;
            padding: 16px;
            color: #34d399;
            font-weight: 600;
        }}

        /* Search & Filter Toolbar */
        .toolbar {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            margin-bottom: 16px;
        }}
        .search-box {{
            flex: 1;
            min-width: 250px;
            background: var(--bg-panel);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 8px 14px;
            color: var(--text-main);
            font-size: 14px;
        }}
        .search-box:focus {{ outline: none; border-color: var(--blue); }}
        .filter-group {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .filter-pill {{
            background: var(--bg-panel);
            border: 1px solid var(--border);
            color: var(--text-muted);
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .filter-pill:hover {{ background: var(--border); color: var(--text-main); }}
        .filter-pill.active {{ background: var(--blue); color: #ffffff; border-color: var(--blue); }}
        .filter-flagged.active {{ background: var(--rose); border-color: var(--rose); }}

        /* Interactive Table */
        .main-row {{ cursor: pointer; }}
        .col-q {{
            max-width: 320px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .btn-details {{
            background: var(--bg-panel);
            border: 1px solid var(--border);
            color: var(--cyan);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
        }}
        .btn-details:hover {{ background: #1e293b; }}
        .drawer-row td {{
            background: #0d1527;
            padding: 16px 20px;
            border-top: none;
        }}
        .drawer-content {{
            font-size: 13px;
        }}
        .drawer-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 16px;
        }}
        @media(max-width: 900px) {{
            .drawer-grid {{ grid-template-columns: 1fr; }}
        }}
        .box {{
            background: #090e1a;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px 12px;
            margin: 6px 0 12px;
            font-family: inherit;
        }}
        .box-output {{ color: #e2e8f0; border-left: 3px solid var(--cyan); }}
        .box-gt {{ color: var(--text-muted); border-left: 3px solid var(--purple); }}
        .metric-reason {{
            background: #090e1a;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px 10px;
            margin-bottom: 8px;
            font-size: 12px;
        }}
        .context-list {{
            list-style: decimal inside;
            max-height: 200px;
            overflow-y: auto;
            background: #090e1a;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 10px 14px;
            font-size: 12px;
            color: #cbd5e1;
        }}
        .context-list li {{ margin-bottom: 6px; }}
        .link {{ color: var(--cyan); text-decoration: none; word-break: break-all; }}
        .link:hover {{ text-decoration: underline; }}
        footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 13px;
            margin-top: 32px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div class="header-top">
                <div>
                    <h1>Mutual Fund Facts RAG</h1>
                    <p style="color: var(--text-muted); margin-top: 4px;">Automated Repeatable Evaluation & Quality Benchmark Report</p>
                </div>
                <div>{run_status_badge}</div>
            </div>
            <div class="meta-strip">
                <div class="meta-item"><strong>Date:</strong> {meta['timestamp']}</div>
                <div class="meta-item"><strong>Dataset:</strong> {meta['dataset']} ({meta['total_samples']} samples)</div>
                <div class="meta-item"><strong>Target Model:</strong> {meta['target_model']}</div>
                <div class="meta-item"><strong>Judge Engine:</strong> {meta['judge_model']}</div>
                <div class="meta-item"><strong>Executions:</strong> {meta['successful_executions']} OK / {meta['failed_executions']} Failed</div>
            </div>
        </header>

        <!-- KPI Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Answer Relevancy</div>
                <div class="kpi-value val-cyan">{qm['answer_relevancy']['mean'] or 'N/A'}</div>
                <div class="kpi-sub">Median: {qm['answer_relevancy']['median'] or 'N/A'} | Min: {qm['answer_relevancy']['min']} Max: {qm['answer_relevancy']['max']}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Faithfulness</div>
                <div class="kpi-value val-emerald">{qm['faithfulness']['mean'] or 'N/A'}</div>
                <div class="kpi-sub">Median: {qm['faithfulness']['median'] or 'N/A'} | N/A unanswerable: {qm['faithfulness'].get('unanswerable_na_count', 0)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Contextual Recall</div>
                <div class="kpi-value val-purple">{qm['contextual_recall']['mean'] or 'N/A'}</div>
                <div class="kpi-sub">Median: {qm['contextual_recall']['median'] or 'N/A'} | N/A unanswerable: {qm['contextual_recall'].get('unanswerable_na_count', 0)}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Avg Total Latency</div>
                <div class="kpi-value val-amber">{lat['total']['mean'] or 0:.0f} <span style="font-size:16px;">ms</span></div>
                <div class="kpi-sub">p50: {lat['total']['median'] or 0:.0f} ms | p95: {lat['total']['p95'] or 0:.0f} ms</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Retrieval Latency</div>
                <div class="kpi-value val-blue">{lat['retrieval']['mean'] or 0:.0f} <span style="font-size:16px;">ms</span></div>
                <div class="kpi-sub">Gen mean: {lat['generation']['mean'] or 0:.0f} ms</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Cost & Tokens</div>
                <div class="kpi-value val-emerald" style="font-size: 26px;">{cost_display}</div>
                <div class="kpi-sub">{cost['total_tokens']:,} total tokens</div>
            </div>
        </div>

        <!-- Visualizations -->
        <section class="section">
            <h2 class="section-title">📊 Quality & Performance Visualizations</h2>
            <div class="chart-container">
                <a href="{chart_filename}" target="_blank" title="Click to view full image in new tab">
                    <img src="{chart_filename}" alt="Evaluation Charts" class="chart-img" />
                </a>
            </div>
        </section>

        <!-- Category Breakdown -->
        <section class="section">
            <h2 class="section-title">📁 Performance by Question Category</h2>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Sample Count</th>
                        <th>Answer Relevancy (Mean)</th>
                        <th>Faithfulness (Mean)</th>
                        <th>Contextual Recall (Mean)</th>
                        <th>Avg Latency</th>
                    </tr>
                </thead>
                <tbody>
                    {cat_table_html}
                </tbody>
            </table>
        </section>

        <!-- Flagged Questions -->
        {flagged_html}

        <!-- Interactive Per-Question Table -->
        <section class="section">
            <h2 class="section-title">🔍 Detailed Sample Evaluation Results</h2>
            <div class="toolbar">
                <input type="text" id="searchInput" class="search-box" placeholder="Search questions, categories, IDs, answers..." onkeyup="filterTable()">
                <div class="filter-group">
                    {filter_buttons_html}
                </div>
            </div>
            <div style="overflow-x: auto;">
                <table id="resultsTable">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Category</th>
                            <th>Question</th>
                            <th>Relevancy</th>
                            <th>Faithfulness</th>
                            <th>Recall</th>
                            <th>Latency</th>
                            <th>Status</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {all_table_rows_html}
                    </tbody>
                </table>
            </div>
        </section>

        <footer>
            Generated automatically by Mutual Fund Facts RAG Evaluation Pipeline • {meta['timestamp']}
        </footer>
    </div>

    <script>
        let currentCategory = 'all';

        function toggleDetails(rowId) {{
            const row = document.getElementById(rowId);
            if (!row) return;
            if (row.style.display === 'none' || row.style.display === '') {{
                row.style.display = 'table-row';
            }} else {{
                row.style.display = 'none';
            }}
        }}

        function setCategoryFilter(cat, btn) {{
            currentCategory = cat.toLowerCase();
            document.querySelectorAll('.filter-pill').forEach(b => b.classList.remove('active'));
            if (btn) btn.classList.add('active');
            filterTable();
        }}

        function filterTable() {{
            const input = document.getElementById('searchInput');
            const term = (input ? input.value : '').toLowerCase().trim();
            const rows = document.querySelectorAll('#resultsTable tbody tr.main-row');

            rows.forEach(r => {{
                const cat = (r.getAttribute('data-category') || '').toLowerCase();
                const isFlagged = r.getAttribute('data-flagged') === 'true';
                const text = r.innerText.toLowerCase();

                let matchesCat = true;
                if (currentCategory === 'flagged') {{
                    matchesCat = isFlagged;
                }} else if (currentCategory !== 'all') {{
                    matchesCat = (cat === currentCategory);
                }}

                const matchesSearch = !term || text.includes(term);

                const nextDrawer = r.nextElementSibling;
                if (matchesCat && matchesSearch) {{
                    r.style.display = '';
                }} else {{
                    r.style.display = 'none';
                    if (nextDrawer && nextDrawer.classList.contains('drawer-row')) {{
                        nextDrawer.style.display = 'none';
                    }}
                }}
            }});
        }}
    </script>
</body>
</html>"""

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    return out_file



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
