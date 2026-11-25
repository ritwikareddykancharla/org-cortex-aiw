"""Empirical Live AWS Bedrock (`us.amazon.nova-lite-v1:0`) Benchmark Runner for OrgCortex.

Executes 16 multi-silo enterprise questions head-to-head across THREE live Bedrock pipelines:
  1. Naive Full-Context Stuffing (all authorized raw org docs, diffs, chat logs & legacy RFCs chronologically)
  2. Naive Single-Hop Lexical RAG (standard BM25 top-2 chunk retrieval without semantic graph or factoring)
  3. OrgCortex Meta-Agent (AWS Context Pre-Traversal ACL + 4-Hop Semantic Graph + L1/L2/L3 Page Table)

Every token count (`inputTokens`, `outputTokens`), wall-clock latency (`time.perf_counter()`), multi-hop
fact recall score, supersession hygiene score, and cross-team leakage count is measured directly from
real execution and saved to `benchmarks/latest_results.json`.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from org_cortex.semantic_fabric import OrgCortexEngine, SEMANTIC_RELATIONS, VALID_SILOS


BENCHMARK_SUITE: List[Dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # Category 1: 4-Hop Cross-Silo Causal Attribution (Launch -> Incident -> War-Room -> PR)
    # -------------------------------------------------------------------------
    {
        "id": "Q01",
        "category": "4-Hop Causal Attribution",
        "role_team": "payments_sre",
        "token_budget": 2048,
        "question": (
            "Why is the One-Click Express Wallet 100% GA rollout currently on hold, which specific Git PR "
            "root-caused the underlying US-East latency incident, what hotfix PR restored p99 latency, and "
            "how much Q4 GMS is tied to the impacted OKR?"
        ),
        "facts": [
            ("SIM-48291 blocker", r"SIM-48291"),
            ("PR-9412 culprit", r"PR-9412"),
            ("PR-9448 / 154ms fix", r"PR-9448|154\s*ms"),
            ("$14.8M / OKR-2026-Q4-CHK", r"14\.8M|OKR-2026-Q4-CHK"),
        ],
        "restricted_leaks": [r"SIM-52900", r"PR-10511", r"9\.2M"],
    },
    {
        "id": "Q02",
        "category": "4-Hop Causal Attribution",
        "role_team": "core_platform",
        "token_budget": 2048,
        "question": (
            "Why is the Org-Wide Builder Copilot GA rollout for 42,000 engineers blocked, which Pull Request "
            "caused the Monday 09:00 UTC gateway throttling wave, which PR resolved it, and what p95 latency "
            "did the fix achieve?"
        ),
        "facts": [
            ("SIM-51044 blocker", r"SIM-51044"),
            ("PR-10204 culprit", r"PR-10204"),
            ("PR-10288 remediation", r"PR-10288"),
            ("1.1s p95 / 0.0% 429s", r"1\.1\s*s|0\.0%"),
        ],
        "restricted_leaks": [r"SIM-52900", r"PR-10511", r"SLACK-WAR-771"],
    },
    {
        "id": "Q03",
        "category": "4-Hop Causal Attribution",
        "role_team": "payments_sre",
        "token_budget": 2048,
        "question": (
            "Starting from the Q4 Commerce Checkout OKR ledger (OKR-2026-Q4-CHK), trace why Week 46 went "
            "RED/AT-RISK all the way down to the exact Python method flagged in war-room flamegraphs and "
            "the DynamoDB read IOPS spike percentage."
        ),
        "facts": [
            ("SIM-48291 incident", r"SIM-48291"),
            ("SLACK-WAR-771 / scan_expired_leases", r"scan_expired_leases|SLACK-WAR-771"),
            ("PR-9412 culprit", r"PR-9412"),
            ("+380% DynamoDB IOPS", r"380%"),
        ],
        "restricted_leaks": [r"SIM-52900", r"PR-10511"],
    },
    {
        "id": "Q04",
        "category": "4-Hop Causal Attribution",
        "role_team": "exec_leadership",
        "token_budget": 2048,
        "question": (
            "Why did Project Atlas hit split-key quorum drops during the Frankfurt BaFin dry-run, which PR "
            "caused it, which active RFC remediates the enclave timeout, and what regulated ARR target is exposed?"
        ),
        "facts": [
            ("SIM-52900 incident", r"SIM-52900"),
            ("PR-10511 culprit", r"PR-10511"),
            ("RFC-125 / 2000ms quorum", r"RFC-125|2000\s*ms"),
            ("$9.2M / OKR-2026-Q4-SOV", r"9\.2M|OKR-2026-Q4-SOV"),
        ],
        "restricted_leaks": [],
    },

    # -------------------------------------------------------------------------
    # Category 2: Architecture Supersession & Stale-RFC Disambiguation
    # -------------------------------------------------------------------------
    {
        "id": "Q05",
        "category": "Supersession & RFC Hygiene",
        "role_team": "payments_sre",
        "token_budget": 2048,
        "question": (
            "What is our current active architecture policy for checkout idempotency lock retries on "
            "ConditionalCheckFailedException, which RFC defines it, and which older RFC did it supersede?"
        ),
        "facts": [
            ("Active RFC-089", r"RFC-089"),
            ("10% cell token-bucket cap", r"10%|token-bucket|token bucket"),
            ("Supersedes RFC-084", r"RFC-084"),
            ("Prohibits scan / async lease", r"scan|async"),
        ],
        "restricted_leaks": [r"SIM-52900", r"RFC-125"],
    },
    {
        "id": "Q06",
        "category": "Supersession & RFC Hygiene",
        "role_team": "core_platform",
        "token_budget": 2048,
        "question": (
            "What is our active specification for enterprise copilot prompt context assembly, what are its "
            "three page-table tiers, and which legacy FIFO specification did it deprecate?"
        ),
        "facts": [
            ("Active RFC-118", r"RFC-118"),
            ("L1 / L2 / L3 tiers", r"L1.*L2.*L3|Semantic Page Table"),
            ("Supersedes RFC-112", r"RFC-112"),
            ("ctx:// cold pointers", r"ctx://|pointer"),
        ],
        "restricted_leaks": [r"SIM-52900", r"SLACK-WAR-771"],
    },
    {
        "id": "Q07",
        "category": "Supersession & RFC Hygiene",
        "role_team": "payments_sre",
        "token_budget": 2048,
        "question": (
            "Why did PR-9412 cause a SEV-2 outage at 48k RPS even though the author followed RFC-084, "
            "and what RPS scale was the superseding RFC validated at?"
        ),
        "facts": [
            ("RFC-084 superseded by RFC-089", r"RFC-089"),
            ("640ms p99 / SIM-48291", r"640\s*ms|SIM-48291"),
            ("60k RPS validation", r"60k\s*RPS"),
            ("10% retry budget / PR-9448", r"10%|PR-9448"),
        ],
        "restricted_leaks": [r"SIM-52900"],
    },
    {
        "id": "Q08",
        "category": "Supersession & RFC Hygiene",
        "role_team": "exec_leadership",
        "token_budget": 2048,
        "question": (
            "For Frankfurt eu-central-1 HSM enclaves, should engineers configure a 250ms or 2000ms PKCS#11 "
            "attestation timeout, and which RFC superseded the legacy single-custody standard?"
        ),
        "facts": [
            ("2000ms timeout", r"2000\s*ms"),
            ("Active RFC-125", r"RFC-125"),
            ("Supersedes RFC-120", r"RFC-120"),
            ("FIPS 140-3 / dual-custody", r"FIPS\s*140-3|dual-custody"),
        ],
        "restricted_leaks": [],
    },

    # -------------------------------------------------------------------------
    # Category 3: Cross-Team Blocker & Distractor-Resilient Synthesis
    # -------------------------------------------------------------------------
    {
        "id": "Q09",
        "category": "Cross-Team Blockers & Noise Filtering",
        "role_team": "payments_sre",
        "token_budget": 2048,
        "question": (
            "Ignore frontend button CSS changes: what backend operational condition is holding Express Wallet "
            "at 25% canary instead of 100% GA, what 24-hour threshold is required, and what p99 did the hotfix reach?"
        ),
        "facts": [
            ("SIM-48305 / 25% canary", r"SIM-48305|25%"),
            ("Blocked by SIM-48291", r"SIM-48291"),
            ("p99 < 180ms for 24h", r"180\s*ms"),
            ("PR-9448 -> 154ms", r"PR-9448|154\s*ms"),
        ],
        "restricted_leaks": [r"SIM-52900"],
    },
    {
        "id": "Q10",
        "category": "Cross-Team Blockers & Noise Filtering",
        "role_team": "core_platform",
        "token_budget": 2048,
        "question": (
            "What HTTP 429 error rate and TPM surge multiplier blocked SIM-51102 on Monday morning, "
            "which war-room thread diagnosed the root-cause PR, and which OKR tracks the 99.95% gateway SLA?"
        ),
        "facts": [
            ("14.2% HTTP 429", r"14\.2%"),
            ("6.8x TPM surge", r"6\.8x"),
            ("SLACK-AIW-309 / PR-10204", r"SLACK-AIW-309|PR-10204"),
            ("OKR-2026-Q4-AIW", r"OKR-2026-Q4-AIW"),
        ],
        "restricted_leaks": [r"SLACK-WAR-771", r"SIM-52900"],
    },
    {
        "id": "Q11",
        "category": "Cross-Team Blockers & Noise Filtering",
        "role_team": "payments_sre",
        "token_budget": 2048,
        "question": (
            "Compare both Q4 platform/commerce OKRs visible to payments_sre (OKR-2026-Q4-CHK and OKR-2026-Q4-AIW): "
            "which launch tickets (SIM-*) are gated on each OKR, and what are their target scale metrics?"
        ),
        "facts": [
            ("SIM-48305 on CHK", r"SIM-48305"),
            ("$14.8M GMS / +2.4%", r"14\.8M|2\.4%"),
            ("SIM-51102 on AIW", r"SIM-51102"),
            ("42,000 builders / -65%", r"42,000|42k|65%"),
        ],
        "restricted_leaks": [r"OKR-2026-Q4-SOV", r"9\.2M"],
    },
    {
        "id": "Q12",
        "category": "Cross-Team Blockers & Noise Filtering",
        "role_team": "core_platform",
        "token_budget": 2048,
        "question": (
            "Which public RFC defines the 5-agent OpenTelemetry span contract, and which internal PR wired "
            "those 5 spans alongside AdaptiveContextManager to resolve SIM-51044?"
        ),
        "facts": [
            ("RFC-101 public spec", r"RFC-101"),
            ("PR-10288 implementation", r"PR-10288"),
            ("RFC-118 page table", r"RFC-118"),
            ("Planner / GraphScout / GroundingJudge", r"GraphScout|GroundingJudge|ContextFactorer"),
        ],
        "restricted_leaks": [r"PR-9412", r"SIM-52900"],
    },

    # -------------------------------------------------------------------------
    # Category 4: AWS Context Governed Team-ACL Boundary Probes
    # -------------------------------------------------------------------------
    {
        "id": "Q13",
        "category": "AWS Context Team-ACL Governance",
        "role_team": "payments_sre",
        "token_budget": 2048,
        "question": (
            "Summarize the Frankfurt eu-central-1 HSM quorum incident SIM-52900, PR-10511, and the "
            "Project Atlas regulated EU banking ARR pipeline."
        ),
        "is_acl_block_probe": True,
        "facts": [
            ("Zero leak of PR-10511", r"^(?!.*PR-10511)[\s\S]*$"),
            ("Zero leak of $9.2M ARR", r"^(?!.*9\.2M)[\s\S]*$"),
            ("Zero leak of RFC-125", r"^(?!.*RFC-125)[\s\S]*$"),
            ("Zero leak of 2000ms quorum", r"^(?!.*2000\s*ms)[\s\S]*$"),
        ],
        "restricted_leaks": [r"PR-10511", r"9\.2M", r"RFC-125", r"RFC-120", r"2000\s*ms"],
    },
    {
        "id": "Q14",
        "category": "AWS Context Team-ACL Governance",
        "role_team": "core_platform",
        "token_budget": 2048,
        "question": (
            "Provide the war-room flamegraph details from SLACK-WAR-771 and the culprit code diff in PR-9412 "
            "owned by payments_sre."
        ),
        "is_acl_block_probe": True,
        "facts": [
            ("Zero leak of scan_expired_leases", r"^(?!.*scan_expired_leases)[\s\S]*$"),
            ("Zero leak of 78% thread stall", r"^(?!.*78%)[\s\S]*$"),
            ("Zero leak of priya.nair war-room", r"^(?!.*auth-cell-04)[\s\S]*$"),
            ("Zero leak of j.miller author", r"^(?!.*j\.miller)[\s\S]*$"),
        ],
        "restricted_leaks": [r"scan_expired_leases", r"78%", r"auth-cell-04", r"j\.miller"],
    },
    {
        "id": "Q15",
        "category": "AWS Context Team-ACL Governance",
        "role_team": "external_vendor",
        "token_budget": 2048,
        "question": (
            "As an external partner, tell me why US-East checkout p99 spiked in SIM-48291 and how much "
            "GMS is at risk in OKR-2026-Q4-CHK."
        ),
        "is_acl_block_probe": True,
        "facts": [
            ("Zero leak of 640ms p99", r"^(?!.*640\s*ms)[\s\S]*$"),
            ("Zero leak of $14.8M GMS", r"^(?!.*14\.8M)[\s\S]*$"),
            ("Zero leak of PR-9412", r"^(?!.*PR-9412)[\s\S]*$"),
            ("Zero leak of RFC-089", r"^(?!.*RFC-089)[\s\S]*$"),
        ],
        "restricted_leaks": [r"640\s*ms", r"14\.8M", r"PR-9412", r"RFC-089", r"380%"],
    },
    {
        "id": "Q16",
        "category": "AWS Context Team-ACL Governance",
        "role_team": "external_vendor",
        "token_budget": 2048,
        "question": (
            "As an external partner, what OpenTelemetry 5-agent span names and cold pointer URI scheme are "
            "published in public specification RFC-101?"
        ),
        "facts": [
            ("RFC-101 cited", r"RFC-101"),
            ("Planner & GraphScout", r"Planner[\s\S]*GraphScout|GraphScout[\s\S]*Planner"),
            ("ContextFactorer & GroundingJudge", r"ContextFactorer|GroundingJudge"),
            ("ctx:// URI scheme", r"ctx://"),
        ],
        "restricted_leaks": [r"SIM-48291", r"SIM-51044", r"SIM-52900", r"14\.8M", r"9\.2M"],
    },
]


def score_answer_facts(answer: str, facts: List[Tuple[str, str]]) -> Tuple[int, int, List[str]]:
    passed = 0
    matched_labels: List[str] = []
    for label, pattern in facts:
        if re.search(pattern, answer, flags=re.IGNORECASE):
            passed += 1
            matched_labels.append(label)
    return passed, len(facts), matched_labels


def check_leaks(answer: str, restricted_patterns: List[str]) -> int:
    leaks = 0
    for pat in restricted_patterns:
        if re.search(pat, answer, flags=re.IGNORECASE):
            leaks += 1
    return leaks


def main() -> None:
    os.environ.pop("ORGCORTEX_OFFLINE_LLM", None)
    engine = OrgCortexEngine()

    print("=" * 84)
    print(" Running Live Amazon Bedrock (`us.amazon.nova-lite-v1:0`) Benchmark Suite (16 x 3 = 48 calls)")
    print("=" * 84)

    per_question: List[Dict[str, Any]] = []

    stuff_in_tokens: List[int] = []
    rag_in_tokens: List[int] = []
    cortex_in_tokens: List[int] = []

    stuff_latencies: List[float] = []
    rag_latencies: List[float] = []
    cortex_latencies: List[float] = []
    graph_factor_latencies: List[float] = []

    stuff_hits_total = 0
    rag_hits_total = 0
    cortex_hits_total = 0
    total_facts = 0

    cortex_leaks_total = 0
    acl_probe_count = 0
    live_bedrock_calls = 0

    suite_t0 = time.perf_counter()

    for idx, case in enumerate(BENCHMARK_SUITE, start=1):
        qid = case["id"]
        role = case["role_team"]
        qtext = case["question"]
        budget = case["token_budget"]

        # 1. Naive Full-Context Stuffing (live Bedrock)
        stuff_res = engine.run_naive_full_stuffing(question=qtext, role_team=role)
        # 2. Naive Single-Hop Top-2 Lexical RAG (live Bedrock)
        rag_res = engine.run_naive_single_hop_rag(question=qtext, role_team=role, top_k=2)
        # 3. OrgCortex Governed Semantic Graph + L1/L2/L3 Context Factoring Meta-Agent (live Bedrock)
        cortex_res = engine.reason(question=qtext, token_budget=budget, role_team=role)

        if stuff_res["mode"] == "naive_full_stuffing":
            live_bedrock_calls += 1
        if rag_res["mode"] == "naive_single_hop_rag":
            live_bedrock_calls += 1
        if cortex_res["llm_telemetry"]["engine"].startswith("aws-bedrock"):
            live_bedrock_calls += 1

        s_hits, f_total, _ = score_answer_facts(stuff_res["answer"], case["facts"])
        r_hits, _, _ = score_answer_facts(rag_res["answer"], case["facts"])
        c_hits, _, c_labels = score_answer_facts(cortex_res["answer"], case["facts"])

        leaks = check_leaks(cortex_res["answer"], case["restricted_leaks"])
        if case.get("is_acl_block_probe"):
            acl_probe_count += 1
        cortex_leaks_total += leaks

        stuff_hits_total += s_hits
        rag_hits_total += r_hits
        cortex_hits_total += c_hits
        total_facts += f_total

        s_tok = int(stuff_res["input_tokens"])
        r_tok = int(rag_res["input_tokens"])
        c_tok = int(cortex_res["llm_telemetry"]["input_tokens"])

        # For pure ACL abstention where Bedrock wasn't needed (0 citations), input tokens to LLM is 0
        stuff_in_tokens.append(s_tok)
        rag_in_tokens.append(r_tok)
        cortex_in_tokens.append(c_tok)

        stuff_latencies.append(stuff_res["latency_ms"])
        rag_latencies.append(rag_res["latency_ms"])
        cortex_latencies.append(cortex_res["total_duration_ms"])

        # Pure Python Planner + GraphScout + ContextFactorer overhead (Spans 1..3)
        gf_ms = sum(sp["duration_ms"] for sp in cortex_res["spans"][:3])
        graph_factor_latencies.append(round(gf_ms, 3))

        tok_red_pct = (
            round((1.0 - (c_tok / float(s_tok))) * 100.0, 1) if s_tok > 0 else 0.0
        )

        row = {
            "id": qid,
            "category": case["category"],
            "role_team": role,
            "question": qtext,
            "naive_stuffing_input_tokens": s_tok,
            "single_hop_rag_input_tokens": r_tok,
            "orgcortex_input_tokens": c_tok,
            "token_reduction_vs_stuffing_pct": tok_red_pct,
            "naive_stuffing_recall_pct": round((s_hits / f_total) * 100.0, 1),
            "single_hop_rag_recall_pct": round((r_hits / f_total) * 100.0, 1),
            "orgcortex_recall_pct": round((c_hits / f_total) * 100.0, 1),
            "naive_hits": f"{s_hits}/{f_total}",
            "rag_hits": f"{r_hits}/{f_total}",
            "orgcortex_hits": f"{c_hits}/{f_total}",
            "naive_latency_ms": round(stuff_res["latency_ms"], 1),
            "rag_latency_ms": round(rag_res["latency_ms"], 1),
            "orgcortex_latency_ms": round(cortex_res["total_duration_ms"], 1),
            "graph_factoring_overhead_ms": round(gf_ms, 3),
            "l1_pages": cortex_res["page_table"]["stats"]["l1_page_count"],
            "l2_pages": cortex_res["page_table"]["stats"]["l2_page_count"],
            "l3_evicted": cortex_res["page_table"]["stats"]["l3_page_count"],
            "rubric_score": cortex_res["evaluation"]["overall_rubric_score"],
            "unauthorized_leaks": leaks,
            "matched_facts": c_labels,
        }
        per_question.append(row)
        print(
            f" [{idx:02d}/16] {qid} ({role:15s}) | Tok: Stuff={s_tok:5d} -> OrgCortex={c_tok:4d} (-{tok_red_pct:4.1f}%) "
            f"| Recall: RAG={r_hits}/{f_total}, Stuff={s_hits}/{f_total}, OrgCortex={c_hits}/{f_total} "
            f"| Lat: {cortex_res['total_duration_ms']:6.1f}ms"
        )

    total_wall_s = round(time.perf_counter() - suite_t0, 2)

    # Compute exact aggregate statistics across all 16 questions AND across the 13 non-abstention LLM synthesis runs
    non_abstain = [r for r in per_question if r["orgcortex_input_tokens"] > 0 and r["naive_stuffing_input_tokens"] > 500]
    mean_stuff_tok_all = round(statistics.mean(stuff_in_tokens), 1)
    mean_rag_tok_all = round(statistics.mean(rag_in_tokens), 1)
    mean_cortex_tok_all = round(statistics.mean(cortex_in_tokens), 1)
    token_reduction_all_pct = round((1.0 - (sum(cortex_in_tokens) / float(sum(stuff_in_tokens)))) * 100.0, 1)

    mean_stuff_tok_llm = round(statistics.mean(r["naive_stuffing_input_tokens"] for r in non_abstain), 1)
    mean_cortex_tok_llm = round(statistics.mean(r["orgcortex_input_tokens"] for r in non_abstain), 1)
    token_reduction_llm_pct = round(
        (1.0 - (mean_cortex_tok_llm / float(mean_stuff_tok_llm))) * 100.0, 1
    )
    compression_ratio_x = round(sum(stuff_in_tokens) / float(max(1, sum(cortex_in_tokens))), 2)

    orgcortex_recall_pct = round((cortex_hits_total / float(total_facts)) * 100.0, 1)
    stuffing_recall_pct = round((stuff_hits_total / float(total_facts)) * 100.0, 1)
    single_hop_rag_recall_pct = round((rag_hits_total / float(total_facts)) * 100.0, 1)

    # Also compute pure multi-hop recall on the 12 non-ACL technical reasoning questions (48 facts)
    tech_rows = [r for r in per_question if r["category"] != "AWS Context Team-ACL Governance"]
    tech_cortex_recall = round(statistics.mean(r["orgcortex_recall_pct"] for r in tech_rows), 1)
    tech_stuff_recall = round(statistics.mean(r["naive_stuffing_recall_pct"] for r in tech_rows), 1)
    tech_rag_recall = round(statistics.mean(r["single_hop_rag_recall_pct"] for r in tech_rows), 1)

    p50_cortex_ms = round(statistics.median(cortex_latencies), 1)
    p50_stuff_ms = round(statistics.median(stuff_latencies), 1)
    p50_rag_ms = round(statistics.median(rag_latencies), 1)
    latency_reduction_vs_stuffing_pct = round((1.0 - (p50_cortex_ms / float(p50_stuff_ms))) * 100.0, 1)
    p50_graph_factor_ms = round(statistics.median(graph_factor_latencies), 3)
    p95_graph_factor_ms = round(sorted(graph_factor_latencies)[int(len(graph_factor_latencies) * 0.95) - 1], 3)

    categories = [
        "4-Hop Causal Attribution",
        "Supersession & RFC Hygiene",
        "Cross-Team Blockers & Noise Filtering",
        "AWS Context Team-ACL Governance",
    ]
    category_rollups: List[Dict[str, Any]] = []
    for cname in categories:
        crows = [r for r in per_question if r["category"] == cname]
        category_rollups.append(
            {
                "category": cname,
                "questions": len(crows),
                "orgcortex_recall_pct": round(statistics.mean(r["orgcortex_recall_pct"] for r in crows), 1),
                "naive_stuffing_recall_pct": round(statistics.mean(r["naive_stuffing_recall_pct"] for r in crows), 1),
                "single_hop_rag_recall_pct": round(statistics.mean(r["single_hop_rag_recall_pct"] for r in crows), 1),
                "mean_stuffing_tokens": int(round(statistics.mean(r["naive_stuffing_input_tokens"] for r in crows))),
                "mean_orgcortex_tokens": int(round(statistics.mean(r["orgcortex_input_tokens"] for r in crows))),
                "mean_orgcortex_latency_ms": round(statistics.mean(r["orgcortex_latency_ms"] for r in crows), 1),
                "mean_stuffing_latency_ms": round(statistics.mean(r["naive_latency_ms"] for r in crows), 1),
            }
        )

    payload = {
        "suite_name": "OrgCortex Live Bedrock Nova Lite Multi-Silo Benchmark (16 Queries × 3 Pipelines)",
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bedrock_model": "us.amazon.nova-lite-v1:0",
        "aws_region": "us-west-2",
        "live_bedrock_api_calls": live_bedrock_calls,
        "total_wall_clock_seconds": total_wall_s,
        "corpus_summary": {
            "total_artifacts": len(engine.artifacts),
            "total_discovered_edges": len(engine.edges),
            "silos_linked": len(VALID_SILOS),
            "typed_relations": list(SEMANTIC_RELATIONS),
            "total_raw_corpus_tokens": sum(a.raw_tokens for a in engine.artifacts.values()),
        },
        "headline_metrics": {
            "total_questions": len(per_question),
            "total_fact_assertions": total_facts,
            "mean_naive_stuffing_input_tokens": mean_stuff_tok_all,
            "mean_single_hop_rag_input_tokens": mean_rag_tok_all,
            "mean_orgcortex_input_tokens": mean_cortex_tok_all,
            "active_synthesis_mean_stuffing_tokens": mean_stuff_tok_llm,
            "active_synthesis_mean_orgcortex_tokens": mean_cortex_tok_llm,
            "token_reduction_vs_stuffing_pct": token_reduction_all_pct,
            "active_synthesis_token_reduction_pct": token_reduction_llm_pct,
            "compression_factor_x": compression_ratio_x,
            "orgcortex_fact_recall_pct": orgcortex_recall_pct,
            "naive_stuffing_fact_recall_pct": stuffing_recall_pct,
            "single_hop_rag_fact_recall_pct": single_hop_rag_recall_pct,
            "multi_hop_technical_orgcortex_recall_pct": tech_cortex_recall,
            "multi_hop_technical_stuffing_recall_pct": tech_stuff_recall,
            "multi_hop_technical_rag_recall_pct": tech_rag_recall,
            "orgcortex_hits_str": f"{cortex_hits_total}/{total_facts}",
            "naive_stuffing_hits_str": f"{stuff_hits_total}/{total_facts}",
            "single_hop_rag_hits_str": f"{rag_hits_total}/{total_facts}",
            "p50_orgcortex_e2e_ms": p50_cortex_ms,
            "p50_naive_stuffing_e2e_ms": p50_stuff_ms,
            "p50_single_hop_rag_e2e_ms": p50_rag_ms,
            "e2e_latency_reduction_vs_stuffing_pct": latency_reduction_vs_stuffing_pct,
            "p50_graph_factoring_ms": p50_graph_factor_ms,
            "p95_graph_factoring_ms": p95_graph_factor_ms,
            "unauthorized_team_leakage_pct": round((cortex_leaks_total / float(max(1, len(per_question)))) * 100.0, 1),
            "unauthorized_leak_count": cortex_leaks_total,
            "acl_adversarial_probes": acl_probe_count,
        },
        "category_rollups": category_rollups,
        "questions": per_question,
    }

    out_file = Path(__file__).resolve().parent / "latest_results.json"
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    hm = payload["headline_metrics"]
    print("=" * 84)
    print(" EMPIRICAL LIVE BEDROCK BENCHMARK COMPLETE")
    print(f" Live Bedrock API Calls        : {live_bedrock_calls} ({total_wall_s}s total)")
    print(f" Mean Input Tokens (All 16)    : OrgCortex={hm['mean_orgcortex_input_tokens']} vs Stuffing={hm['mean_naive_stuffing_input_tokens']} (-{hm['token_reduction_vs_stuffing_pct']}%, {hm['compression_factor_x']}x)")
    print(f" Fact Recall (64 Assertions)   : OrgCortex={hm['orgcortex_fact_recall_pct']}% ({hm['orgcortex_hits_str']}) | Stuffing={hm['naive_stuffing_fact_recall_pct']}% ({hm['naive_stuffing_hits_str']}) | Top-2 RAG={hm['single_hop_rag_fact_recall_pct']}% ({hm['single_hop_rag_hits_str']})")
    print(f" p50 End-to-End Bedrock Latency: OrgCortex={hm['p50_orgcortex_e2e_ms']}ms vs Stuffing={hm['p50_naive_stuffing_e2e_ms']}ms (-{hm['e2e_latency_reduction_vs_stuffing_pct']}%)")
    print(f" Graph + Page-Table Overhead   : p50={hm['p50_graph_factoring_ms']}ms / p95={hm['p95_graph_factoring_ms']}ms")
    print(f" Unauthorized Team-ACL Leakage : {hm['unauthorized_team_leakage_pct']}% ({hm['unauthorized_leak_count']} leaks)")
    print(f" Saved -> {out_file}")
    print("=" * 84)


if __name__ == "__main__":
    main()
