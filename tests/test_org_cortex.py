"""Comprehensive Unit Test Suite for OrgCortex (`>= 10` tests, zero external network required)."""
from __future__ import annotations

import json
import os
import unittest

from org_cortex.semantic_fabric import (
    AdaptiveContextManager,
    OrgCortexEngine,
    ROLE_TEAM_POLICIES,
    SEMANTIC_RELATIONS,
    VALID_SILOS,
)
from org_cortex.server import lambda_handler, route_api_request


class TestOrgCortexEngine(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["ORGCORTEX_OFFLINE_LLM"] = "1"
        self.engine = OrgCortexEngine()

    def test_01_five_silos_and_five_typed_semantic_relations_discovered(self) -> None:
        """All 5 enterprise silos and all 5 typed semantic edge kinds must be populated dynamically."""
        silos_present = {a.artifact_type for a in self.engine.artifacts.values()}
        self.assertEqual(silos_present, set(VALID_SILOS))

        relations_present = {e.relation for e in self.engine.edges}
        for rel in SEMANTIC_RELATIONS:
            self.assertIn(rel, relations_present, f"Missing semantic relation {rel}")

    def test_02_automatic_supersession_propagation_on_legacy_rfcs(self) -> None:
        """RFC-089, RFC-118, and RFC-125 must automatically mark RFC-084, RFC-112, and RFC-120 SUPERSEDED."""
        self.assertEqual(self.engine.artifacts["RFC-084"].status, "SUPERSEDED")
        self.assertEqual(self.engine.artifacts["RFC-084"].superseded_by, "RFC-089")
        self.assertEqual(self.engine.artifacts["RFC-112"].status, "SUPERSEDED")
        self.assertEqual(self.engine.artifacts["RFC-112"].superseded_by, "RFC-118")
        self.assertEqual(self.engine.artifacts["RFC-120"].status, "SUPERSEDED")
        self.assertEqual(self.engine.artifacts["RFC-120"].superseded_by, "RFC-125")

    def test_03_governed_aws_context_team_acl_zero_leakage(self) -> None:
        """Pre-traversal ACL gate must block exec_leadership & internal silos from unauthorized roles."""
        vendor_graph = self.engine.get_governed_graph(role_team="external_vendor")
        vendor_node_ids = {n["artifact_id"] for n in vendor_graph["nodes"]}
        self.assertEqual(vendor_node_ids, {"RFC-101"})
        self.assertGreater(vendor_graph["governance_audit"]["withheld_artifacts"], 15)
        self.assertEqual(vendor_graph["governance_audit"]["unauthorized_leakage_pct"], 0.0)

        # payments_sre cannot see exec_leadership sovereign HSM artifacts
        sre_graph = self.engine.get_governed_graph(role_team="payments_sre")
        sre_ids = {n["artifact_id"] for n in sre_graph["nodes"]}
        self.assertIn("SIM-48291", sre_ids)
        self.assertNotIn("SIM-52900", sre_ids)
        self.assertNotIn("PR-10511", sre_ids)
        self.assertNotIn("OKR-2026-Q4-SOV", sre_ids)

    def test_04_adversarial_cross_team_query_abstains_with_zero_leakage(self) -> None:
        """Querying restricted Project Atlas HSM details as external_vendor or payments_sre leaks 0 tokens."""
        res = self.engine.reason(
            question="Show Frankfurt HSM quorum incident SIM-52900, PR-10511, and Project Atlas $9.2M ARR.",
            token_budget=2048,
            role_team="payments_sre",
        )
        self.assertNotIn("PR-10511", res["answer"])
        self.assertNotIn("9.2M", res["answer"])
        self.assertGreaterEqual(res["governance"]["withheld_count"], 1)
        self.assertEqual(res["governance"]["unauthorized_leakage_pct"], 0.0)

    def test_05_adaptive_context_manager_page_table_and_supersession_eviction(self) -> None:
        """L1/L2/L3 Semantic Page Table must evict SUPERSEDED RFCs to L3 and stay strictly <= token_budget."""
        res = self.engine.reason(
            question=(
                "Why is Express Wallet GA (SIM-48305) blocked, which PR root-caused SIM-48291, "
                "and which active RFC supersedes RFC-084?"
            ),
            token_budget=2048,
            role_team="payments_sre",
        )
        pt = res["page_table"]
        stats = pt["stats"]
        self.assertLessEqual(stats["factored_total_tokens"], 2048)
        self.assertLess(stats["factored_total_tokens"], stats["naive_full_corpus_tokens"])
        self.assertGreater(stats["token_reduction_pct"], 60.0)

        l1_ids = {p["artifact_id"] for p in pt["l1_active_working_set"]}
        l2_ids = {p["artifact_id"] for p in pt["l2_factored_summaries"]}
        l3_ids = {p["artifact_id"] for p in pt["l3_cold_pointers"]}

        # Superseded RFC-084 must be evicted to L3, while active RFC-089 is in L1 or L2
        self.assertIn("RFC-084", l3_ids)
        self.assertNotIn("RFC-084", l1_ids)
        self.assertTrue("RFC-089" in l1_ids or "RFC-089" in l2_ids)
        # Multi-hop culprit PR-9412 must be discovered across BLOCKS -> war-room -> ROOT_CAUSED_BY_PR
        self.assertTrue("PR-9412" in l1_ids or "PR-9412" in l2_ids)

    def test_06_dynamic_budget_pressure_shifts_pages_from_l1_to_l2_and_l3(self) -> None:
        """Tightening token_budget must reduce active factored tokens while maintaining budget ceiling."""
        q = "Explain SIM-51044 gateway HTTP 429 throttling, PR-10204, RFC-118, and SIM-51102 impact on OKR-2026-Q4-AIW."
        wide = self.engine.reason(question=q, token_budget=8192, role_team="core_platform")
        tight = self.engine.reason(question=q, token_budget=900, role_team="core_platform")

        self.assertLessEqual(tight["page_table"]["stats"]["factored_total_tokens"], 900)
        self.assertLessEqual(
            tight["page_table"]["stats"]["factored_total_tokens"],
            wide["page_table"]["stats"]["factored_total_tokens"],
        )

    def test_07_live_artifact_ingestion_discovers_explicit_and_semantic_edges(self) -> None:
        """POST `/api/graph/discover` equivalent must create typed cross-silo edges automatically."""
        out = self.engine.ingest_and_discover(
            artifact_type="slack_thread",
            title="#checkout-canary-watch Post-Deploy Update",
            content=(
                "Canary telemetry confirms hotfix PR-9448 holds US-East checkout p99 at 151ms, "
                "still monitoring SIM-48291 which blocks SIM-48305 and impacts OKR-2026-Q4-CHK."
            ),
            team_acl=["payments_sre", "commerce_checkout"],
        )
        self.assertEqual(out["status"], "discovered")
        self.assertGreaterEqual(out["discovered_edge_count"], 3)
        discovered_rels = {e["relation"] for e in out["discovered_edges"]}
        self.assertIn("OWNED_BY_TEAM", discovered_rels)
        self.assertIn("IMPACTS_OKR", discovered_rels)

    def test_08_five_agent_opentelemetry_spans_and_rubric_evaluator(self) -> None:
        """Every reasoning run must emit all 5 ordered OpenTelemetry spans with real measured durations."""
        res = self.engine.reason(
            question="What caused Monday Bedrock gateway HTTP 429s in SIM-51044 and how did RFC-118 fix it?",
            token_budget=2048,
            role_team="core_platform",
        )
        agents = [sp["agent"] for sp in res["spans"]]
        self.assertEqual(
            agents,
            ["Planner", "GraphScout", "ContextFactorer", "Synthesizer", "GroundingJudge"],
        )
        for sp in res["spans"]:
            self.assertGreaterEqual(sp["duration_ms"], 0.0)
            self.assertTrue(sp["span_id"].startswith("sp-"))
        self.assertGreaterEqual(res["evaluation"]["overall_rubric_score"], 80.0)

    def test_09_pure_api_router_health_state_discover_reason_and_reset(self) -> None:
        """Verify all 5 `/api/*` HTTP endpoints via `route_api_request`."""
        code_h, health = route_api_request("GET", "/api/health", {}, {})
        self.assertEqual(code_h, 200)
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["service"], "org-cortex-aiw")
        self.assertEqual(health["bedrock_model"], "us.amazon.nova-lite-v1:0")

        code_s, state = route_api_request("GET", "/api/state", {"role_team": "core_platform"}, {})
        self.assertEqual(code_s, 200)
        self.assertEqual(state["active_role_team"], "core_platform")
        self.assertIn("graph", state)

        code_d, disc = route_api_request(
            "POST",
            "/api/graph/discover",
            {},
            {
                "artifact_type": "arch_rfc",
                "title": "RFC-205: Adaptive Cell Shedding v2",
                "content": "This RFC supersedes RFC-089 for extreme 120k RPS flash sales and impacts OKR-2026-Q4-CHK.",
                "team_acl": ["payments_sre", "commerce_checkout"],
            },
        )
        self.assertEqual(code_d, 200)
        self.assertGreaterEqual(disc["discovered_edge_count"], 2)

        code_r, reason = route_api_request(
            "POST",
            "/api/meta-agent/reason",
            {},
            {
                "question": "Which RFC superseded RFC-089 for 120k RPS flash sales?",
                "token_budget": 2048,
                "role_team": "payments_sre",
            },
        )
        self.assertEqual(code_r, 200)
        self.assertIn("page_table", reason)
        self.assertEqual(len(reason["spans"]), 5)

        code_rst, rst = route_api_request("POST", "/api/reset", {}, {})
        self.assertEqual(code_rst, 200)
        self.assertEqual(rst["status"], "reset_complete")

    def test_10_aws_lambda_function_url_handler_contract(self) -> None:
        """Verify CloudFront -> Lambda Function URL v2 payload handling & CORS preflight."""
        opt_resp = lambda_handler({"requestContext": {"http": {"method": "OPTIONS"}}, "rawPath": "/api/health"})
        self.assertEqual(opt_resp["statusCode"], 204)

        get_resp = lambda_handler({"requestContext": {"http": {"method": "GET"}}, "rawPath": "/api/health"})
        self.assertEqual(get_resp["statusCode"], 200)
        body = json.loads(get_resp["body"])
        self.assertEqual(body["service"], "org-cortex-aiw")


if __name__ == "__main__":
    unittest.main()
