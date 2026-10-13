import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service
from gtm_agent.gtm_agent import build_prospect_profile, get_prospect, score_prospect


class SensitiveDataTests(unittest.TestCase):
    def tearDown(self):
        data_service._PROFILES.clear()

    def test_get_prospect_returns_only_allowlisted_fields(self):
        record = {
            "prospect_id": "LEAD-1",
            "name": "Alex Example",
            "email": "alex@example.com",
            "disqualified": False,
            "annual_revenue": 1000000,
            "billing_qualification": {"tax_id": "sensitive"},
        }
        with patch.object(data_service, "get_prospect_record", return_value=record):
            result = get_prospect.invoke({"prospect_id": "LEAD-1"})

        self.assertEqual(
            result["prospect"],
            {key: record[key] for key in ("prospect_id", "name", "email", "disqualified")},
        )

    def test_new_and_cached_profiles_are_sanitized(self):
        record = {
            "prospect_id": "LEAD-1",
            "name": "Alex Example",
            "annual_revenue": 1000000,
            "billing_qualification": {"tax_id": "sensitive"},
        }
        with (
            patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(data_service, "get_prospect_record", return_value=record),
            patch.object(data_service, "fetch_engagement_history", return_value=[]),
            patch.object(data_service, "fetch_account_details", return_value={}),
            patch.object(data_service, "fetch_tech_stack", return_value=["Kafka"]),
            patch.object(data_service, "save_profile_to_db") as save_profile,
        ):
            built = build_prospect_profile.invoke({"prospect_id": "LEAD-1"})["prospect_profile"]

        self.assertNotIn("billing_qualification", built)
        self.assertEqual(built["annual_revenue"], 1000000)
        self.assertNotIn("billing_qualification", save_profile.call_args.args[1])

        data_service._PROFILES["LEAD-1"] = {**built, "billing_qualification": {"tax_id": "sensitive"}}
        cached = build_prospect_profile.invoke({"prospect_id": "LEAD-1"})["prospect_profile"]

        self.assertNotIn("billing_qualification", cached)
        self.assertNotIn(
            "billing_qualification",
            data_service.get_profile_from_db("LEAD-1")["prospect_profile"],
        )

    def test_scoring_prompt_excludes_sensitive_fields_and_keeps_scoring_inputs(self):
        captured = {}

        def capture(messages):
            captured["messages"] = messages
            return SimpleNamespace(model_dump=lambda: {"score": 80})

        profile = {
            "prospect_id": "LEAD-1",
            "annual_revenue": 1000000,
            "tech_stack": ["Kafka"],
            "billing_qualification": {"tax_id": "sensitive"},
        }
        offering = {
            "required_tech_stack": ["Kafka"],
            "min_annual_revenue": 500000,
            "description": "Data platform",
        }
        with (
            patch.object(data_service, "fetch_tech_stack", return_value=["Kafka"]),
            patch("gtm_agent.gtm_agent._scoring_llm", SimpleNamespace(invoke=capture)),
        ):
            score_prospect.invoke({"prospect_profile": profile, "offering": offering})

        prompt = captured["messages"][1]["content"]
        self.assertNotIn("billing_qualification", prompt)
        self.assertNotIn("sensitive", prompt)
        offering_json, profile_json = prompt.split("\n\nProspect profile:\n", 1)
        self.assertEqual(json.loads(profile_json)["annual_revenue"], 1000000)
        self.assertEqual(json.loads(profile_json)["tech_stack"], ["Kafka"])
        self.assertEqual(json.loads(offering_json.removeprefix("Offering:\n"))["required_tech_stack"], ["Kafka"])


if __name__ == "__main__":
    unittest.main()
