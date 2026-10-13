import os
import re
import unittest
from types import SimpleNamespace

HAS_OPENAI_API_KEY = bool(os.environ.get("OPENAI_API_KEY"))
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent.gtm_agent import SYSTEM_PROMPT, run_agent, send_prospect_email


class EmailAttachmentGuidanceTests(unittest.TestCase):
    def test_prompt_discloses_attachment_limitation_for_email_and_reply(self):
        self.assertIn("You cannot attach files.", SYSTEM_PROMPT)
        self.assertIn("outbound email body or in your response", SYSTEM_PROMPT)
        self.assertIn("must be shared separately", SYSTEM_PROMPT)
        self.assertIn("files/attachments are NOT supported", send_prospect_email.description)

    def test_ordinary_email_tool_send_preserves_requested_body(self):
        body = "Hi Sofia, are you available for an onsite workshop?"
        result = send_prospect_email.func(
            prospect={"name": "Sofia Rossi", "email": "sofia@example.com"},
            subject="Workshop availability",
            body=body,
            runtime=SimpleNamespace(config={"metadata": {}}),
            from_rep={"name": "Sara Brown", "email": "sara@example.com"},
        )

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["body"], body)
        self.assertNotRegex(result["body"], re.compile(r"(?i)attachment|attached|share separately"))


@unittest.skipUnless(HAS_OPENAI_API_KEY, "requires OPENAI_API_KEY for live agent coverage")
class LiveEmailAttachmentTests(unittest.TestCase):
    def _email_call_and_reply(self, request, user_id):
        result = run_agent(request, user_id=user_id, environment="test")
        calls = [
            call
            for message in result["messages"]
            for call in getattr(message, "tool_calls", [])
            if call.get("name") == "send_prospect_email"
        ]
        self.assertEqual(len(calls), 1)
        return calls[0]["args"], result["reply"]

    def test_pricing_deck_request_discloses_separate_sharing(self):
        email, reply = self._email_call_and_reply(
            "Email LEAD-50003 to send over our pricing deck.", "rep_sbrown"
        )

        self.assertNotRegex(email["body"], re.compile(r"(?i)\b(attached|included|enclosed)\b"))
        self.assertRegex(reply, re.compile(r"(?i)\b(not attached|cannot attach|unable to attach)\b"))
        self.assertRegex(reply, re.compile(r"(?i)\b(shared|share|send) separately\b"))

    def test_ordinary_email_has_no_attachment_warning(self):
        email, reply = self._email_call_and_reply(
            "Send LEAD-50005 an email asking about their availability for an onsite workshop.",
            "rep_rgarcia",
        )

        self.assertRegex(email["body"], re.compile(r"(?i)availability|available"))
        self.assertNotRegex(reply, re.compile(r"(?i)attachment|attached|share separately"))
