import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain.agents import create_agent as build_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ["LANGSMITH_TRACING"] = "false"

with patch("langchain.agents.create_agent") as create_agent:
    create_agent.return_value = SimpleNamespace(invoke=Mock())
    gtm_module = importlib.import_module("gtm_agent.gtm_agent")


class ToolCallingChatModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class GTMAgentTests(unittest.TestCase):
    def test_agent_uses_only_custom_tools_and_application_prompt(self):
        tools = gtm_module.create_agent.call_args.kwargs["tools"]

        self.assertEqual(
            [tool.name for tool in tools],
            [
                "lookup_offering",
                "build_prospect_profile",
                "get_prospect",
                "send_prospect_email",
                "score_prospect",
                "update_prospect_info",
                "get_current_rep",
            ],
        )
        self.assertEqual(
            gtm_module.create_agent.call_args.kwargs["system_prompt"],
            gtm_module.SYSTEM_PROMPT,
        )
        self.assertNotIn("middleware", gtm_module.create_agent.call_args.kwargs)

    def test_get_current_rep_uses_runtime_user_id(self):
        rep = {"rep_id": "rep_jchen", "name": "Jamie Chen"}
        model = ToolCallingChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "get_current_rep", "args": {}, "id": "rep", "type": "tool_call"}
                    ],
                ),
                AIMessage(content="Done"),
            ]
        )
        agent = build_agent(
            model=model,
            tools=gtm_module.create_agent.call_args.kwargs["tools"],
            system_prompt=gtm_module.SYSTEM_PROMPT,
        )

        with patch.object(gtm_module.data_service, "get_rep", return_value=rep) as get_rep:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "Identify me"}]},
                config={"metadata": {"user_id": "rep_jchen"}},
            )

        get_rep.assert_called_once_with("rep_jchen")
        self.assertIn('"rep_id": "rep_jchen"', result["messages"][2].content)
        self.assertEqual(gtm_module.get_current_rep.args, {})

    def test_send_prospect_email_uses_runtime_sender(self):
        rep = {"rep_id": "rep_jchen", "name": "Jamie Chen", "email": "jamie@example.com"}
        prospect = {"name": "Alex Lee", "email": "alex@example.com"}
        model = ToolCallingChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "send_prospect_email",
                            "args": {"prospect": prospect, "subject": "Intro", "body": "Hello"},
                            "id": "email",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Done"),
            ]
        )
        agent = build_agent(
            model=model,
            tools=gtm_module.create_agent.call_args.kwargs["tools"],
            system_prompt=gtm_module.SYSTEM_PROMPT,
        )

        with patch.object(gtm_module.data_service, "get_rep", return_value=rep) as get_rep:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "Email Alex"}]},
                config={"metadata": {"user_id": "rep_jchen"}},
            )

        get_rep.assert_called_once_with("rep_jchen")
        self.assertIn('"status": "sent"', result["messages"][2].content)
        self.assertIn('"from": "jamie@example.com"', result["messages"][2].content)
        self.assertIn('"to": "alex@example.com"', result["messages"][2].content)
        self.assertEqual(
            set(gtm_module.send_prospect_email.args),
            {"prospect", "subject", "body", "from_rep"},
        )

    def test_run_agent_invokes_with_messages_and_reads_messages_result(self):
        response = AIMessage(content="Done")
        invoke = gtm_module.gtm_agent.invoke
        invoke.return_value = {"messages": [response]}

        result = gtm_module.run_agent("Check an offering", user_id="rep_jchen")

        invoke.assert_called_once()
        inputs = invoke.call_args.args[0]
        self.assertEqual(
            inputs,
            {"messages": [{"role": "user", "content": "Check an offering"}]},
        )
        self.assertEqual(result["reply"], "Done")
        self.assertEqual(result["messages"], [response])


if __name__ == "__main__":
    unittest.main()
