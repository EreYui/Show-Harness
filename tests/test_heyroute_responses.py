"""HeyRoute Responses transport for Piper's two-camera action loop."""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.config import load_yaml, resolve_vlm_config
from core.sim.launch import make_vlm_client


ROOT = Path(__file__).resolve().parents[1]


def completed(text: str):
    """Match the response shape returned by the user's HeyRoute text probe."""
    return SimpleNamespace(
        status_code=200,
        json=lambda: {
            "object": "response",
            "status": "completed",
            "output": [
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": text, "annotations": []}
                ]}
            ],
        },
    )


class HeyRouteResponsesTests(unittest.TestCase):
    def client(self):
        config = load_yaml(ROOT / "configs/robot_piper_isaaclab.yaml")
        with patch.dict(os.environ, {"HEYROUTE_API_KEY": "test-heyroute-key"}):
            vlm = resolve_vlm_config(config, backend="heyroute")
            client = make_vlm_client(None, {"vlm": vlm})
        return client, vlm

    def test_two_camera_action_request_uses_responses_wire_format(self):
        client, vlm = self.client()
        sent = []

        def capture(url, *, json, timeout):
            sent.append((url, json, timeout))
            return completed("MV_UP")

        client.session.post = capture
        front = np.zeros((8, 8, 3), dtype=np.uint8)
        wrist = np.ones((8, 8, 3), dtype=np.uint8) * 255
        response = client.complete_action_token("Return one action token", ["MV_UP"], front, wrist)

        self.assertEqual(response.token, "MV_UP")
        self.assertEqual(client.session.headers["Authorization"], "Bearer test-heyroute-key")
        url, payload, timeout = sent[0]
        self.assertEqual(url, "https://heyroute.ai/v1/responses")
        self.assertEqual(timeout, vlm["timeout_s"])
        self.assertEqual(payload["model"], "gpt-5.6-sol")
        self.assertEqual(payload["max_output_tokens"], 256)
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        self.assertIs(payload["store"], False)
        self.assertNotIn("messages", payload)
        self.assertNotIn("temperature", payload)
        self.assertNotIn("guided_choice", payload)
        parts = payload["input"][0]["content"]
        self.assertEqual([part["type"] for part in parts], ["input_image", "input_image", "input_text"])
        self.assertTrue(parts[0]["image_url"].startswith("data:image/png;base64,"))
        self.assertTrue(parts[1]["image_url"].startswith("data:image/png;base64,"))
        self.assertNotEqual(parts[0]["image_url"], parts[1]["image_url"])

    def test_non_token_answer_gets_strict_retry(self):
        client, _ = self.client()
        sent = []

        def capture(url, *, json, timeout):
            sent.append(json)
            return completed("I cannot decide" if len(sent) == 1 else "MV_UP")

        client.session.post = capture
        response = client.complete_action_token(
            "Choose one action", ["MV_UP"], np.zeros((8, 8, 3), dtype=np.uint8)
        )
        self.assertEqual(response.token, "MV_UP")
        self.assertEqual(len(sent), 2)
        self.assertIn("Critical output format", sent[1]["input"][0]["content"][-1]["text"])

    def test_missing_key_fails_before_simulator_launch(self):
        with self.assertRaisesRegex(ValueError, "HEYROUTE_API_KEY"):
            make_vlm_client(None, {"vlm": {
                "provider": "openai", "wire_api": "responses",
                "api_key_env": "HEYROUTE_API_KEY", "api_key": "EMPTY",
            }})

    def test_empty_success_body_is_not_treated_as_an_action(self):
        client, _ = self.client()
        client.max_retries = 0
        client.session.post = lambda url, *, json, timeout: completed("")
        with self.assertRaisesRegex(RuntimeError, "no output_text"):
            client.complete_action_token("Choose one action", ["MV_UP"], np.zeros((8, 8, 3), dtype=np.uint8))

    def test_insufficient_quota_fails_without_retrying(self):
        client, _ = self.client()
        calls = []

        def no_quota(url, *, json, timeout):
            calls.append(url)
            return SimpleNamespace(status_code=402, text="insufficient quota")

        client.session.post = no_quota
        with self.assertRaisesRegex(RuntimeError, "HTTP 402"):
            client.complete_action_token("Choose one action", ["MV_UP"], np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
