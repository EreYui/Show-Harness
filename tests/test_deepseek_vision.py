"""DeepSeek Piper profile and two-camera Chat Completions wire contract."""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from core.config import load_yaml, resolve_vlm_config
from core.sim.launch import enable_original_zero_shot_reasoning, make_vlm_client


ROOT = Path(__file__).resolve().parents[1]


class DeepSeekVisionTests(unittest.TestCase):
    def test_piper_deepseek_uses_gripper_aware_prompt(self):
        from scripts.run_piper_isaaclab_mvtoken import _prompt_path

        prompt = _prompt_path(ROOT / "prompts", "v3", "deepseek")
        self.assertEqual(prompt.name, "piper_deepseek.txt")
        rendered = prompt.read_text().format(
            task="Pick up the red cube", gripper_state="closed", recent_moves="MV_DOWN",
            tcp_position_m="0.292, 0.000, 0.025", finger_width_m="0.040",
            piper_phase="CUBE_HELD",
        )
        self.assertIn("Gripper: closed", rendered)
        self.assertIn("0.292, 0.000, 0.025", rendered)
        self.assertIn("0.040", rendered)
        self.assertIn("Current sensor phase: CUBE_HELD", rendered)
        self.assertIn("CUBE_HELD: output MV_UP", rendered)
        self.assertIn("AT_GRASP_HEIGHT: output GRASP", rendered)
        self.assertEqual(
            _prompt_path(ROOT / "prompts", "v3", "heyroute").name,
            "mvtoken_generator_lite.txt",
        )

    def test_pick_place_task_selects_long_horizon_prompt(self):
        from scripts.run_piper_isaaclab_mvtoken import _prompt_path

        prompt = _prompt_path(
            ROOT / "prompts", "v3", "deepseek", "pick_place_left"
        )
        self.assertEqual(prompt.name, "piper_deepseek_pick_place.txt")
        rendered = prompt.read_text().format(
            task="Place the red cube on the green target",
            gripper_state="closed",
            recent_moves="MV_UP",
            tcp_position_m="0.300, 0.020, 0.120",
            finger_width_m="0.040",
            piper_phase="CARRY_TO_TARGET",
            cube_position_m="0.300, 0.020, 0.110",
            target_position_m="0.300, 0.180, 0.000",
            home_position_m="0.292, 0.000, 0.144",
        )
        self.assertIn("Current sensor phase: CARRY_TO_TARGET", rendered)
        self.assertIn("CARRY_TO_TARGET: align TCP x/y", rendered)
        self.assertIn("target y is larger choose MV_LEFT", rendered)
        self.assertIn("RETREAT_UP", rendered)
        self.assertIn("RETURN_HOME_RIGHT: output MV_RIGHT", rendered)

    def test_deepseek_can_force_the_original_generic_prompt(self):
        from scripts.run_piper_isaaclab_mvtoken import _prompt_path

        prompt = _prompt_path(
            ROOT / "prompts",
            "v3",
            "deepseek",
            "pick_place_forward",
            profile="generic",
        )
        self.assertEqual(prompt.name, "mvtoken_generator_lite.txt")

    def test_piper_profile_sends_two_user_images_with_deepseek_parameters(self):
        config = load_yaml(ROOT / "configs/robot_piper_isaaclab.yaml")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-deepseek-key"}):
            vlm = resolve_vlm_config(config, backend="deepseek")
            client = make_vlm_client(None, {"vlm": vlm})

        self.assertEqual(vlm["model"], "deepseek-flash")
        self.assertEqual(client.base_url, "https://api.deepseek.com")
        self.assertEqual(client.session.headers["Authorization"], "Bearer test-deepseek-key")
        sent = []

        def capture(url, *, json, timeout):
            sent.append((url, json, timeout))
            return SimpleNamespace(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": "MV_UP"}}]},
            )

        client.session.post = capture
        front = np.zeros((8, 8, 3), dtype=np.uint8)
        wrist = np.ones((8, 8, 3), dtype=np.uint8) * 255
        response = client.complete_action_token("Return one action token", ["MV_UP"], front, wrist)

        self.assertEqual(response.token, "MV_UP")
        url, payload, timeout = sent[0]
        self.assertEqual(url, "https://api.deepseek.com/chat/completions")
        self.assertEqual(timeout, vlm["timeout_s"])
        self.assertEqual(payload["model"], "deepseek-flash")
        self.assertEqual(payload["max_tokens"], 128)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertNotIn("max_completion_tokens", payload)
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertNotIn("guided_choice", payload)
        self.assertEqual(payload["messages"][0]["role"], "user")
        parts = payload["messages"][0]["content"]
        self.assertEqual([part["type"] for part in parts], ["image_url", "image_url", "text"])
        self.assertTrue(parts[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertNotEqual(parts[0]["image_url"]["url"], parts[1]["image_url"]["url"])

    def test_missing_key_fails_before_paid_rollout(self):
        with self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
            make_vlm_client(None, {"vlm": {"provider": "deepseek", "api_key": "EMPTY"}})

    def test_zero_shot_bad_answer_gets_strict_token_retry(self):
        from core.vlm.vlm_client import VLMClient

        client = VLMClient(
            base_url="https://api.deepseek.com",
            model="deepseek-flash",
            api_key="test-key",
            timeout_s=10,
            max_tokens=128,
            temperature=0,
            provider="deepseek",
            thinking_mode="disabled",
        )
        sent = []

        def capture(url, *, json, timeout):
            sent.append(json)
            reply = "I cannot decide." if len(sent) == 1 else "MV_UP"
            return SimpleNamespace(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": reply}}]},
            )

        client.session.post = capture
        response = client.complete_action_token(
            "Choose one action", ["MV_UP"], np.zeros((8, 8, 3), dtype=np.uint8)
        )
        self.assertEqual(response.token, "MV_UP")
        self.assertEqual(len(sent), 2)
        self.assertIn("Critical output format", sent[1]["messages"][0]["content"][-1]["text"])

    def test_model_list_check_catches_unavailable_vision_model(self):
        from core.vlm.vlm_client import VLMClient

        client = VLMClient(
            base_url="https://api.deepseek.com",
            model="deepseek-flash",
            api_key="test-key",
            timeout_s=10,
            max_tokens=128,
            temperature=0,
            provider="deepseek",
        )

        def listed(ids):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"data": [{"id": model} for model in ids]},
            )

        client.session.get = lambda url, timeout: listed(["deepseek-flash"])
        client.verify_model_available()
        client.session.get = lambda url, timeout: listed(["deepseek-v4-flash"])
        with self.assertRaisesRegex(RuntimeError, "not listed for this API key"):
            client.verify_model_available()

    def test_original_zero_shot_explicitly_enables_deepseek_thinking(self):
        config = load_yaml(ROOT / "configs/robot_piper_isaaclab.yaml")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-deepseek-key"}):
            vlm = resolve_vlm_config(config, backend="deepseek")
        cfg = {"vlm": vlm}
        enable_original_zero_shot_reasoning(cfg)
        client = make_vlm_client(None, cfg)
        self.assertTrue(vlm["reasoning_cot"])
        self.assertEqual(vlm["thinking_mode"], "enabled")
        self.assertEqual(vlm["reasoning_effort"], "high")
        self.assertGreaterEqual(vlm["cot_max_tokens"], 1024)

        sent = []
        client.session.post = lambda url, *, json, timeout: (
            sent.append(json)
            or SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "choices": [{
                        "message": {
                            "reasoning_content": "The TCP is high, so descend.",
                            "content": "FINAL: MV_DOWN",
                        }
                    }]
                },
            )
        )
        response = client.complete_text(
            "choose",
            np.zeros((8, 8, 3), dtype=np.uint8),
            strip_reasoning=False,
        )
        self.assertEqual(sent[0]["thinking"], {"type": "enabled"})
        self.assertEqual(sent[0]["reasoning_effort"], "high")
        self.assertNotIn("temperature", sent[0])
        self.assertIn("The TCP is high", response.raw_text)
        self.assertIn("FINAL: MV_DOWN", response.raw_text)
        self.assertIn("reasoning_content", response.payload)

        sent.clear()
        client.session.post = lambda url, *, json, timeout: (
            sent.append(json)
            or SimpleNamespace(
                status_code=200,
                json=lambda: {"choices": [{"message": {"content": '{"ok": true}'}}]},
            )
        )
        client.complete_json(
            "plan",
            np.zeros((8, 8, 3), dtype=np.uint8),
            chat_template_kwargs={"enable_thinking": False, "thinking": False},
        )
        self.assertEqual(sent[0]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", sent[0])


if __name__ == "__main__":
    unittest.main()
