from __future__ import annotations

import unittest

from src.eval_core.evaluate_toolcall_accuracy import payload_to_commands
from src.protocols.toolcall import extract_first_json, normalize_payload, validate_payload


class ToolcallProtocolTests(unittest.TestCase):
    def test_extract_pure_json(self) -> None:
        text = '{"commands":[{"action":"wait"}]}'
        obj = extract_first_json(text)
        self.assertIsInstance(obj, dict)
        self.assertIn("commands", obj)

    def test_extract_json_from_markdown_block(self) -> None:
        text = "```json\n{\"commands\":[{\"action\":\"wait\"}]}\n```"
        obj = extract_first_json(text)
        self.assertEqual(obj["commands"][0]["action"], "wait")

    def test_extract_json_with_dirty_text(self) -> None:
        text = "前置说明...\n{\"commands\":[{\"action\":\"wait\"}]}\n后置说明..."
        obj = extract_first_json(text)
        self.assertEqual(obj["commands"][0]["action"], "wait")

    def test_extract_partial_commands_from_truncated_json(self) -> None:
        text = (
            '{"commands": ['
            '{"action": "move_ee", "pos": [0.65, 0.0, 0.12], "quat": [0, 1, 0, 0]}, '
            '{"action": "close_gripper", "position": 0.0}, '
            '{"action": "move_ee", "pos": [0.65, 0.0, 0.18], "quat": [0, 1'
        )
        obj = extract_first_json(text)
        self.assertEqual(len(obj["commands"]), 2)
        self.assertEqual(obj["commands"][0]["action"], "move_ee")
        self.assertEqual(obj["commands"][1]["action"], "close_gripper")

    def test_wait_valid_form(self) -> None:
        payload = {"commands": [{"action": "wait"}]}
        cmds = validate_payload(payload)
        self.assertEqual(len(cmds), 1)
        self.assertEqual(cmds[0]["action"], "wait")

    def test_wait_invalid_steps_form(self) -> None:
        payload = {"commands": [{"action": "wait", "steps": 10}]}
        with self.assertRaises(ValueError):
            validate_payload(payload)

    def test_move_ee_missing_required_field(self) -> None:
        payload = {"commands": [{"action": "move_ee", "pos": [0.6, 0.0, 0.2]}]}
        with self.assertRaises(ValueError):
            validate_payload(payload)

    def test_eval_and_execution_consistent_for_same_payload(self) -> None:
        valid_payload = '{"commands":[{"action":"wait"}]}'
        invalid_payload = '{"commands":[{"action":"wait","steps":5}]}'

        eval_cmds = payload_to_commands(valid_payload)
        exec_cmds = validate_payload(valid_payload, policy="execution")
        self.assertEqual(eval_cmds, exec_cmds)

        with self.assertRaises(ValueError):
            payload_to_commands(invalid_payload)
        with self.assertRaises(ValueError):
            validate_payload(invalid_payload, policy="execution")

    def test_normalize_payload_keeps_legacy_shapes(self) -> None:
        single = normalize_payload({"action": "wait"})
        as_list = normalize_payload([{"action": "wait"}])
        wrapped = normalize_payload({"commands": [{"action": "wait"}]})
        self.assertEqual(single, as_list)
        self.assertEqual(as_list, wrapped)


if __name__ == "__main__":
    unittest.main()
