"""Coordinate-system prompt tool.

When disabled, this tool returns the controller prompt unchanged. When enabled, it
overrides only the controller prompt's ``DIRECTION:`` and ``REMARK:`` sections in
memory before the prompt template is passed to the controller agent. The source
``prompts/controller.txt`` file is not modified.

The active prompt keeps atomic actions axis-only, then adds a minimal camera-to-axis
calibration so the VLM can translate raw image evidence into robot-frame displacement
without reusing the base prompt's full image-direction policy.
"""
from __future__ import annotations

from plugins.prompt_text import fragment


class CoordsPlugin:
    """Prompt-only coordinate-system override for the controller role."""

    def __init__(
        self,
        enabled: bool = False,
        positive_y_token: str = "MV_RIGHT",
        positive_y_image_edge: str = "right",
    ) -> None:
        self.enabled = bool(enabled)
        self.positive_y_token = str(positive_y_token).strip().upper()
        if self.positive_y_token not in {"MV_LEFT", "MV_RIGHT"}:
            raise ValueError("positive_y_token must be MV_LEFT or MV_RIGHT")
        self.positive_y_image_edge = str(positive_y_image_edge).strip().lower()
        if self.positive_y_image_edge not in {"left", "right"}:
            raise ValueError("positive_y_image_edge must be left or right")

    def apply(self, prompt_template: str) -> str:
        """Return ``prompt_template`` unchanged or with coordinate-mode sections.
        The replacement blocks live in the co-located coords.txt."""
        if not self.enabled:
            return prompt_template
        direction = fragment(__file__, "coords.txt", "direction")
        if self.positive_y_token != "MV_RIGHT":
            direction = direction.replace(
                "- MV_RIGHT -> move along +y\n- MV_LEFT  -> move along -y",
                "- MV_LEFT  -> move along +y\n- MV_RIGHT -> move along -y",
            )
        negative_edge = "left" if self.positive_y_image_edge == "right" else "right"
        negative_token = (
            "MV_LEFT" if self.positive_y_token == "MV_RIGHT" else "MV_RIGHT"
        )
        direction = direction.replace(
            "- right in the image -> +y (MV_RIGHT) ; left -> -y (MV_LEFT)",
            f"- {self.positive_y_image_edge} in the image -> +y "
            f"({self.positive_y_token}) ; {negative_edge} -> -y ({negative_token})",
        )
        remark = fragment(__file__, "coords.txt", "remark")
        # The dual controller has separate wrist/front camera rules. Preserve those
        # per-view calibrations (EgoPlugin may rewrite the wrist depth lines next) and
        # prepend only the robot-axis legend. A single camera mapping here would
        # incorrectly assume that the front and eye-in-hand cameras share orientation.
        if "DIRECTION (per arm)" in prompt_template:
            remark = remark.replace("{mem_text_rules}", "{mem_rules}")
            axis_legend = direction.split(
                "Camera-to-axis calibration", 1
            )[0].strip().replace("DIRECTION:", "ROBOT BASE AXES:", 1)
            axis_legend += (
                "\nUse the calibrated per-view DIRECTION rules below to infer these "
                "axis displacements."
            )
            prompt = prompt_template.replace(
                "DIRECTION (per arm)",
                axis_legend + "\n\nDIRECTION (per arm)",
                1,
            )
            return _replace_section(
                prompt, "ATTENTION:", "GRIPPER (per arm):", remark
            )
        prompt = _replace_section(prompt_template, "DIRECTION:", "REMARK:", direction)
        return _replace_section(prompt, "REMARK:", "GRIPPER:", remark)


def _replace_section(
    prompt: str, start_marker: str, end_marker: str, replacement: str
) -> str:
    start = prompt.find(start_marker)
    if start < 0:
        return prompt
    end = prompt.find(end_marker, start + len(start_marker))
    if end < 0:
        return prompt
    before = prompt[:start].rstrip()
    after = prompt[end:].lstrip()
    return before + "\n\n" + replacement.strip() + "\n\n" + after
