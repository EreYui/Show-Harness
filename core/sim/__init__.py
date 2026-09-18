"""Simulator-side task glue and rollout runners.

Everything in this package talks to a SIMULATOR; nothing in it touches real hardware.
``core/`` proper keeps the shared pieces (``config``, ``images``, ``episode_logger``,
``v0_types``, ...) and the real-robot runners (``real_runner``, ``mvtoken_runner``,
``dual_runner``, ``teleop*``, ``franka/``, ``piper/``).

Two simulator engines are wired. Isaac Lab has both the RoboLab/Franka task
and a standalone single-arm Piper scene. Task modules construct environments
and expose observation/success/TCP/gripper accessors and axis probes:

===========  ==========================  ===============================
Simulator    task module                 MVTOKEN runner
===========  ==========================  ===============================
ManiSkill    ``maniskill_task``          ``mvtoken_maniskill_runner``
RoboLab      ``robolab_task``            ``mvtoken_robolab_runner``
Piper/Isaac  ``piper_isaaclab_task``      ``mvtoken_robolab_runner`` (backend injected)
===========  ==========================  ===============================

Plus ``maniskill_scenes`` -- the ManiSkill scene table, one row per environment (see its
docstring).

Nothing is re-exported here on purpose: importing a simulator's task module pulls that
simulator's heavyweight dependencies in, and the two cannot generally coexist in one
interpreter (ManiSkill wants its own conda env, RoboLab its own Python 3.11 venv).
Import the specific module you need::

    from core.sim.maniskill_task import make_maniskill_task
"""
