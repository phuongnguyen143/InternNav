"""LLM prompt templates for instruction generation and summarization."""

TRAJECTORY_PROMPT_BEFORE = (
    "Assume you are a robot designed for navigation. "
    "You are provided with captured egocentric image sequences:"
)

TRAJECTORY_PROMPT_AFTER = """Based on this image sequence, please describe the navigation trajectory of the robot.

** Important instructions **:
- The instruction is ONE sentence or two at most, but should cover the full path from start to end.
- Mention key actions, e.g: turn left, turn right, go forward pivot to key landmark e.g: tree, road, stair, building, etc. in order.
- Do NOT use bullet points or numbering.
"""

SUMMARIZE_PROMPT = """You are a navigation instruction summarizer.Below are sequential navigation instructions describing parts of a single trajectory: {instructions}.
Summarize all of them into one fluent, long-horizon navigation instruction. Keep only the necessary information to navigate, drop all the not relevent things.
"""
