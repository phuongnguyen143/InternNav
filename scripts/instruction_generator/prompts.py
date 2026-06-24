"""LLM prompt templates for instruction generation and summarization."""

TRAJECTORY_PROMPT_BEFORE = "There are a sequence of egocentric images of a robot navigating through an office environment captured by a camera on the robot head."

TRAJECTORY_PROMPT_AFTER = """Based on this image sequence, please briefly describe the navigation trajectory of the robot using 1 or at most 2 natual sentences. Focus only on the main action: turn left, turn right, go forward pivot to key landmark e.g: tree, road, stair, building, etc. in order. Remember to investigate carefully between 2 consecutive images to determine the action robot takes.
"""

SUMMARIZE_PROMPT = """You are a navigation instruction summarizer.Below are sequential navigation instructions describing parts of a single trajectory: {instructions}.
Summarize all of them into one fluent, long-horizon navigation instruction. Keep only the necessary information to navigate, drop all the not relevent things.
"""
