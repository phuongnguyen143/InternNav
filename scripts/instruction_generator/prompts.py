"""LLM prompt templates for instruction generation and summarization."""

TRAJECTORY_PROMPT_BEFORE = (
    "Assume you are a robot designed for navigation. " "You are provided with captured image sequences:"
)

TRAJECTORY_PROMPT_AFTER = """Based on this image sequence, please describe the navigation trajectory of the robot.

** Important instructions **:
- The instruction is ONE sentence or two at most, but should cover the full path from start to end.
- Mention key landmarks, turns, and actions in order.
- Do NOT use bullet points or numbering.

### Example 1:
Instruction:
Walk straight ahead, passing the black armchair on your left and the white curtains on your right, until you reach the wooden staircase at the end of the hallway. Stop at the base of the stairs.

###  Example 2:
Instruction:
Turn right at the black chair, continue toward the white curtains, and stop beside the first curtain on your left.

### Example 3:
Instruction:
Walk forward past the wooden dresser on your left and the patterned ottoman on your right until you reach the white door at the end of the hallway. Stop in front of the door.
"""

SUMMARIZE_PROMPT = """You are a navigation instruction summarizer.

Below are sequential navigation instructions describing parts of a single trajectory:
{instructions}

Summarize all of them into ONE fluent, long-horizon navigation instruction.
- Cover the full path from start to end
- Mention key landmarks and turns in order
- Natural language, one sentence or two at most
- No bullet points, no numbering

Output only the final instruction, nothing else.
"""
