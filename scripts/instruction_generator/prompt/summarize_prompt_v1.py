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