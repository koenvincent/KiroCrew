"""Business adapters for the Jev decision seam.

``skills.select`` picks the skill a message loads: an exact offered key selects
one, an explicit no-skill answer selects none, and a refusal keeps trigger
matching. ``message.steer`` decides whether a message sent into a RUNNING turn
steers it or queues for the next one, and a refusal takes the steer path the
composer has always defaulted to. ``memory.recall`` decides which of the memories
vector similarity recalled are worth their place in the prompt, and a refusal
injects the similarity top-k unchanged. ``tool.risk`` decides nothing and only
annotates. Each adapter's refusal is the shipped behaviour, never a third outcome.

The core package owns transport, sampling and diagnostic logging.
"""

# Skill keys and memory ids must remain exact when passed back to the loader or
# matched against a search result. Drop a key exceeding this bound rather than
# truncating it into a different identifier.
MAX_KEY_CHARS = 120
