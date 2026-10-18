"""Bedtime story loop using Claude Agent SDK subagents.

Python decides when to rewrite. Each step asks one subagent, through the
Agent tool, to write, score, or revise the story.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, ClaudeSDKClient

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)

PASS_SCORE = 7
MAX_REWRITES = 2

AGENTS = {
    "storyteller": AgentDefinition(
        description="Writes a gentle bedtime story for children ages 5 to 10.",
        prompt=(
            "You are a children's storyteller. Write a bedtime story for ages 5–10. "
            "Use simple words, a beginning, a middle, and a happy ending. "
            "Include a soft lesson about kindness, courage, or friendship. "
            "Do not write anything scary or violent. Reply with the story only."
        ),
        tools=[],
        model="inherit",
    ),
    "judge": AgentDefinition(
        description="Scores a children's bedtime story from 1 to 10.",
        prompt=(
            "You are a children's book reviewer. Score the story from 1 to 10 for "
            "age fit, clear language, structure, bedtime tone, and a positive lesson. "
            "Reply with JSON only, using keys score, problems, and suggestions. "
            "problems is a list of strings. suggestions is one string."
        ),
        tools=[],
        model="inherit",
    ),
    "rewriter": AgentDefinition(
        description="Rewrites a bedtime story using the reviewer's notes.",
        prompt=(
            "You rewrite bedtime stories for ages 5–10. Fix every problem and use "
            "the suggestions. Keep the same characters, a gentle tone, and a happy ending. "
            "Reply with the full story only."
        ),
        tools=[],
        model="inherit",
    ),
}


def on_stderr(line: str) -> None:
    if "unrecognized_model" in line or "claude.ai connectors are disabled" in line:
        return
    print(line, end="" if line.endswith("\n") else "\n", flush=True)


def model_env() -> tuple[dict[str, str], str]:
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    token = os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip()
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    model = os.getenv("OLLAMA_MODEL", "").strip() or "gemma4:31b-cloud"
    if not api_key and not (base_url and token):
        sys.exit("Set ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN, or set ANTHROPIC_API_KEY.")
    return {
        **os.environ,
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_AUTH_TOKEN": token,
        "ANTHROPIC_API_KEY": api_key,
        "OLLAMA_MODEL": model,
    }, model


def parse_review(text: str) -> tuple[int, str, str]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The judge did not return JSON.")
    payload = json.loads(text[start : end + 1])
    score = int(re.search(r"\d+", str(payload["score"])).group())
    problems = payload.get("problems") or []
    if isinstance(problems, list):
        problems = "\n".join(f"- {item}" for item in problems if str(item).strip())
    suggestions = payload.get("suggestions") or ""
    if isinstance(suggestions, list):
        suggestions = " ".join(str(item) for item in suggestions)
    return score, str(problems).strip(), str(suggestions).strip()


async def ask(name: str, task: str, env: dict[str, str], model: str) -> str:
    """Delegate one task to a subagent and return its answer."""
    print(f"\n[{name}]")
    subagent_text: list[str] = []
    lead_text: list[str] = []
    announced = False
    options = ClaudeAgentOptions(
        cwd=ROOT,
        model=model,
        permission_mode="bypassPermissions",
        system_prompt=(
            f'Call the Agent tool once with subagent_type "{name}". '
            "Pass the user message as the subagent prompt. "
            "When it finishes, reply with only the subagent's answer."
        ),
        agents={name: AGENTS[name]},
        tools=["Agent"],
        allowed_tools=["Agent"],
        skills=[],
        setting_sources=[],
        max_turns=4,
        forward_subagent_text=True,
        env=env,
        stderr=on_stderr,
    )
    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt=task)
        async for message in client.receive_response():
            if type(message).__name__ != "AssistantMessage":
                continue
            bucket = subagent_text if getattr(message, "parent_tool_use_id", None) else lead_text
            for block in message.content:
                if type(block).__name__ == "ToolUseBlock" and block.name in {"Agent", "Task"} and not announced:
                    print("  delegated")
                    announced = True
                elif type(block).__name__ == "TextBlock" and block.text.strip():
                    bucket.append(block.text)
    answer = "".join(subagent_text).strip() or "".join(lead_text).strip()
    print(answer)
    return answer


async def run(topic: str, env: dict[str, str], model: str) -> None:
    story = await ask("storyteller", f"Write a bedtime story about: {topic}", env, model)
    review_text = await ask("judge", f"Review this story:\n\n{story}", env, model)
    score, problems, suggestions = parse_review(review_text)
    rewrites = 0
    while score <= PASS_SCORE and rewrites < MAX_REWRITES:
        story = await ask(
            "rewriter",
            f"Story:\n{story}\n\nProblems:\n{problems}\n\nSuggestions:\n{suggestions}",
            env,
            model,
        )
        rewrites += 1
        if rewrites >= MAX_REWRITES:
            break
        review_text = await ask("judge", f"Review this story:\n\n{story}", env, model)
        score, problems, suggestions = parse_review(review_text)

    print("\n" + "=" * 50)
    print(f"Score: {score}/10")
    print(problems or "(no problems)")
    if suggestions:
        print(suggestions)
    print("\nFinal story\n")
    print(story)
    if rewrites == 0:
        print("\nThe first draft scored above the threshold, so it was kept.")
    elif rewrites >= MAX_REWRITES and score <= PASS_SCORE:
        print("\nThe last rewrite was not scored again, because the rewrite limit was reached.")


async def main() -> None:
    env, model = model_env()
    topics = [arg.strip() for arg in sys.argv[1:] if arg.strip()]
    print(f"Story teller  |  model: {model}")
    print("Subagents: storyteller, judge, rewriter")
    if not topics:
        print("Enter a topic. Type exit to quit.\n")
    while True:
        if topics:
            topic = topics.pop(0)
            print(f"\nTopic: {topic}")
        else:
            try:
                topic = input("\nTopic: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
        if not topic or topic.lower() in {"exit", "quit", "q"}:
            break
        try:
            await run(topic, env, model)
        except Exception as exc:
            print(f"\nThe story run failed: {exc}")
        if not topics and len(sys.argv) > 1:
            break


if __name__ == "__main__":
    asyncio.run(main())
