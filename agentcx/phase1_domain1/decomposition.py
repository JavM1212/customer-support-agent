"""
AgentCX Phase 1 — TS 1.6: Prompt chaining vs dynamic decomposition.

prompt_chain:      developer hardcodes steps — predictable, auditable
dynamic_decompose: Claude decides steps at runtime — flexible, but risks attention dilution
"""

import json
import os
from dotenv import load_dotenv
import anthropic

from agentcx.phase1_domain1.tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS
from agentcx.phase1_domain1.hooks import AgentState, pre_tool_gate, on_post_tool_use

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ─────────────────────────────────────────────
# Shared: single-turn API call (no tools)
# ─────────────────────────────────────────────

def call(system: str, user: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


# ─────────────────────────────────────────────
# Shared: agentic loop with tools
# ─────────────────────────────────────────────

def run_with_tools(system: str, user: str, tools: list) -> str:
    messages = [{"role": "user", "content": user}]
    state = AgentState()

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return response.content[0].text

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                gate_error = pre_tool_gate(block.name, block.input, state)
                if gate_error:
                    result = gate_error
                else:
                    result = TOOL_FUNCTIONS[block.name](**block.input)
                    print(f"  [tool] {block.name}({block.input}) → {result}")
                on_post_tool_use(block.name, result, state)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})


# ─────────────────────────────────────────────
# Pattern 1: Prompt Chaining
# Steps are hardcoded by the developer.
# ─────────────────────────────────────────────

def prompt_chain(user_message: str) -> str:
    print("\n── Prompt Chaining ──")

    # Step 1: Classify the request
    print("[step 1] Classifying request...")
    classification = call(
        system="Classify the customer request. Reply with ONE word: refund, inquiry, or complaint.",
        user=user_message,
    )
    print(f"  → {classification.strip()}")

    # Step 2: Look up customer and order
    print("[step 2] Looking up customer and order...")
    lookup = run_with_tools(
        system=(
            "Look up the customer and order. "
            "Return ONLY JSON with: customer_id, customer_name, order_id, item, amount, order_status."
        ),
        user=user_message,
        tools=[t for t in TOOL_DEFINITIONS if t["name"] in ("get_customer", "get_order")],
    )
    print(f"  → {lookup}")

    # Step 3: Process refund if classified as such
    if "refund" in classification.lower():
        print("[step 3] Processing refund...")
        resolution = run_with_tools(
            system="Process the refund based on the case details provided.",
            user=f"Case: {lookup}\nRequest: {user_message}",
            tools=[t for t in TOOL_DEFINITIONS if t["name"] == "process_refund"],
        )
    else:
        resolution = call(
            system="You are a customer support agent. Answer the customer's question.",
            user=f"Case: {lookup}\nRequest: {user_message}",
        )

    return resolution


# ─────────────────────────────────────────────
# Pattern 2: Dynamic Decomposition
# Claude decides the steps at runtime.
# ─────────────────────────────────────────────

def dynamic_decompose(user_message: str) -> str:
    print("\n── Dynamic Decomposition ──")
    return run_with_tools(
        system=(
            "You are a customer support agent. "
            "Use the available tools to fully resolve the customer's request. "
            "Decide which tools to call and in what order based on what the request requires."
        ),
        user=user_message,
        tools=TOOL_DEFINITIONS,
    )


# ─────────────────────────────────────────────
# Run both and compare
# ─────────────────────────────────────────────

if __name__ == "__main__":
    message = "I'm customer C001. I'd like a refund for order ORD-100. It arrived damaged."

    print("=== TS 1.6: Prompt Chaining vs Dynamic Decomposition ===")

    result_chain = prompt_chain(message)
    print(f"\n[chaining] Final answer:\n{result_chain}")

    print("\n" + "="*50)

    result_dynamic = dynamic_decompose(message)
    print(f"\n[dynamic] Final answer:\n{result_dynamic}")
