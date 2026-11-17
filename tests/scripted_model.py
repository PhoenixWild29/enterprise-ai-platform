"""A Strands model that replays scripted turns, for offline tests."""
import json

from strands.models import Model


class ScriptedModel(Model):
    def __init__(self, turns, model_id="scripted-model"):
        self.turns, self.calls, self.model_id = list(turns), [], model_id

    def update_config(self, **kwargs):
        pass

    def get_config(self):
        return {"model_id": self.model_id}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError
        yield {}

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls.append({"system": system_prompt, "messages": messages, "tools": [t["name"] for t in (tool_specs or [])]})
        if not self.turns:
            raise RuntimeError("script exhausted")
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        yield {"messageStart": {"role": "assistant"}}
        if "tool" in turn or "tools" in turn:  # "tools": several tool uses in one model turn
            for i, (name, args) in enumerate(turn.get("tools") or [turn["tool"]]):
                yield {"contentBlockStart": {"start": {"toolUse": {"name": name, "toolUseId": f"tu-{len(self.calls)}-{i}"}}}}
                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}}
                yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": turn.get("text", "")}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": turn.get("in", 100), "outputTokens": turn.get("out", 20),
                                      "totalTokens": turn.get("in", 100) + turn.get("out", 20)}, "metrics": {"latencyMs": 1}}}

    def last_tool_result(self, call_index=-1):
        for block in reversed(self.calls[call_index]["messages"][-1]["content"]):
            if "toolResult" in block:
                return block["toolResult"]["content"][0]["text"]
        return None
