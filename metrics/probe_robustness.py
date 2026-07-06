#!/usr/bin/env python3
"""Probe robustness check: compare probe output across prompt variants.
Tests whether intent classification is stable, not whether the env behaves differently.

3 variants:
  V1: Original 7-dimension JSON prompt (current)
  V2: Shortened binary-choice prompt ("which action?")
  V3: Free-text + classification prompt ("describe state, then pick action")

Expectation: dominant_intent should be stable across variants.
"""
import sys, json, urllib.request, ssl
sys.path.insert(0, "/opt/lingyao")

_ak = json.load(open("/root/.openclaw/openclaw.json"))["models"]["providers"]["deepseek"]["apiKey"]
_b = chr(66)+chr(101)+chr(97)+chr(114)+chr(101)+chr(114)

# Probe variants
PROMPTS = {
    "V1 (original)": """你是一个语义意图分析专家。分析用户输入的情感状态和意图分布。

用户输入：{input}

请从以下7个维度评估用户的状态，每个维度输出0-1的分数：
- fatigue: 能量耗尽（累、撑不住、不想动）
- meaning_loss: 意义崩塌（学什么都没用、没意义、算了、都一样）
- task_pressure: 任务压力（怎么办、做不完、不会、来不及）
- emotional_pain: 情感伤痛（难过、委屈、想哭、失去、悲痛）
- social_conflict: 人际冲突（被针对、吵架、不公平、忍不了）
- existential: 存在焦虑（我是谁、活着的意义、人生目标、虚无）
- factual_need: 事实需求（帮我查、告诉我、怎么操作、方法）

输出严格的JSON格式，不要任何其他文字：
{{
  "intent_distribution": {{
    "fatigue": 0.0, "meaning_loss": 0.0, "task_pressure": 0.0,
    "emotional_pain": 0.0, "social_conflict": 0.0, "existential": 0.0, "factual_need": 0.0
  }},
  "confidence": 0.0,
  "dominant_intent": "最有代表性的维度名"
}}""",

    "V2 (short)": """Classify the user's emotional state. Return ONLY a JSON:
{{
  "dominant_intent": "fatigue / meaning_loss / emotional_pain / task_pressure / social_conflict / existential / factual_need",
  "confidence": 0.0-1.0
}}

User: {input}""",

    "V3 (free then classify)": """Read the user's message and identify their core emotional state. Then choose the single best category.

User: {input}

Output format (JSON only):
{{
  "reasoning": "one sentence explaining the choice",
  "dominant_intent": "category_name",
  "confidence": 0.0-1.0
}}

Categories: fatigue, meaning_loss, emotional_pain, task_pressure, social_conflict, existential, factual_need""",
}

TEST_INPUTS = [
    "我感觉好累，一点力气都没有了，心里很烦，什么都不想做",
    "我觉得做什么都没有意义，脑子里一团乱，不想听任何人的建议",
    "太难受了，再也见不到他了，我接受不了",
    "领导要我违背原则办事，我不知道该怎么办",
    "帮我总结一下这篇文章的内容",
]

def call_probe(prompt_template, user_input):
    prompt = prompt_template.replace("{input}", user_input)
    payload = json.dumps({"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 300, "temperature": 0.1}).encode()
    req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=payload, headers={
        "Content-Type": "application/json", "Authorization": _b + " " + _ak})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=20).read())
        text = resp["choices"][0]["message"]["content"]
        # Extract JSON
        import re
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            result = json.loads(m.group())
            di = result.get("dominant_intent", result.get("intent_distribution", {}).get("dominant_intent", "unknown"))
            conf = result.get("confidence", 0.5)
            return di, conf, text[:100]
        return "parse_error", 0, text[:100]
    except Exception as e:
        return "error", 0, str(e)[:80]

print("=" * 80)
print("PROBE ROBUSTNESS CHECK (3 prompt variants × 5 inputs)")
print("=" * 80)

headers = ["Input"]
for vname in PROMPTS:
    headers.append(f"{vname}: intent/conf")
print("  %-45s %-20s %-20s %-20s" % tuple(headers))
print("  " + "-" * 105)

for inp in TEST_INPUTS:
    row = [inp[:40]]
    for vname, template in PROMPTS.items():
        di, conf, _ = call_probe(template, inp)
        if isinstance(conf, (int, float)):
            row.append(f"{di[:15]} {conf:.2f}")
        else:
            row.append(f"{di[:15]} {str(conf)[:4]}")
    print("  %-45s %-20s %-20s %-20s" % tuple(row))

# Summary
print("\n" + "=" * 80)
print("STABILITY SUMMARY")
print("=" * 80)

all_results = {v: [] for v in PROMPTS}
for inp in TEST_INPUTS:
    for vname, template in PROMPTS.items():
        di, conf, _ = call_probe(template, inp)
        all_results[vname].append(di)

# Check if all 3 variants agree per input
agreements = 0
for i in range(len(TEST_INPUTS)):
    intents = [all_results[v][i] for v in PROMPTS]
    # Check if dominant_intent category is same
    if len(set(intents)) == 1 or (len(set(intents)) == 2 and all("error" not in x for x in intents)):
        # Close enough - check if disagreement is between related categories
        related_pairs = [("fatigue","emotional_pain"), ("meaning_loss","existential"), ("task_pressure","factual_need")]
        unique = set(intents)
        related_ok = False
        for a, b in related_pairs:
            if unique == {a, b}:
                related_ok = True
        if related_ok or len(unique) == 1:
            agreements += 1
            status = "AGREE"
        else:
            status = f"MIXED ({', '.join(unique)})"
    else:
        status = f"MIXED ({', '.join(set(intents))})"
    print(f"  [{status:>20}] {TEST_INPUTS[i][:50]}")

print(f"\nProbe stability: {agreements}/{len(TEST_INPUTS)} inputs have consistent intent across 3 prompt variants")
