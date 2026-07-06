#!/usr/bin/env python3
"""Semantic robustness test — probe on paraphrased inputs.
Same meaning, different wording. Does probe still classify correctly?
"""
import sys, json, urllib.request, re, ssl
sys.path.insert(0, "/opt/lingyao")
_ak = json.load(open("/root/.openclaw/openclaw.json"))["models"]["providers"]["deepseek"]["apiKey"]
_b = chr(66)+chr(101)+chr(97)+chr(114)+chr(101)+chr(114)

PROMPT = """Classify the user's emotional state. Return ONLY a JSON:
{"dominant_intent": "fatigue / meaning_loss / emotional_pain / task_pressure / social_conflict / existential / factual_need", "confidence": 0.0-1.0}

User: {input}"""

# 10 inputs covering 5 categories with paraphrases
TEST_CASES = [
    # fatigue (3 variants)
    ("fatigue", "我感觉好累，一点力气都没有了"),
    ("fatigue", "完全提不起劲，连呼吸都觉得重"),
    ("fatigue", "不想动也不想说话，整个人被掏空了"),
    # meaning_loss (2 variants)
    ("meaning_loss", "我觉得做什么都没有意义"),
    ("meaning_loss", "不知道每天在忙什么，一切都很空虚"),
    # emotional_pain (2 variants)
    ("emotional_pain", "太难受了，再也见不到他了"),
    ("emotional_pain", "想到这件事就忍不住掉眼泪"),
    # social_conflict (2 variants)
    ("social_conflict", "领导要我违背原则办事"),
    ("social_conflict", "同事一直甩锅给我，忍了很久了"),
    # task_pressure (1)
    ("task_pressure", "写不完论文了，deadline快到了"),
]

def call(input_text):
    prompt = PROMPT.replace("{input}", input_text)
    payload = json.dumps({"model":"deepseek-chat","messages":[{"role":"user","content":prompt}],
                          "max_tokens":100,"temperature":0.1}).encode()
    req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=payload,
        headers={"Content-Type":"application/json","Authorization":_b+" "+_ak})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=20).read())
        text = resp["choices"][0]["message"]["content"]
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            r = json.loads(m.group())
            return r.get("dominant_intent","?"), r.get("confidence",0)
        return "?", 0
    except:
        return "error", 0

print("="*80)
print("SEMANTIC ROBUSTNESS: Same intent, different wording")
print("="*80)
print(f"  {'Expected':15s} {'Detected':15s} {'Conf':5s} {'Input':40s}")
print(f"  {'-'*15} {'-'*15} {'-'*5} {'-'*40}")

correct = 0
total = 0
for expected, text in TEST_CASES:
    total += 1
    detected, conf = call(text)
    ok = detected == expected
    if ok: correct += 1
    status = "✅" if ok else "❌"
    print(f"  {expected:15s} {detected:15s} {conf:.2f}  {status} {text[:38]:40s}")

print(f"\nAccuracy: {correct}/{total}")
print(f"\nConclusion: {'Probe handles paraphrases correctly' if correct/total > 0.8 else 'Probe fails on some paraphrases — needs investigation'}")
