import asyncio
import os
import json
from openai import OpenAI
from my_env_v4 import MyEnvV4Env, MyEnvV4Action

API_KEY = os.getenv("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")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

TASK_NAME = "triage"
BENCHMARK = "healthcare_env"

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ================= RULE-BASED OVERRIDE =================
HIGH_RISK_KEYWORDS = [
    "chest pain", "breathing", "unconscious",
    "bleeding", "fainting", "severe", "pressure"
]

def rule_override(text):
    t = text.lower()
    for k in HIGH_RISK_KEYWORDS:
        if k in t:
            return MyEnvV4Action(
                action_type="finalize",
                urgency="high",
                department="emergency",
                escalate=True
            )
    return None

# ================= FEW-SHOT PROMPT =================
def build_prompt(obs):
    return f"""
You are a medical triage assistant.

Determine:
- urgency: low / medium / high
- department: general / dermatology / cardiology / emergency
- escalate: true / false

Examples:

Patient: mild headache since morning
Answer:
{{"action_type":"finalize","urgency":"low","department":"general","escalate":false}}

Patient: small rash and itching
Answer:
{{"action_type":"finalize","urgency":"low","department":"dermatology","escalate":false}}

Patient: chest pain with sweating
Answer:
{{"action_type":"finalize","urgency":"high","department":"emergency","escalate":true}}

Patient: occasional chest discomfort
Answer:
{{"action_type":"finalize","urgency":"medium","department":"cardiology","escalate":false}}

Patient: severe breathing problem
Answer:
{{"action_type":"finalize","urgency":"high","department":"emergency","escalate":true}}

Rules:
- High = life-threatening
- Medium = persistent symptoms
- Low = mild symptoms

Now classify:

Patient: {obs.current_query.message}

Return ONLY JSON:
"""

# ================= SAFE PARSER =================
def safe_parse(text):
    try:
        text = text.strip()

        if text.startswith("```"):
            text = text.split("```")[1]

        data = json.loads(text)

        return MyEnvV4Action(
            action_type="finalize",
            urgency=data.get("urgency", "medium"),
            department=data.get("department", "general"),
            escalate=data.get("escalate", False)
        )

    except:
        return MyEnvV4Action(
            action_type="finalize",
            urgency="medium",
            department="general",
            escalate=False
        )

# ================= MAIN =================
async def main():
    env = MyEnvV4Env()
    obs = env.reset()

    rewards = []
    step = 0
    done = False
    last_error = None

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

    while not done and step < 20:
        step += 1

        try:
            # 🔥 RULE-BASED SHORTCUT
            action = rule_override(obs.current_query.message)

            # 🤖 If no rule → call API
            if not action:
                res = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": build_prompt(obs)}],
                    temperature=0.0,
                    max_tokens=120
                )

                action = safe_parse(res.choices[0].message.content)

            obs, reward, done, info = env.step(action)
            last_error = info.get("error", None)

        except Exception as e:
            action = "error"
            reward = 0.0
            done = True
            last_error = str(e)

        rewards.append(reward)

        print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={last_error if last_error else 'null'}")

    env.close()

    score = env.normalized_score()
    success = score >= 0.5

    reward_str = ",".join([f"{r:.2f}" for r in rewards])

    print(f"[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={reward_str}")

# ================= ENTRY =================
if __name__ == "__main__":
    asyncio.run(main())
