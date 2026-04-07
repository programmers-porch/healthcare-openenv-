import asyncio
import os
import json
from openai import OpenAI
from my_env_v4 import MyEnvV4Env, MyEnvV4Action

API_KEY = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

HIGH_RISK = ["chest pain", "breathing", "unconscious", "bleeding", "fainting"]


def override(text):
    t = text.lower()
    for k in HIGH_RISK:
        if k in t:
            return MyEnvV4Action(
                action_type="finalize",
                urgency="high",
                department="emergency",
                escalate=True
            )
    return None


def prompt(obs):
    return f"""
Classify patient query.

Urgency: low / medium / high
Department: general / dermatology / cardiology / emergency

Return JSON:
{{"action_type":"finalize","urgency":"...","department":"...","escalate":true/false}}

Patient: {obs.current_query.message}
"""


def parse(txt):
    try:
        return MyEnvV4Action(**json.loads(txt))
    except:
        return MyEnvV4Action(action_type="finalize", urgency="medium", department="general")


async def main():
    env = MyEnvV4Env()
    obs = env.reset()

    rewards = []
    step = 0
    done = False

    print(f"[START] task=triage env=healthcare_env model={MODEL_NAME}")

    while not done and step < 20:
        step += 1

        try:
            act = override(obs.current_query.message)

            if not act:
                res = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt(obs)}],
                    temperature=0.1
                )
                act = parse(res.choices[0].message.content)

            obs, reward, done, info = env.step(act)
            err = info.get("error", None)

        except Exception as e:
            act = "error"
            reward = 0.0
            done = True
            err = str(e)

        rewards.append(reward)

        print(f"[STEP] step={step} action={act} reward={reward:.2f} done={str(done).lower()} error={err if err else 'null'}")

    score = env.normalized_score()
    success = score >= 0.5

    print(f"[END] success={str(success).lower()} steps={step} score={score:.2f} rewards={','.join([f'{r:.2f}' for r in rewards])}")


if __name__ == "__main__":
    asyncio.run(main())
