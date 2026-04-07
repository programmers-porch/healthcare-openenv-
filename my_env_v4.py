from pydantic import BaseModel
from typing import List, Optional
import json
from utils.graders import (
    grade_urgency,
    grade_department,
    grade_escalation,
    grade_request_info,
    action_valid,
)

MAX_STEPS_PER_QUERY = 3


class PatientQuery(BaseModel):
    message: str
    expected: dict


class Observation(BaseModel):
    current_query: PatientQuery
    history: List[str]
    step_in_query: int
    remaining_queries: int


class MyEnvV4Action(BaseModel):
    action_type: str
    urgency: Optional[str] = None
    department: Optional[str] = None
    escalate: Optional[bool] = None


class MyEnvV4Env:

    def __init__(self):
        self.data = []
        self.index = 0
        self.step_in_query = 0
        self.history = []
        self.done = False
        self.total_reward = 0.0

    def load_data(self):
        with open("dataset.json") as f:
            return json.load(f)

    def reset(self):
        self.data = self.load_data()
        self.index = 0
        self.step_in_query = 0
        self.history = []
        self.done = False
        self.total_reward = 0.0
        return self._get_obs()

    def _get_obs(self):
        return Observation(
            current_query=PatientQuery(**self.data[self.index]),
            history=self.history,
            step_in_query=self.step_in_query,
            remaining_queries=len(self.data) - self.index
        )

    def _advance(self):
        self.index += 1
        self.step_in_query = 0
        if self.index >= len(self.data):
            self.done = True
            self.index = len(self.data) - 1

    def step(self, action: MyEnvV4Action):
        if self.done:
            return self._get_obs(), 0.0, True, {"error": None}

        item = self.data[self.index]
        expected = item["expected"]

        reward = 0.0
        error = None

        try:
            if not action_valid(action.action_type):
                return self._finish(action, -0.3, "invalid_action")

            u = grade_urgency(action.urgency, expected["urgency"])
            d = grade_department(action.department, expected["department"])
            e = grade_escalation(action.escalate, expected.get("escalate", False))
            i = grade_request_info(action.action_type, expected.get("needs_info", False))

            if action.action_type == "classify":
                reward += 0.4 * u

            elif action.action_type == "route":
                reward += 0.4 * d

            elif action.action_type == "request_info":
                reward += 0.2 * i

            elif action.action_type == "escalate":
                reward += 0.3 * e

            elif action.action_type == "finalize":
                reward += 0.35 * u + 0.35 * d + 0.15 * e + 0.1 * i
                if u == 1 and d == 1:
                    reward += 0.1
                self._advance()

            reward -= 0.02  # step penalty

            if self.history and str(action) == self.history[-1]:
                reward -= 0.05

        except Exception as ex:
            error = str(ex)
            reward -= 0.2

        return self._finish(action, reward, error)

    def _finish(self, action, reward, error):
        reward = max(-1.0, min(1.0, reward))
        self.total_reward += reward
        self.history.append(str(action))
        self.step_in_query += 1

        if self.step_in_query >= MAX_STEPS_PER_QUERY:
            self._advance()

        return self._get_obs(), round(reward, 2), self.done, {"error": error}

    def normalized_score(self):
        max_score = len(self.data)
        if max_score == 0:
            return 0.0
        return max(0.0, min(1.0, self.total_reward / max_score))

    def state(self):
        return {
            "index": self.index,
            "remaining": len(self.data) - self.index,
            "total_reward": round(self.total_reward, 2),
        }

    def close(self):
        pass
