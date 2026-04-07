# 🏥 Healthcare Triage OpenEnv Environment

## 📌 Overview

This project implements a **real-world healthcare triage simulation environment** using the OpenEnv framework.  
The environment allows an AI agent to classify patient symptoms and make decisions regarding:

- Urgency level (low / medium / high)
- Department routing (general / dermatology / cardiology / emergency)
- Escalation requirement

This simulates real-world workflows used in hospitals, telemedicine systems, and emergency triage systems.

---

## 🎯 Motivation

Healthcare triage is a **critical real-world task** where incorrect decisions can have serious consequences.  
This environment enables training and evaluation of AI agents for:

- Automated patient triage
- Decision support systems
- AI-assisted healthcare routing

---

## ⚙️ Environment Design

### 🔹 Observation Space

Each observation contains:

- `current_query`: Patient message and expected labels
- `history`: Previous agent actions
- `step_in_query`: Current step number
- `remaining_queries`: Number of queries left

---

### 🔹 Action Space

The agent outputs:

```json
{
  "action_type": "finalize",
  "urgency": "low | medium | high",
  "department": "general | dermatology | cardiology | emergency",
  "escalate": true | false
}
