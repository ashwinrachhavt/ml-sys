# Presentation Script (60 minutes)

Use this outline to lead the panel through the repo. Time boxes keep the conversation tight while leaving room for discussion.

## 0:00 – 0:03 | Opening hook
- Problem framing: sales teams need trustworthy conversion scores quickly.
- Two design pillars: (1) config-first experimentation, (2) single pipeline reused for CLI + API.
- Outcome: analysts can ship a model to production in hours, not days.

## 0:03 – 0:12 | Walk the architecture
- Share the high-level diagram from the README.
- Call out package boundaries (`data`, `features`, `models`, `training`, `serving`). Mention the registries pattern and why it keeps the framework extendable.
- Highlight `Settings` for configuration + environment overrides and `config/config.schema.json` for validation.

## 0:12 – 0:22 | Demo the experimentation loop
- Show `config/config.yaml`, tweak a hyperparameter, and run `make train` (or describe expected output if running live is risky).
- Explain how `Trainer` performs CV, surfaces metrics, and logs to MLflow when enabled.
- Mention callbacks and how they can power Slack alerts, lineage tracking, or notifications.

## 0:22 – 0:32 | Serving + monitoring story
- Run `make serve` and open `/docs` to show the FastAPI surface.
- Walk through a sample `/predict` payload and explain how `PredictorService` loads either local artifacts or the registry.
- Discuss monitoring hooks (Prometheus middleware, MLflow inference logging) and how they tie back to SLA tracking.

## 0:32 – 0:42 | Live Q&A / whiteboard
- Invite questions about scaling data volume, swapping model frameworks, or integrating with Feature Store / Airflow.
- Use `docs/improvement-plan.md` as a cheat sheet for known gaps; emphasize that you already have a plan to address them.

## 0:42 – 0:50 | Follow-up roadmap
- Detail the top three backlog items (data leakage guardrails, threshold persistence, serving resilience).
- Mention long-term bets (model monitoring automation, batch scoring jobs, CI coverage expansion).

## 0:50 – 0:60 | Close strong
- Recap: reproducible runs, flexible abstractions, production-ready serving.
- Share what you would do with extra time (e.g., integrate drift dashboards, auto-retraining triggers).
- Reiterate enthusiasm for enabling DS partners and how this template accelerates onboarding.

