# Rules

- Work in small tasks.
- Do not scan the entire repo.
- Read only required files.
- Be extremely concise.
- Prefer reading `PLAN.md` and `TODO.md` over asking for repeated context.
- Ask permission before modifying `CLAUDE.md` or `.claude/rules/*`.

# Workflow

- Read `PLAN.md` and `TODO.md` before starting.
- In plan mode, first confirm the plan is approved before execution.
- Once approved, add or update `PLAN.md`.
- If the plan changes later, update `PLAN.md` immediately.
- Reorder `TODO.md` after each plan update.
- Maintain a simple priority matrix in `TODO.md`: `P0` now, `P1` next, `P2` later.
- After completing a task, update `PLAN.md` and `TODO.md`.
- Validate the build after code changes: `@.claude/rules/build.md`.

# Context

- Predictify: retail sales forecasting platform.
- Configuration lives in `backend/config.py`.
- `@backend/CLAUDE.md`
- `@frontend/CLAUDE.md`

# Output Format

1. files changed
2. commands run
3. next step
