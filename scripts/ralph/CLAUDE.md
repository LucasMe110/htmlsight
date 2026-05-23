# Ralph Agent Instructions — htmlsight (ia-visao-web)

You are an autonomous coding agent completing the htmlsight project: a multi-task visual detector of Bootstrap web components trained on a synthetic dataset.

## Project context

Read `/home/lucas/workspace/htmlsight/CLAUDE.md` for commands, structure, and gotchas.

Key commands:
- Run tests: `venv/bin/python -m pytest -v`
- Lint: `venv/bin/python -m ruff check .`
- Typecheck: `venv/bin/python -m mypy src`
- Install deps: `venv/bin/python -m pip install -e ".[dev]"`

Working directory is always `/home/lucas/workspace/htmlsight`.

## Your Task

1. Read the PRD at `scripts/ralph/prd.json`
2. Read the progress log at `scripts/ralph/progress.txt` (check Codebase Patterns section first)
3. Check you're on the correct branch from PRD `branchName`. If not, check it out or create from main.
4. Pick the **highest priority** user story where `passes: false`
5. Implement that single user story following TDD: write tests first, then implementation
6. Run quality checks: `venv/bin/python -m pytest -v && venv/bin/python -m ruff check . && venv/bin/python -m mypy src`
7. Update `CLAUDE.md` at the project root if you discover reusable patterns
8. If checks pass, commit ALL changes with: `feat: [Story ID] - [Story Title]`
9. Update `scripts/ralph/prd.json` to set `passes: true` for the completed story
10. Append your progress to `scripts/ralph/progress.txt`

## TDD requirement

Per project rules (CLAUDE.md global), always write tests before production code:
1. Write the failing test
2. Run to confirm it fails
3. Write minimal implementation
4. Run to confirm it passes

## Progress Report Format

APPEND to `scripts/ralph/progress.txt` (never replace, always append):
```
## [Date/Time] - [Story ID]
- What was implemented
- Files changed
- **Learnings for future iterations:**
  - Patterns discovered
  - Gotchas encountered
  - Useful context
---
```

## Consolidate Patterns

If you discover a reusable pattern, add it to the `## Codebase Patterns` section at the TOP of `scripts/ralph/progress.txt`:

```
## Codebase Patterns
- Ultralytics: use X for Y
- Torch: always check torch is not None before instantiating
- Tests: use monkeypatch.setattr to mock heavy dependencies
```

## Quality Requirements

- ALL commits must pass: pytest, ruff, mypy
- Do NOT commit broken code
- Keep changes focused — one story per iteration
- Follow existing code patterns (src/ layout, dataclasses, typed, no unneeded comments)
- Heavy dependencies (torch, ultralytics, playwright) must fail with actionable errors when absent

## Stop Condition

After completing a story, check if ALL stories have `passes: true`.

If ALL stories are complete:
<promise>COMPLETE</promise>

Otherwise end your response normally.

## Important

- Work on ONE story per iteration
- Commit after each story passes quality checks
- ALWAYS use TDD (test first, then implementation)
- Monkeypatch or mock ultralytics/torch in tests — don't require real GPU
- When modifying model code, always guard with `if torch is None` checks
