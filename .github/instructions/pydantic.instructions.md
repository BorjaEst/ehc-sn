---
description: "Use when editing Python files that define Pydantic models or Field(...) metadata. Prefer direct literal description=... values over wrapping plain string descriptions in description=(...)."
applyTo: "{src/**/*.py,scripts/**/*.py}"
---

# Pydantic Field Description Style

- For `Field(..., description=...)` calls, when the description value is a plain string literal or a multi-line implicit concatenation of string literals, write it directly as the `description=` value.
- Prefer this form:
  `description="Path to save checkpoints and logs. If not set, it "`
  `"defaults to \`checkpoints/<project_name>/<run_name>\`."`
- Avoid wrapping plain literal descriptions as `description=(...)` when the outer parentheses add no meaning.
- Use outer parentheses only when the description value is not a plain literal expression, such as conditional expressions, helper calls, interpolation, or other non-literal composition.
- Do not change wording, punctuation, or content while applying this rule; normalize structure only.
