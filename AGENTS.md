# Agent Instructions

## Project context

- This is a research project.
- Optimize for learning, iteration, and a working result—not production-grade architecture.
- Prefer the simplest solution that answers the current research question.

## Engineering scope

- Do not over-engineer or introduce unnecessary abstractions, frameworks, dependencies, or infrastructure.
- Keep implementations small and easy to change.
- Avoid solving hypothetical future requirements unless the user explicitly asks for them.
- Spend effort on the requested behavior and useful experiments rather than polish that does not support the research.

## Images and websites

- Do not spend tokens inspecting images or browsing/viewing websites just to infer what should be built.
- Build a reasonable first pass from the user’s written request and available project context, then let the user inspect it and say what should change.
- Only inspect an image or website when the user explicitly asks for that inspection or when it is strictly necessary to complete a specific, stated requirement.

## Assumptions and documentation

- State every material assumption explicitly in task notes, code comments, or the final response as appropriate.
- When requirements are ambiguous, make the smallest reasonable assumption that keeps work moving and document it.
- Distinguish clearly between user-provided requirements, observed project facts, and agent assumptions.
- Record important tradeoffs, shortcuts, and intentionally out-of-scope work so they are not mistaken for omissions.

## Delivery

- Implement the smallest useful version, run proportionate checks, and report what was built.
- Tell the user what assumptions were made and invite concrete feedback for the next iteration.
