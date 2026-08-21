---
description: Python-specific development guidance for fastf1_analytics
applyTo: "**/*.py"
---

# Python Development Instructions

## Follow Existing Project Conventions

Before introducing a new Python pattern, inspect nearby code and existing
modules for established conventions.

Prefer consistency with the existing project over introducing a newer pattern
merely because it is more modern.

## Preserve Interfaces

When changing functions, classes, modules, or data structures:

- Search for callers and usages.
- Check tests.
- Check CLI tools.
- Check plotting tools.
- Check documentation examples.
- Check public imports where relevant.

Do not change a public interface without considering compatibility.

## Data and Object Ownership

When modifying data structures or object relationships, explicitly determine:

- Who creates the object.
- Who owns the data.
- Who mutates it.
- Who consumes it.
- Whether callers rely on object identity.
- Whether data is copied or shared.
- Whether serialization or caching depends on the structure.

Do not infer ownership solely from the location of a class definition.

## Refactoring

Prefer focused refactoring tied to the current task.

Do not rewrite unrelated Python code simply to make it more modern or
consistent.

When a refactor changes an established interface, identify all affected
consumers before implementing it.

## Dependencies

Do not introduce or upgrade Python dependencies without a reason related to the
task.

Prefer existing dependencies and project utilities when they already provide
the required functionality.

## Validation

After Python changes:

- Run the most relevant focused tests.
- Run broader tests when appropriate.
- Run linting/type checking when configured.
- Exercise relevant CLI or plotting functionality when affected.

Never assume that successful syntax or import checks establish behavioral
correctness.