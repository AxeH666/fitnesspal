# Barbarik Engineering Rules

## Current Objective

Build only the approved Barbarik POC v1 described in:

`docs/POC-v1.md`

That document defines product scope.

Do not expand it independently.

## Hard Scope Rule

Do not implement features merely because they are useful, interesting, elegant, future-proof, or easy to add.

If a requested change is outside `docs/POC-v1.md`, stop and surface it as a scope decision.

Do not silently add it.

Approval means `docs/POC-v1.md` is updated first. A conversational "yes" does not authorize implementing anything the specification does not describe.

## Engineering Principle

Prefer the smallest reliable implementation that proves the required behaviour.

Priority:

1. correctness
2. working end-to-end behaviour
3. simplicity
4. tests
5. maintainability required for the POC

Novelty and architectural sophistication are not goals.

## Do Not Overengineer

Avoid unless directly required:

- microservices
- Kubernetes
- event buses
- unnecessary queues
- extra databases
- unnecessary Redis usage
- vector databases
- custom ML models
- generic plugin architectures
- speculative abstractions
- premature scaling
- duplicate services
- unnecessary frameworks

Reuse the existing Phase 0 scaffold where practical.

## Git Workflow — Mandatory

Never work directly on `main`.

Every change must use a branch.

Typical format:

`poc/<short-description>`

Every completed milestone/change is delivered through a Pull Request.

Never push implementation commits directly to `main`.

Before requesting merge:

- run relevant tests
- review diff
- verify scope
- verify no secrets
- summarize changes
- identify remaining limitations

One coherent PR is preferred over a large mixed PR.

## Scope Discipline

Before changing code, identify which POC requirement the change satisfies.

If there is no direct requirement, do not implement it without explicit approval.

If product requirements appear contradictory, stop and ask instead of guessing.

## Existing Code

Do not replace working Phase 0 components without a demonstrated reason.

Prefer modification over unnecessary rewrites.

## Data Truth

PostgreSQL owns structured fitness history and chronology.

Do not rely on LLM conversation history as the source of truth for:

- dates
- meals
- workouts
- bodyweight
- targets
- video-analysis history

The recommended daily protein target is calculated by deterministic backend code from the member's basic profile. PostgreSQL stores both the recommendation and the user's chosen active target. Later nutrition behaviour uses the active target.

Date handling must be deterministic and timezone-aware.

## AI Behaviour

The model handles:

- natural-language interpretation
- reasoning
- response generation
- explaining recommendations
- recognizing accept, reject, or override intent

Deterministic code handles:

- member identity
- timestamps
- local dates
- data persistence
- recommended protein target calculation
- activity intensity scoring
- all stored totals and remaining-intake arithmetic

The LLM must not invent or independently calculate the authoritative protein target or the workout-intensity score.

Per-meal calorie and macro values are the documented exception: for the POC they may be model estimates. Totals derived from stored meals are calculated by backend code.

### Reasoning and Retrieval Policy

Use the LLM as the primary reasoning engine. It handles interpretation, advice, allowed estimation, explanation, and the decision about whether an authoritative tool is needed.

Do not retrieve database state merely because member data exists. Answer directly from general knowledge or the current interaction when exact personal state is not required. This includes:

- open-ended fitness or nutrition advice
- food-option questions such as "What can I eat from these options?"
- meal-quality questions such as "Is this a good post-workout meal?"
- substitutions and meal ideas
- general fitness questions
- reasoning over information already supplied in the current interaction or available context

Browsing or searching may be used when current external information is genuinely useful and available agent tooling supports it. This policy does not authorize adding browsing infrastructure or expand the POC scope.

Retrieve authoritative state before answering when the result depends on exact personal facts, chronology, or persistence. This includes:

- today's remaining protein or consumed calories
- yesterday's meals or training
- the last performance of an exercise
- comparisons between bodyweight records
- a previously stored video-analysis summary
- any claim about stored member history or exact day-specific facts

Every log, mutation, or target update must use the appropriate backend operation.

The LLM must not invent exact personal history, dates, targets, totals, or stored values. When those facts are required, retrieve them. When they are not required, let the LLM answer directly.

In short:

- LLM = reasoning, interpretation, advice, estimation, explanation, and deciding whether a tool is needed.
- Backend/PostgreSQL = authoritative personal facts, dates, historical state, persistence, and deterministic calculations already defined by the POC.

## Video

Use an existing video-capable model for the POC.

Do not build a custom computer-vision pipeline unless the POC specification is explicitly changed.

## Dashboard

The POC dashboard is local.

Do not add cloud deployment, production authentication, or mobile-app complexity unless `docs/POC-v1.md` is explicitly changed to require it.

Do not add member fields or panels beyond those listed in the dashboard section of `docs/POC-v1.md`.

## Security

Never commit:

- API keys
- tokens
- passwords
- real `.env` files
- private credentials

Use `.env.example` for documented variables.

## Working Style

Make small, inspectable changes.

Run tests after meaningful changes.

Do not combine unrelated work.

When uncertain, surface the decision instead of inventing product behaviour.
