# Barbarik AI — Proof of Concept v1

## Status

POC specification.

This document defines the exact scope for the gym-owner demonstration.

Do not expand this scope without an explicit product decision.

---

# 1. Objective

Build a working proof of concept showing that Barbarik is more useful than using a generic AI chatbot for fitness.

The POC must demonstrate two important differentiators:

1. persistent, real calendar-day fitness memory
2. short training-video analysis integrated into the member's fitness history

The system should be narrow, reliable, and demonstrable to the Barbarik gym owner.

This is not the production V1.

---

# 2. Demo User

The POC supports one seeded demo member.

Multi-user production onboarding is out of scope for the POC.

## Minimum profile

The minimum POC profile consists of:

- age
- weight
- gender
- fitness goal
- timezone where required for chronological handling

A demo member name may exist in seed data for identification.

Do not require additional onboarding inputs such as:

- height
- training days
- activity level
- diet preference
- existing workout plan
- injuries
- other profile fields

Seed data for the POC consists of the demo member name plus the minimum profile above. Any other profile field left over from Phase 0 is leftover schema, not a POC feature. Do not build POC behaviour or dashboard surface on top of it.

The demo member must not arrive with a manually configured protein target as a required onboarding input.

Calorie target is not a required onboarding input. It can still be set later through an explicit target update.

## Protein target

After the basic profile is available:

1. Barbarik calculates a recommended daily protein target using deterministic backend logic.
2. Barbarik presents and explains the recommendation conversationally.
3. The user may accept the recommendation.
4. The user may instead set their own protein target.
5. The active selected value is persisted.
6. The active value is used for nutrition tracking, remaining-protein calculations, and contextual food recommendations.

### Recommendation formula

The POC uses one simple documented deterministic formula: grams of protein per kilogram of bodyweight, selected by fitness goal.

- fat loss: 2.0 g/kg
- muscle gain: 1.8 g/kg
- general fitness / maintenance: 1.6 g/kg

The result is rounded to the nearest whole gram.

The formula uses bodyweight and fitness goal only. Age and gender are stored profile data and are not inputs to the POC calculation, so the same weight and goal always produce the same recommendation.

This formula lives in backend code, is covered by tests, and is not chosen by the LLM.

### Behaviour before an active target exists

Until the user accepts or overrides a recommendation, there is no active protein target.

In that state Barbarik must present the recommendation and ask the user to accept it or set their own, rather than silently treating the recommendation as the active target.

Remaining-protein answers and contextual food recommendations require an active protein target. If none exists, Barbarik asks the user to choose one first.

### Source of truth

LLM:

- understands user intent
- reasons conversationally
- explains the recommendation
- discusses alternatives
- recognizes whether the user is accepting, rejecting, or overriding the recommendation

Deterministic backend:

- calculates the default recommended protein target
- stores the recommendation
- stores the user's chosen active target
- owns exact numerical state

The LLM may reason around the recommendation. It must not invent or independently calculate the authoritative protein target.

Example:

Profile: 79 kg, fat loss goal

Backend recommendation: 79 × 2.0 = 158 g/day

Assistant:

"Based on your weight and goal, I'd recommend about 158 g of protein per day. Want to use that, or set your own target?"

The user may accept that value, or override it:

"That seems high. Can I do 140?"

If the user clearly chooses 140 g/day, Barbarik stores 140 g/day as the active target.

The LLM may discuss tradeoffs. Mere discussion without a clear choice must not change the stored target.

### Acceptance criteria

- the recommended protein target is deterministic
- the same profile inputs produce the same recommendation
- the recommendation comes from backend code, not the LLM
- both the recommendation and the active target are stored
- the assistant can explain the recommendation naturally
- the user can accept the recommendation
- the user can override the recommendation
- an override becomes the stored active target
- later queries use the active target, not the original recommendation
- with no active target yet, Barbarik asks the user to choose instead of assuming the recommendation

---

# 3. Primary Interface

The main interaction interface is WhatsApp.

The user communicates naturally.

Examples:

"Weight 79.4"

"Had 4 eggs and 2 rotis."

"Bench 80kg for 8,8,6."

"What did I bench last time?"

"What did I eat yesterday?"

"I have chicken, rice and eggs. What should I eat now?"

"Set my calories to 2300."

The user should not need special commands or syntax.

---

# 4. Bodyweight Logging

The user can log bodyweight naturally.

Example:

"Weight 79.4"

Barbarik stores:

- member
- weight
- exact timestamp
- local calendar date

Historical bodyweight must remain queryable.

---

# 5. Food and Nutrition Logging

The user can describe food naturally.

Example:

"Had 4 eggs, 2 rotis and curd."

Barbarik should:

1. interpret the meal
2. estimate calories and macros
3. store the meal
4. associate it with the correct local calendar date
5. update today's totals

The user can ask:

"How much protein do I have left today?"

Barbarik answers using stored data and the member's current active targets.

For the POC, nutrition values may be reasonable AI estimates and do not need a complete commercial food database.

---

# 6. Workout Logging

The user can log workouts naturally.

Example:

"Bench 80kg 8,8,6. Incline dumbbells 30kg 10,9,8."

Barbarik stores:

- exercise
- sets
- reps
- load
- timestamp
- local calendar date

The member can later ask:

"What did I bench last time?"

"What did I train yesterday?"

"What was my previous chest workout?"

---

# 7. Real Calendar-Day Memory

This is a mandatory POC capability.

A long WhatsApp conversation must never be treated as one continuous day.

Every relevant stored event must belong to a real calendar day based on the member's timezone.

The backend/database, not the LLM chat history, owns chronological truth.

Example:

30 August
- weight
- meals
- workout
- video analysis

31 August
- separate weight
- separate meals
- separate workout state

When the local date changes, the new day's nutrition and activity state starts separately.

Historical records remain unchanged.

Required queries include:

"What did I eat yesterday?"

"What did I train yesterday?"

"What did I bench last time?"

"Compare today's weight with yesterday."

"What did you tell me about my squat yesterday?"

The agent must retrieve the relevant stored records instead of guessing from conversation history.

---

# 8. Contextual Food Recommendation

The member can ask:

"I have chicken, rice, eggs and curd. What should I eat now?"

This can be either a general food-options question or a request grounded in the member's exact current state. The LLM decides which case applies.

A general question such as "What can I eat from these options?" or "Is this a good post-workout meal?" can be answered directly from the foods and context in the current interaction. The existence of stored member data does not by itself require retrieval.

For the personalized, state-dependent version of the request, Barbarik retrieves and uses:

- current fitness goal
- calorie target, if one has been set
- active protein target
- food already logged today
- remaining intake
- foods the user says are available

That result should be contextual to that member and that day. If the response states or depends on exact remaining intake, current targets, or food already logged today, those values must be retrieved rather than inferred.

If no calorie target has been set, Barbarik reasons from the active protein target and today's logged food, and does not report remaining calories.

An active protein target is required for the personalized, state-dependent version of this feature. See section 2.

---

# 9. Target Changes

The user can explicitly change a nutrition target.

Example:

"Set my calories to 2300."

This should update the stored target.

Protein target follows the recommendation, accept, and override flow in section 2. After an active protein target exists, the user can also change it explicitly, for example:

"Set my protein to 140."

The assistant must distinguish discussion from commands.

Example:

"I'm thinking about trying 2300 calories."

must not automatically change the target.

---

# 10. Video Analysis

Video analysis is mandatory for the POC.

The member can send a short training video.

The supported demonstration categories for the POC are:

- squat
- bench press
- deadlift

Other movement categories, including combat-sport examples, are outside the POC and require an explicit update to this document.

The POC does not require a custom computer-vision model.

Use an existing video-capable AI model.

Flow:

WhatsApp video
→ media ingestion
→ video-capable model
→ structured observations
→ Barbarik response
→ saved analysis

The analysis should include useful observations such as:

- movement/exercise identified
- visible technique observations
- consistency across repetitions
- obvious issues
- practical improvements
- uncertainty where appropriate

Do not claim medical, biomechanical, or professional coaching certainty.

---

# 11. Persistent Video History

A completed video analysis must be stored against:

- member
- timestamp
- local calendar date
- exercise/activity if known
- analysis summary

Later the member must be able to ask:

"What did you say about my squat yesterday?"

and receive the saved analysis from the correct day.

---

# 12. Local Dashboard

The POC dashboard is a local website running on the demonstration laptop.

It is not a production cloud application.

The interface is inspired by the GitHub profile and contribution graph interaction model.

It shows exactly these member details:

- name
- goal
- current bodyweight
- calorie target, if set
- active protein target

Do not add other profile fields to the dashboard for the POC.

## Activity Grid

The centerpiece is a GitHub-style calendar activity grid.

Each tile represents one real calendar day.

Workout intensity is represented using red intensity:

- empty/dark = no workout activity
- faint red = light
- moderate red = moderate
- strong red = hard
- strongest red = very intense

The intensity score should be deterministic from stored workout data.

The LLM must not arbitrarily choose the tile intensity.

For the POC, a simple deterministic scoring formula is sufficient.

## Day Details

Clicking a calendar tile opens the stored record for that day.

Where available it shows:

- bodyweight
- meals
- calories/macros
- workout
- exercises
- sets/reps/load
- the persisted video-analysis summary for that day

It displays stored records only. The dashboard does not generate new commentary or insights.

This dashboard visually demonstrates Barbarik's chronological memory.

---

# 13. POC Architecture

Keep the architecture simple.

WhatsApp
→ Barbarik backend
→ member/date resolution
→ agent/model
→ explicit fitness operations
→ PostgreSQL

Dashboard
→ Barbarik backend
→ PostgreSQL

Video
→ backend
→ video-capable model
→ saved structured analysis

PostgreSQL is the source of truth for what happened and when.

The LLM interprets language and generates useful responses.

The LLM must not be the database.

Deterministic backend code owns calculated numerical values such as the recommended protein target and the workout-intensity score. The LLM may explain those values. It must not invent them.

Per-meal calorie and macro values are the one exception: for the POC they may be model estimates, as stated in section 5. Once stored they are data, and totals are computed by the backend.

## Reasoning and Retrieval Policy

Barbarik uses the LLM as the primary reasoning engine. The LLM interprets the request, gives advice and explanations, makes allowed estimates, and decides whether an authoritative tool or database read is needed.

The existence of a member record or other stored data does not automatically require database retrieval.

The LLM should normally answer directly, without personal database retrieval, when the request can be handled from general knowledge or information already present in the current interaction. Examples include:

- open-ended fitness or nutrition advice
- "What can I eat from these options?"
- "Is this a good post-workout meal?"
- substitutions and meal ideas
- general fitness questions
- reasoning over information already supplied in the current interaction or available context

When current external information is genuinely useful, the LLM may use browsing or search if the available agent tooling supports it. This policy does not add a browsing requirement or browsing infrastructure to the POC.

Backend or PostgreSQL retrieval is required when the answer depends on exact personal state, chronology, or persisted history. Examples include:

- "How much protein do I have left today?"
- "How many calories have I eaten today?"
- "What did I eat yesterday?"
- "What did I train yesterday?"
- "What did I bench last time?"
- "Compare today's weight with yesterday."
- "What did you say about my squat yesterday?"
- any claim about stored member history or exact day-specific facts

Every log, mutation, or target update must use the appropriate backend operation so the change is validated and persisted.

Decision rule:

- If exact personal state is not required, let the LLM answer directly.
- If the answer would state, compare, calculate from, or change exact stored personal facts, retrieve or mutate through the backend first.
- Retrieved state is authoritative. The LLM must not invent exact personal history, dates, targets, totals, or stored values.

In short:

- LLM = reasoning, interpretation, advice, estimation, explanation, and deciding whether a tool is needed.
- Backend/PostgreSQL = authoritative personal facts, dates, historical state, persistence, and deterministic calculations already defined by the POC.

---

# 14. Implementation Milestones

## POC-0 — Stabilize Existing Phase 0

Verify:

- backend starts
- PostgreSQL starts
- Redis starts if actually needed
- migrations run
- seed works
- tests pass
- dashboard shell starts

Do not rebuild working Phase 0 infrastructure unnecessarily.

## POC-1 — Date-Aware Fitness Core

Implement the smallest deterministic backend required for:

- member state
- weight logging
- food logging
- workout logging
- today's state
- arbitrary-day state
- recent workout history
- exercise history
- deterministic recommended protein target
- recommended vs active protein target storage
- nutrition target update
- timezone/local-date handling

Acceptance requirement:

Records from two simulated calendar days cannot become mixed.

## POC-2 — Natural Language Assistant

Connect an LLM to the core operations.

Prove:

- natural language logging
- historical retrieval
- command vs discussion distinction
- protein recommendation explanation
- accept vs override of protein target
- contextual food recommendation

Do not add unnecessary agent infrastructure.

## POC-3 — WhatsApp

Connect one real demonstration WhatsApp account/number.

Required path:

phone
→ WhatsApp
→ Barbarik
→ database/action
→ reply

Only one demo member is required.

## POC-4 — Local Dashboard

Connect the existing React dashboard to real backend data.

Implement:

- member details
- GitHub-inspired red calendar activity grid
- clickable day records

Do not build a complete production application.

## POC-5 — Video Analysis

Add short-video ingestion and analysis.

Persist the useful analysis summary against the correct member and calendar day.

## POC-6 — Final Cross-Day Demo

Prove the complete system across at least two calendar days, either using controlled test dates or real elapsed days.

Day A:

- log weight
- log food
- log workout
- analyze training video

Day B:

- log new weight

Then successfully answer:

"What did I do yesterday?"

"What did I eat yesterday?"

"What did I bench yesterday?"

"What did you say about my squat yesterday?"

"What have I eaten today?"

---

# 15. Definition of Done

The gym-owner POC is complete when all of the following work:

- one real WhatsApp interface
- one demo member
- natural-language bodyweight logging
- natural-language food logging
- calorie/protein tracking
- deterministic recommended protein target from the basic profile
- conversational accept or override of protein target
- later nutrition behaviour uses the active protein target
- natural-language workout logging
- workout history retrieval
- deterministic real-calendar-day separation
- today/yesterday queries
- contextual food recommendations
- explicit nutrition target update
- command vs discussion distinction
- local GitHub-inspired dashboard
- clickable calendar-day history
- deterministic red workout-intensity tiles
- short training-video analysis
- persisted/retrievable video analysis
- data survives application restart

---

# 16. Explicitly Out of Scope

Do not implement these for the POC:

- production multi-user onboarding
- onboarding fields beyond age, weight, gender, fitness goal, and timezone
- full subscription/payment system
- native Android/iOS application
- wearables
- Apple Health
- coach dashboard
- multi-gym support
- social features
- full gym management
- complex workout-generation engine
- custom pose-estimation/computer-vision model
- advanced biomechanics system
- vector database
- production-scale infrastructure
- Kubernetes
- microservice decomposition
- unnecessary abstractions for hypothetical future scale

These require a separate post-POC decision.

---

# 17. Product Rule

The purpose of this POC is to prove the product to the gym owner.

Working functionality has priority over technical novelty.

Any proposed feature outside this document must be discussed and explicitly approved before implementation.

Approval means this document is updated first. A conversational "yes" is not sufficient authorization to implement anything that is not described here.

POC
→ gym-owner demonstration
→ decision
→ production V1
