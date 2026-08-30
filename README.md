# Barbarik Fitness Pal

Phase 0 scaffold for Barbarik AI, the WhatsApp-first fitness assistant included with Barbarik Gym membership.

## Included

- FastAPI application shell and SQLAlchemy 2.0 models
- Alembic initial migration for the Phase 0 PostgreSQL schema
- PostgreSQL 15 with `pgvector`, Redis, and Docker Compose
- Idempotent demo-data seed script: three members, 100 exercises, plans, four weeks of history, and trend flags
- React/Vite/Tailwind PWA shell (no dashboard features yet)

## Prerequisites

- Docker Compose v2 for the container workflow
- Python 3.11 if running the seed script from the host
- Node 20 for local dashboard development

## Quick start

```bash
cp .env.example .env
docker compose up -d postgres redis
docker compose run --rm backend alembic upgrade head
docker compose run --rm backend python scripts/seed_demo_data.py
docker compose up --build backend dashboard
```

The requested host form also works once the backend Python dependencies are installed. The example `DATABASE_URL` is already configured for the Compose-exposed local PostgreSQL port:

```bash
python backend/scripts/seed_demo_data.py
```

Demo accounts: `rajesh.demo@barbarik.local`, `priya.demo@barbarik.local`, and `amit.demo@barbarik.local`. Their password hashes are intentionally non-production placeholders; authentication is deferred to Phase 1.

## Scope boundary

MCP tools, NLP parsing, WhatsApp webhooks, dashboard features, and video analysis are intentionally not implemented in this phase.
