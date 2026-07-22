# ─────────────────────────────────────────────────────────────────────────────
# Industrial Smart Grid AI — Makefile
# Run any command with: make <target>
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help up down build logs ps clean restart shell-backend shell-frontend \
        up-sim up-server-only up-prod train-models

# Default target
help:
	@echo ""
	@echo "  ⚡ Industrial Smart Grid AI"
	@echo ""
	@echo "  DOCKER COMMANDS"
	@echo "  ─────────────────────────────────────────────────────"
	@echo "  make up            Start everything (backend + frontend + 3 simulated substations)"
	@echo "  make up-sim        Same as 'up' — explicit simulation mode"
	@echo "  make up-server-only  Start only backend + frontend (connect real hardware manually)"
	@echo "  make up-prod       Start everything + Streamlit dashboard"
	@echo "  make down          Stop all containers"
	@echo "  make build         Rebuild all images (after code changes)"
	@echo "  make restart       Rebuild and restart everything"
	@echo "  make logs          Follow logs from all containers"
	@echo "  make ps            Show running containers and their status"
	@echo "  make clean         Stop containers and remove volumes (fresh start)"
	@echo ""
	@echo "  DEVELOPMENT COMMANDS"
	@echo "  ─────────────────────────────────────────────────────"
	@echo "  make train-models  Train all ML models and save to disk"
	@echo "  make shell-backend Open a shell inside the backend container"
	@echo "  make ports         Show which ports are in use"
	@echo ""
	@echo "  REAL HARDWARE"
	@echo "  ─────────────────────────────────────────────────────"
	@echo "  make up-server-only   then run on your machine:"
	@echo "    python substations/substation_client.py --id S1 --host localhost"
	@echo "    python substations/substation_client.py --id S2 --host localhost --source adb"
	@echo ""

# ── Start everything with simulated substations ───────────────────────────────
up:
	docker-compose up --build -d
	@echo ""
	@echo "  ✅ Started!"
	@echo "  React Dashboard  →  http://localhost"
	@echo "  API Docs         →  http://localhost:8000/docs"
	@echo "  Socket port      →  9999"
	@echo ""
	@echo "  Run 'make logs' to follow logs"

up-sim: up

# ── Start only backend + frontend (for real hardware) ─────────────────────────
up-server-only:
	docker-compose up --build -d backend frontend
	@echo ""
	@echo "  ✅ Server started!"
	@echo "  React Dashboard  →  http://localhost"
	@echo "  API Docs         →  http://localhost:8000/docs"
	@echo "  Socket port      →  9999 (connect real hardware here)"
	@echo ""
	@echo "  Connect your devices:"
	@echo "    python substations/substation_client.py --id S1 --host localhost"
	@echo "    python substations/substation_client.py --id S2 --host localhost --source adb"

# ── Start with Streamlit too ──────────────────────────────────────────────────
up-prod:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d
	@echo ""
	@echo "  ✅ Full stack started!"
	@echo "  React Dashboard     →  http://localhost"
	@echo "  Streamlit Dashboard →  http://localhost:8501"
	@echo "  API Docs            →  http://localhost:8000/docs"

# ── Stop ──────────────────────────────────────────────────────────────────────
down:
	docker-compose down
	@echo "  Stopped all containers."

# ── Rebuild images ────────────────────────────────────────────────────────────
build:
	docker-compose build --no-cache

# ── Rebuild and restart ───────────────────────────────────────────────────────
restart:
	docker-compose down
	docker-compose up --build -d
	@echo "  Restarted."

# ── Follow logs ───────────────────────────────────────────────────────────────
logs:
	docker-compose logs -f

logs-backend:
	docker-compose logs -f backend

logs-frontend:
	docker-compose logs -f frontend

# ── Show container status ─────────────────────────────────────────────────────
ps:
	docker-compose ps

# ── Clean everything (removes volumes = ML models will retrain) ───────────────
clean:
	docker-compose down -v --remove-orphans
	@echo "  Cleaned. ML models will retrain on next start."

# ── Shell access ──────────────────────────────────────────────────────────────
shell-backend:
	docker exec -it smartgrid-backend /bin/bash

shell-frontend:
	docker exec -it smartgrid-frontend /bin/sh

# ── Train ML models ───────────────────────────────────────────────────────────
train-models:
	docker exec -it smartgrid-backend python ml_models/anomaly_detection/model_trainer.py
	@echo "  Models saved to volume."

# ── Show ports ────────────────────────────────────────────────────────────────
ports:
	@echo "  Port 80   → React Dashboard (nginx)"
	@echo "  Port 8000 → FastAPI REST API"
	@echo "  Port 9999 → TCP Socket (substation telemetry)"
	@echo "  Port 8501 → Streamlit (only with make up-prod)"
