.PHONY: up down restart logs

up:
	docker compose up --build -d
	docker compose logs -f chatbot app

down:
	docker compose down

restart: down up

logs:
	docker compose logs -f chatbot app
