REPO := $(abspath .)
export PYTHONPATH := $(REPO)

.PHONY: airflow airflow-init compose compose-down pull-winner demo-predict drift-reference drift-score report figures

airflow-init:
	mkdir -p $(REPO)/.airflow
	AIRFLOW_HOME=$(REPO)/.airflow \
		AIRFLOW__CORE__EXECUTOR=LocalExecutor \
		AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:///$(REPO)/.airflow/airflow.db \
		AIRFLOW__CORE__DAGS_FOLDER=$(REPO)/airflow/dags \
		AIRFLOW__CORE__LOAD_EXAMPLES=false \
		$(REPO)/.venv/bin/airflow db migrate

airflow:
	bash $(REPO)/scripts/run_airflow.sh

compose:
	docker compose --env-file .env -f docker/docker-compose.yml up --build -d

compose-down:
	docker compose --env-file .env -f docker/docker-compose.yml down

pull-winner:
	$(REPO)/.venv/bin/python -m src.data_processing.versioning pull-winner $(REPO)/models

demo-predict:
	$(REPO)/.venv/bin/python -m src.deployment.demo --url http://localhost:8000

LOG ?= predictions.jsonl

drift-reference:
	$(REPO)/.venv/bin/python -m src.monitoring.drift_cli --config $(REPO)/config/config.yaml build-reference --with-features

drift-score:
	$(REPO)/.venv/bin/python -m src.monitoring.drift_cli --config $(REPO)/config/config.yaml score-predictions $(LOG)

report: figures
	cd $(REPO)/report && pdflatex -interaction=nonstopmode main && bibtex main && pdflatex -interaction=nonstopmode main && pdflatex -interaction=nonstopmode main

figures:
	@command -v npx >/dev/null || (echo "npx required to render Mermaid figures" && exit 1)
	cd $(REPO)/report/figures && \
	  PUPPETEER_CFG=$$(mktemp) && \
	  echo '{"args":["--no-sandbox","--disable-setuid-sandbox"]}' > $$PUPPETEER_CFG && \
	  for name in architecture dag lifecycle; do \
	    npx -y @mermaid-js/mermaid-cli@11 \
	      -i $${name}.mmd -o $${name}.png \
	      -t neutral -b white -w 1800 -s 2 \
	      -p $$PUPPETEER_CFG; \
	  done && rm -f $$PUPPETEER_CFG
