REPO := $(abspath .)
export PYTHONPATH := $(REPO)

.PHONY: airflow airflow-init compose compose-down

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
