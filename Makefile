PYTHON ?= python

.PHONY: install third-party test smoke-data train infer

install:
	PY="$(PYTHON)" bash scripts/setup_env.sh

third-party:
	PY="$(PYTHON)" bash scripts/install_third_party.sh

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -q -p no:cacheprovider

smoke-data:
	$(PYTHON) -m llm_sana.data.prepare_llavarad_prompt_parquet \
		--train_csv examples/train_example.csv \
		--val_csv examples/val_example.csv \
		--output_dir /tmp/justllmgrpo_example \
		--balanced_val_per_label 0

train:
	bash scripts/train_justllmgrpo.sh

infer:
	bash scripts/infer.sh
