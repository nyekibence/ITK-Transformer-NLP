DATA_DIR := data
TEST_DIR := tests
SOURCE_DIR := itk_transformer_nlp
OUT_DIR := outputs

SQUAD_SAMPLE = $(DATA_DIR)/squad_sample.jsonl
$(SQUAD_SAMPLE):
	@if ! [ -d $(DATA_DIR) ]; then mkdir $(DATA_DIR); fi
	python3 itk_transformer_nlp/download_dataset.py squad $(SQUAD_SAMPLE) --split validation \
	--shuffle --sample-size 10

qa_on_squad: $(SQUAD_SAMPLE)
	@echo "Running tests and inference"
	@python3 $(TEST_DIR)/test_transformer_qa.py; \
	if [ $$? -eq 0 ]; then python3 $(SOURCE_DIR)/transformer_qa.py $(SQUAD_SAMPLE); fi
.PHONY: qa_on_squad

qa_solutions:
	@if [ -d .git ]; then git restore --source b69ae811 $(SOURCE_DIR)/transformer_qa.py; echo "OK"; \
	else echo "ERROR: You need version control to perform this action."; fi
.PHONY: qa_solutions

qa_reload_lab:
	@if [ -d .git ]; then git restore --source 43cf14c3 $(SOURCE_DIR)/transformer_qa.py; echo "OK"; \
	else echo "ERROR: You need version control to perform this action."; fi
.PHONY: qa_reload_lab

cola_fine_tune:
	@if ! [ -d $(OUT_DIR) ]; then mkdir $(OUT_DIR); fi
	@echo "Running tests and inference"
	@python3 $(TEST_DIR)/test_encoder_cola.py; \
	if [ $$? -eq 0 ]; then python3 $(SOURCE_DIR)/encoder_cola.py --model-save-path $(OUT_DIR)/hubert_cola; fi
.PHONY: cola_fine_tune
