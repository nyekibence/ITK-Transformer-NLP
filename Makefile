DATA_DIR := data

SQUAD_SAMPLE = $(DATA_DIR)/squad_sample.jsonl
$(SQUAD_SAMPLE):
	python3 itk_transformer_nlp/download_dataset.py squad $(SQUAD_SAMPLE) --split validation \
	--shuffle --sample-size 50
