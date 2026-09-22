export PYTHONPATH := src

.PHONY: data study model quality test all clean

data:
	python -m eventstudy.ingest_prices
	python -m eventstudy.ingest_edgar
	python -m eventstudy.build_events

study:
	python -m eventstudy.market_model
	python -m eventstudy.tests

model:
	python -m eventstudy.features
	python -m eventstudy.model

quality:
	python -m eventstudy.validate

test:
	pytest -q

all: data study model quality test

clean:
	rm -rf data outputs .pytest_cache
