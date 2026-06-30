.PHONY: test run clean

test:
	python -m pytest test_surfalgo.py -v || python test_surfalgo.py

run:
	python main.py

clean:
	rm -rf __pycache__ .pytest_cache
