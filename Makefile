.PHONY: all lint-broker

all: lint-broker

lint-broker:
	python3 -m py_compile genx_kernel/broker/app/main.py
