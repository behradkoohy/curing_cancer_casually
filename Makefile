VENV_NAME := venv
PYTHON_VERSION := python

.DEFAULT: help

help:
	@echo “make help - display this help message”
	@echo “make venv - create a new Python virtual environment”
venv:
	$(PYTHON_VERSION) -m venv $(VENV_NAME)
	@echo “Run ‘source $(VENV_NAME)/bin/activate’ to activate the virtual environment”

