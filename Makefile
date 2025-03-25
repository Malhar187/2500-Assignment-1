# Define variables for paths
VENV_NAME = .venv
PYTHON = python3
PIP = pip
INSTALL_REQUIREMENTS = requirements.txt

# Define tasks for environment setup, testing, training, etc.

# Task to create and activate virtual environment
env:
	@echo "Setting up virtual environment..."
	$(PYTHON) -m venv $(VENV_NAME)
	@echo "Virtual environment created: $(VENV_NAME)"
	@echo "Activate it by running: source .venv/bin/activate"

# Task to install the required dependencies into the environment
install: env
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r $(INSTALL_REQUIREMENTS)

# Task for model training
train:
	@echo "Training the model..."
	python src/train.py

# Task to build Docker images
build-docker:
	@echo "Building Docker images..."
	docker-compose build

# Task to start the application using Docker Compose
up:
	@echo "Starting application with Docker Compose..."
	docker-compose up

# Task to stop the application
down:
	@echo "Stopping application..."
	docker-compose down

# Task to clean up (e.g., removing the virtual environment)
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_NAME)

# Default task
default: install