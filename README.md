# Ignat: Telegram-Colab ML Orchestration Bot

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Aiogram 3.4+](https://img.shields.io/badge/aiogram-3.4+-green.svg)](https://docs.aiogram.dev/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Ignat is an enterprise-grade Telegram bot engineered to orchestrate machine learning inference workflows. By leveraging the [Google Colab CLI](https://github.com/googlecolab/colab-cli), the application securely bridges user media input on Telegram with remote GPU/TPU environments in Google Colab.

The system natively handles media ingestion, payload standardization (converting arbitrary images to JPEG and videos to GIF), subprocess orchestration for remote execution, and structured response delivery.

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Key Capabilities](#key-capabilities)
3. [Prerequisites](#prerequisites)
4. [Installation & Setup](#installation--setup)
5. [Configuration](#configuration)
6. [Execution](#execution)
7. [Testing Framework](#testing-framework)
8. [Project Structure](#project-structure)

---

## System Architecture

The application is built on a modular, event-driven architecture utilizing `aiogram` 3.x. The processing pipeline follows a unidirectional data flow:

1. **Ingestion**: Telegram updates are parsed; media payloads (Photos, Videos, Documents) are downloaded to a secure temporary local cache.
2. **Preprocessing**: The `converter` module ensures all media conforms to target ML pipeline requirements (`.jpg` for static frames, `.gif` for sequential frames) using `FFmpeg` and `Pillow`.
3. **Orchestration**: The `colab_api` module initializes a subprocess interface with `google-colab-cli`, maintaining persistent sessions or executing stateless commands as required.
4. **Delivery**: ML outputs are parsed and routed back to the end-user via Telegram's SendMedia API.

## Key Capabilities

* **Automated Media Standardization**: Graceful handling of edge cases in media dimensions, codecs, and container formats.
* **Remote ML Execution**: Direct subprocess bindings to Colab CLI for seamless remote GPU allocation.
* **Resilient Error Handling**: Graceful degradation, detailed user-facing error reporting, and retry mechanisms.
* **Enterprise Logging**: File-based rotation, structured output, and isolated debug trails for auditing.
* **Extensive Testing Suite**: Over 150 automated unit and integration tests with complete mocking of network and external binary dependencies.

## Prerequisites

Ensure the following dependencies are provisioned in your host environment before deploying the application:

* **Python**: `3.9` or higher.
* **FFmpeg**: Required for robust video-to-GIF conversion and codec manipulation.
  * Debian/Ubuntu: `sudo apt install ffmpeg`
  * macOS: `brew install ffmpeg`
  * CentOS/RHEL: `sudo yum install ffmpeg`
* **Google Colab CLI**: Must be installed and authenticated (`google-colab-cli` binary in `$PATH`).

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd ignat
   ```

2. **Initialize the Virtual Environment**
   It is highly recommended to isolate the application dependencies.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Configuration

The application utilizes runtime environment variables for configuration management. A template file (`.env.example`) is provided.

1. Copy the template:
   ```bash
   cp .env.example .env
   ```

2. Modify `.env` with your deployment specifics:
   *Note: Ensure the bot token is securely provisioned and `DEBUG` is set to `False` in production environments.*

## Execution

To initialize the application daemon:

```bash
python main.py
```

The system will bootstrap the logging configuration, validate asset paths, initialize the Colab session pool, and begin long-polling for Telegram updates. 

## Testing Framework

The project is backed by a robust `pytest` suite designed for CI/CD integration. The tests mock the Telegram API and FFmpeg binaries, allowing verification of business logic without external dependencies or network calls.

**Execute the standard test suite:**
```bash
python -m pytest tests/
```

**Execute with detailed verbosity and stdout capturing:**
```bash
python -m pytest tests/ -v -s
```

**Execute coverage analysis (requires `pytest-cov`):**
```bash
pip install pytest-cov
python -m pytest tests/ --cov=bot
```

## Project Structure

```text
ignat/
├── main.py                 # Application bootstrap and entry point
├── pytest.ini              # Test suite configuration and markers
├── requirements.txt        # Frozen Python dependencies
├── .env.example            # Environment variable template
├── .gitignore              # Source control exclusion rules
├── README.md               # Project documentation
├── bot/                    # Core business logic package
│   ├── colab_api.py        # Subprocess manager for google-colab-cli
│   ├── config.py           # Configuration schema and state
│   ├── converter.py        # Media codec and formatting utilities
│   ├── handlers.py         # Telegram update routing and logic
│   ├── logging_config.py   # System-wide logging formatter
│   ├── media.py            # Local file I/O and payload preparation
│   ├── parser.py           # Colab stdout/stderr parser
│   └── assets/             # Static configuration payloads (JSON)
└── tests/                        # Automated testing directory
    ├── conftest.py               # Pytest fixtures and mock objects
    ├── test_colab_api.py         # Colab API unit tests
    ├── test_config.py            # Configuration schema and state tests
    ├── test_converter.py         # FFmpeg / Pillow abstraction tests
    ├── test_handlers.py          # Pure helper logic tests
    ├── test_handlers_extended.py # Core route and command tests
    ├── test_handlers_coverage.py # Edge-case logic and coverage tests
    ├── test_logging_config.py    # System-wide logging formatter tests
    └── test_media.py             # I/O edge-case tests
```
