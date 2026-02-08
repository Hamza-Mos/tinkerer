# Contributing to Tinkerer

Thank you for your interest in contributing to Tinkerer! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/tinkerer.git
   cd tinkerer
   ```

2. **Set up Python environment**
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Set up environment variables**
   - Copy `.env.example` to `.env` and fill in your API keys
   - See README.md for required secrets

## Making Changes

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests**
   ```bash
   pytest tests/
   ```

4. **Run linting** (if configured)
   ```bash
   flake8 .
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

6. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to public functions and classes
- Keep functions focused and testable

## Testing

- Add tests for new features in the `tests/` directory
- Ensure all tests pass before submitting a PR
- For smoke tests requiring API keys, use `--allow-failures` if appropriate

## Documentation

- Update README.md for user-facing changes
- Add docstrings for new public APIs
- Update inline comments for complex logic

## Questions?

Feel free to open an issue for questions or discussions about potential contributions.
