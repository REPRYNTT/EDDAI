# Contributing to E.D.D.A.I.

Thank you for your interest in contributing to the E.D.D.A.I. (Environmental Data-Driven Adaptive Intelligence) framework! This document provides guidelines and information for contributors.

## 🌟 Ways to Contribute

- **Code**: Implement new features, fix bugs, or improve existing functionality
- **Documentation**: Improve docs, write tutorials, or create examples
- **Testing**: Add unit tests, integration tests, or performance benchmarks
- **Research**: Explore new algorithms, ecological models, or applications
- **Feedback**: Report bugs, suggest features, or share use cases

## 🚀 Getting Started

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/eddai-framework.git
   cd eddai-framework
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```

5. Run tests to ensure everything works:
   ```bash
   pytest tests/
   ```

### Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following the coding standards below

3. Add tests for new functionality

4. Run the test suite:
   ```bash
   pytest tests/
   ```

5. Update documentation if needed

6. Commit your changes:
   ```bash
   git commit -m "Add: Brief description of your changes"
   ```

7. Push to your fork and create a pull request

## 📝 Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions, classes, and modules
- Use descriptive variable and function names

### Code Formatting

We use [Black](https://black.readthedocs.io/) for code formatting:

```bash
black eddai_framework/
```

### Linting

Run the linter to check for issues:

```bash
flake8 eddai_framework/
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_eddai.py

# Run with coverage
pytest --cov=eddai_framework --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names that explain what they're testing
- Test both success and failure cases
- Use fixtures for common test setup

Example test structure:
```python
import unittest
from eddai import EDDAI

class TestEDDAI(unittest.TestCase):
    def setUp(self):
        self.eddai = EDDAI(biome_id="test")

    def test_initialization(self):
        """Test that EDDAI initializes correctly."""
        self.assertEqual(self.eddai.biome_id, "test")
        # ... more assertions
```

## 📚 Documentation

### Building Docs

```bash
cd docs
make html
```

### Documentation Standards

- Use Google-style docstrings
- Include examples in docstrings where helpful
- Keep the README up to date
- Add API documentation for new features

## 🔬 Research Contributions

### Adding New Ecological Models

1. Implement the model in `simulation/ecological_models/`
2. Add configuration options
3. Update the simulation framework
4. Add tests and documentation

### New Sensor Types

1. Extend the `EnvironmentalSensorium` class
2. Add data processing methods
3. Update the sensor configuration
4. Add tests for the new sensor type

### Algorithm Improvements

1. Implement the algorithm in the appropriate module
2. Compare performance with existing approaches
3. Add benchmarks and tests
4. Update documentation

## 🐛 Reporting Issues

When reporting bugs or requesting features:

1. Check if the issue already exists
2. Use a clear, descriptive title
3. Provide steps to reproduce the issue
4. Include relevant code snippets, error messages, and system information
5. Suggest potential solutions if possible

## 📋 Pull Request Guidelines

### Before Submitting

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] Changes are focused and atomic

### PR Description

Include:
- What the change does
- Why it's needed
- How it was tested
- Any breaking changes

### Review Process

1. Automated checks (tests, linting) run
2. Code review by maintainers
3. Discussion and iteration
4. Merge when approved

## 🎯 Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Show empathy towards other contributors
- Help create a positive community

## 📞 Getting Help

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Discord/Slack**: For real-time chat (if available)

## 🙏 Recognition

Contributors are recognized in:
- The CHANGELOG for significant contributions
- GitHub's contributor insights
- Academic publications (where applicable)

Thank you for contributing to E.D.D.A.I.! 🌱
