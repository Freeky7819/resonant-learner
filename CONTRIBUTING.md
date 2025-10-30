# Contributing to Resonant Learner

Thank you for your interest in contributing! 💙

We welcome contributions of all kinds:
- 🐛 Bug reports
- 📝 Documentation improvements
- ✨ New features
- 🧪 Additional examples
- 🔬 Testing on new datasets

---

## Getting Started

1. **Fork the repository**
2. **Clone your fork:**
   ```bash
   git clone https://github.com/yourusername/resonant-learner.git
   cd resonant-learner
   ```
3. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

---

## Development Workflow

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**

3. **Test your changes:**
   ```bash
   # Run examples
   python examples/mnist_rca.py --epochs 5
   
   # Run tests (when available)
   pytest tests/
   ```

4. **Commit:**
   ```bash
   git add .
   git commit -m "Add: your feature description"
   ```

5. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   ```

---

## Code Style

- Follow PEP 8
- Use type hints where possible
- Add docstrings to all public functions
- Keep lines under 100 characters

---

## Areas Where We Need Help

### High Priority
- 🧪 Testing on more datasets (ImageNet, COCO, etc.)
- 📝 Documentation improvements
- 🐛 Bug fixes

### Medium Priority
- 🔬 Integration with other frameworks (TensorFlow, JAX)
- 📊 More examples (NLP, RL, etc.)
- ⚡ Performance optimizations

### Future Ideas
- 🌐 Web dashboard for visualization
- 🎨 Better logging and monitoring
- 📦 Easier integration with popular libraries

---

## Reporting Bugs

Use [GitHub Issues](https://github.com/yourusername/resonant-learner/issues) and include:
- Python version
- PyTorch version
- Minimal code to reproduce
- Expected vs actual behavior
- Error messages (if any)

---

## Suggesting Features

Open a [GitHub Discussion](https://github.com/yourusername/resonant-learner/discussions) to discuss:
- What problem does it solve?
- How would it work?
- Any implementation ideas?

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Questions?

Feel free to:
- Open a [Discussion](https://github.com/yourusername/resonant-learner/discussions)
- Email: your.email@example.com

Thank you for making Resonant Learner better! 🌊
