# E.D.D.A.I. - Environmental Data-Driven Adaptive Intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**E.D.D.A.I.** is an open-source framework implementing a novel paradigm for artificial intelligence that treats the environment as its primary training substrate, co-evolutionary partner, and performance evaluator. This system enables AI to live within ecosystems, evolving and adapting alongside natural processes.

## 🌟 Vision

Traditional AI optimizes against fixed objectives in controlled environments. E.D.D.A.I. embeds ecological dynamics into its core learning loop, creating intelligence that:

- **Observes** real-time environmental patterns
- **Adapts** its architecture and behavior to ecological changes
- **Acts** symbiotically with natural systems
- **Learns** from multi-species outcomes
- **Remembers** disturbance histories epigenetically

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Environmental   │    │   Plastic       │    │ Epigenetic      │
│ Sensorium       │───▶│   Neural       │───▶│ Place Memory    │
│                 │    │   Fabric       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Symbiotic       │    │     EDDAI       │    │   Action        │
│ Objective       │    │   Orchestrator  │    │   Layer         │
│ Function        │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Core Components

1. **Environmental Sensorium**: Multimodal sensory interface processing bioacoustic, hyperspectral, edaphic, atmospheric, and phenocam data.

2. **Plastic Neural Fabric**: Self-rewiring neural architectures that physically deform in response to environmental gradients.

3. **Epigenetic Place Memory**: Long-term spatiotemporal embeddings encoding disturbance history and ecological knowledge.

4. **Symbiotic Objective Function**: Dynamic utility model balancing human, biotic, and abiotic goals with ethical guardrails.

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/eddai-framework.git
cd eddai-framework
pip install -r requirements.txt
```

### Basic Usage

```python
from eddai import EDDAI
from simulation import VirtualForest

# Initialize virtual ecosystem
forest = VirtualForest(grid_size=(20, 20), biome="temperate_forest")

# Initialize E.D.D.A.I.
eddai = EDDAI(biome_id="temperate_forest")

# Run adaptive cycle
for step in range(100):
    forest_state = forest.step()
    eddai_result = eddai.step(forest_state['environment'])
    outcomes = forest.apply_action(eddai_result['action'])

    print(f"Step {step}: Action={eddai_result['action']['type']}, "
          f"Biodiversity={forest_state['metrics']['biodiversity_index']:.3f}")
```

### 🌐 Web Dashboard (NEW!)

**Launch the interactive web app:**

```bash
# Install Flask
pip install flask

# Run the web dashboard
python app.py
```

Then open http://localhost:5000 in your browser to access the AI-powered environmental dashboard!

**Deploy to Vercel:**

1. Push your code to GitHub
2. Connect your GitHub repo to Vercel
3. Deploy automatically - Vercel handles everything!

### Running Demos

```bash
# Basic adaptation demo
python examples/basic_demo.py --steps 50 --visualize

# Disturbance response demo
python examples/disturbance_response_demo.py

# Run all tests
python -m pytest tests/ -v
```

## 📊 Key Features

### Symbiotic Intelligence
- **Multi-species optimization**: Balances human yield, biodiversity, and ecosystem stability
- **Ethical guardrails**: Built-in vetoes prevent harmful actions
- **Adaptive objectives**: SOF weights adjust based on ecological feedback

### Environmental Adaptation
- **Topological plasticity**: Neural networks rewire based on environmental gradients
- **Disturbance memory**: Remembers droughts, fires, and climate shifts
- **Real-time sensing**: Processes multimodal ecological data streams

### Open Ecosystem
- **Modular design**: Easy to extend with new sensors or biomes
- **Simulation framework**: Test in virtual environments before deployment
- **Research-friendly**: Comprehensive logging and analysis tools

### 🌐 Interactive Web Dashboard
- **Real-time AI Analysis**: Live environmental monitoring with AI insights
- **Interactive Simulations**: Test drought, flood, and fire scenarios
- **AI Recommendations**: Get intelligent suggestions for ecosystem management
- **Visual Analytics**: Charts showing biodiversity, carbon flux, and disturbance trends
- **Educational Tool**: Learn about ecological systems and AI decision-making

## 🧪 Simulation Environment

The framework includes a sophisticated virtual forest ecosystem for testing:

```python
from simulation import VirtualForest, DroughtEvent, FireEvent

forest = VirtualForest(grid_size=(20, 20), biome="amazon_rainforest")

# Add environmental disturbances
drought = DroughtEvent(intensity=0.8, duration_days=14)
forest.add_disturbance(drought)

# Visualize ecosystem state
forest.visualize(save_path="forest_state.png")
```

## 🔬 Research Applications

- **Conservation AI**: Adaptive wildlife monitoring and habitat management
- **Climate-resilient agriculture**: Self-adjusting crop management systems
- **Urban ecology**: Smart city ecosystems that respond to environmental changes
- **Biodiversity monitoring**: Real-time assessment and intervention systems

## 🛠️ Technical Details

### Requirements
- Python 3.8+
- PyTorch 1.9+
- NumPy
- Matplotlib (for visualization)

### Dependencies
```
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.5.0
```

### Architecture Details
- **Neural Networks**: PyTorch-based plastic architectures
- **Memory System**: Vector embeddings with attention mechanisms
- **Optimization**: Multi-objective reinforcement learning
- **Simulation**: Grid-based ecosystem dynamics

## 📚 Documentation

### API Reference
- [EDDAI Class](docs/api/eddai.md)
- [Environmental Sensorium](docs/api/sensorium.md)
- [Plastic Neural Fabric](docs/api/neural_fabric.md)
- [Epigenetic Memory](docs/api/place_memory.md)
- [Symbiotic Objectives](docs/api/sof.md)

### Tutorials
- [Getting Started](docs/tutorials/getting_started.md)
- [Custom Biomes](docs/tutorials/custom_biomes.md)
- [Real Deployment](docs/tutorials/real_deployment.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/yourusername/eddai-framework.git
cd eddai-framework
pip install -e .
pip install -r requirements-dev.txt
```

### Running Tests
```bash
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This framework builds upon concepts from:
- Biomimicry and ecological systems theory
- Neuromorphic computing and plastic neural networks
- Multi-objective reinforcement learning
- Environmental monitoring and conservation technology

## 🔗 Links

- **Paper**: [Environmental Data-Driven Adaptive Intelligence](https://arxiv.org/abs/...)
- **Documentation**: [Read the Docs](https://eddai-framework.readthedocs.io/)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/eddai-framework/discussions)

---

**E.D.D.A.I.**: *AI as ecosystem, not just in ecosystem.*
