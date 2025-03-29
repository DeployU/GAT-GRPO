# Network Traffic Optimization with Graph Neural Networks

This project implements a reinforcement learning approach for optimizing network traffic routing and task execution in software-defined networks. It uses Graph Attention Networks (GAT) and Graph-structured Policy Optimization (GRPO) to jointly optimize routing paths and task execution node selection.

## Key Features

- Joint optimization of network routing and task processing node selection
- Graph Neural Network-based policy learning using Graph Attention Networks
- Multiple network topology support (NSFNET, GEANT2, small test network, GBN)
- Consideration of both link bandwidth and node processing capacity constraints
- Comparison with baseline methods (Shortest Available Path, Random)
- Comprehensive evaluation across different network conditions

## Project Structure

```
network_optimization/
├── README.md               # Project documentation
├── requirements.txt        # Package dependencies
├── config/
│   └── default_config.yaml # Hyperparameter configuration
├── src/
│   ├── environment.py      # Network environment definition
│   ├── models/
│   │   └── gat.py          # Graph Attention Network model
│   ├── agents/
│   │   ├── grpo_agent.py   # Graph-structured Policy Optimization agent
│   │   ├── sap_agent.py    # Shortest Available Path agent
│   │   └── rand_agent.py   # Random agent
│   ├── utils/
│   │   ├── evaluation.py   # Evaluation functions
│   │   └── visualization.py # Visualization tools
│   └── train.py            # Training functions
└── main.py                 # Main entry point
```

## Usage

### Basic Training and Evaluation

```bash
python main.py
```

### Customizing Configuration

You can modify hyperparameters in `config/default_config.yaml` or create custom config files.

To use a custom config file:
```bash
python main.py --config path/to/custom_config.yaml
```

### Running Specific Components

For training only:
```bash
python main.py --mode train
```

For evaluation only:
```bash
python main.py --mode evaluate --model_path path/to/model.weights.h5
```

### Output

Results will be saved to the `results/` directory, including:
- Trained models (`.weights.h5` files)
- Evaluation metrics (JSON format)
- Visualizations (PNG format)

## Method Overview

This project implements a joint optimization approach for network traffic routing and task execution:

1. **Environment**: Network with nodes and links, each with capacity constraints
2. **State**: Link bandwidth allocation and node load status
3. **Action**: Joint selection of (route, task_execution_node)
4. **Policy**: Graph Attention Network that processes the network state and outputs action probabilities
5. **Training**: Group Relative Policy Optimization (GRPO) with KL-divergence constraints
6. **Baseline Comparison**: Performance evaluation against SAP and Random agents

## License

This project is licensed under the MIT License - see the LICENSE file for details.