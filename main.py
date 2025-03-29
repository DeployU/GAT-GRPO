import argparse
import yaml
import numpy as np
import tensorflow as tf
import os
import random
import sys
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Network Traffic Optimization with Graph Neural Networks")
    
    # General settings
    parser.add_argument('--config', type=str, default='./config/default_config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'both'], default='both',
                        help='Operation mode: train, evaluate, or both')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (overrides config)')
    
    # Model settings
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to load/save model (overrides config)')
    
    # Environment settings
    parser.add_argument('--topology', type=int, default=None, choices=[0, 1, 2, 3],
                        help='Network topology type: 0=SMALL, 1=NSFNET, 2=GEANT2, 3=GBN (overrides config)')
    
    # Training settings
    parser.add_argument('--iterations', type=int, default=None,
                        help='Number of training iterations (overrides config)')
    
    # Evaluation settings
    parser.add_argument('--eval_episodes', type=int, default=None,
                        help='Number of evaluation episodes (overrides config)')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (overrides config)')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        print("Using default configuration")
        return {
            'seed': 42,
            'topology': {'type': 1, 'demand_values': [8, 32, 64]},
            'model': {
                'link_state_dim': 64,
                'node_state_dim': 32,
                'readout_units': 128,
                'message_passing_steps': 3,
                'attention_heads': 2
            },
            'training': {
                'learning_rate': 0.001,
                'l2_reg': 0.005,
                'dropout_rate': 0.3,
                'gamma': 0.98,
                'batch_size': 64,
                'group_size': 32,
                'clip_ratio': 0.2,
                'kl_coef': 0.01,
                'max_grad_norm': 1.0,
                'entropy_coef': 0.02,
                'min_kl': 0.01,
                'scale_factor': 50.0,
                'reward_scale': 1.5,
                'iterations': 500,
                'update_interval': 20,
                'eval_interval': 100,
                'early_stop_patience': 5
            },
            'joint_optimization': {
                'task_reward_bonus': 0.2,
                'load_penalty': 0.3
            },
            'evaluation': {
                'eval_episodes': 20,
                'eval_episodes_complex': 10,
                'topo_eval_episodes': 10,
                'evaluate_topologies': True,
                'visualize_training': True,
                'load_best_model': True
            },
            'output': {
                'save_dir': './results',
                'model_path': './best_model'
            }
        }


def override_config_with_args(config, args):
    """Override configuration with command line arguments."""
    # Override seed
    if args.seed is not None:
        config['seed'] = args.seed
    
    # Override model path
    if args.model_path is not None:
        config['output']['model_path'] = args.model_path
    
    # Override topology
    if args.topology is not None:
        config['topology']['type'] = args.topology
    
    # Override training iterations
    if args.iterations is not None:
        config['training']['iterations'] = args.iterations
    
    # Override evaluation episodes
    if args.eval_episodes is not None:
        config['evaluation']['eval_episodes'] = args.eval_episodes
    
    # Override output directory
    if args.output_dir is not None:
        config['output']['save_dir'] = args.output_dir
    
    return config


def setup_environment(seed):
    """Set up environment with proper random seeds."""
    # Set random seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Configure TensorFlow to use CPU (for environments without GPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Set logging
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Random seed: {seed}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return timestamp


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override configuration with arguments
    config = override_config_with_args(config, args)
    
    # Setup environment
    timestamp = setup_environment(config['seed'])
    
    # Create output directory with timestamp
    base_output_dir = config['output']['save_dir']
    config['output']['save_dir'] = os.path.join(base_output_dir, f"run_{timestamp}")
    os.makedirs(config['output']['save_dir'], exist_ok=True)
    
    # Save the configuration
    with open(os.path.join(config['output']['save_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Import modules
    from src.environment import NetworkEnvironment
    from src.agents.grpo_agent import GRPOAgent
    from src.train import train_and_evaluate
    from src.utils.evaluation import evaluate_agents
    
    # Create environment
    env = NetworkEnvironment(
        topology_type=config['topology']['type'],
        demand_values=config['topology']['demand_values']
    )
    
    print(f"Environment created: {env.numNodes} nodes, {env.numEdges} edges")
    
    # Prepare hyperparameters
    hparams = {}
    for section in ['model', 'training', 'joint_optimization']:
        hparams.update(config[section])
    
    # Create agent
    print("Creating GRPO agent...")
    agent = GRPOAgent(env, hparams)
    
    # Determine mode
    if args.mode == 'train' or args.mode == 'both':
        # Train and evaluate
        train_and_evaluate(agent, config)
    elif args.mode == 'evaluate':
        # Load model for evaluation
        if os.path.exists(f"{config['output']['model_path']}.weights.h5"):
            agent.load_model(config['output']['model_path'])
            print(f"Model loaded from {config['output']['model_path']}")
        else:
            print(f"Warning: Model file {config['output']['model_path']}.weights.h5 not found, using untrained model")
        
        # Create baseline agents
        from src.agents.sap_agent import SAPAgent
        from src.agents.rand_agent import RANDAgent
        sap_agent = SAPAgent()
        rand_agent = RANDAgent()
        
        # Evaluate
        print("Evaluating agents...")
        results = evaluate_agents(
            agent, sap_agent, rand_agent, env, 
            num_episodes=config['evaluation']['eval_episodes']
        )
        
        # Print results
        print("\n===== Evaluation Results =====")
        print(f"GRPO agent: {results['grpo_reward']:.4f} ± {results['grpo_std']:.4f}, success rate: {results['grpo_success']:.2f}")
        print(f"SAP agent:  {results['sap_reward']:.4f} ± {results['sap_std']:.4f}, success rate: {results['sap_success']:.2f}")
        print(f"RAND agent: {results['rand_reward']:.4f} ± {results['rand_std']:.4f}, success rate: {results['rand_success']:.2f}")
        print(f"GRPO vs SAP advantage: {results['grpo_reward'] - results['sap_reward']:.4f}")


if __name__ == "__main__":
    main()