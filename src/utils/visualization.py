import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_training_history(history, output_dir="./results"):
    """
    Create visualization for training history.
    
    Args:
        history: Dictionary containing training history
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 8))
    
    # Plot training rewards
    plt.subplot(2, 2, 1)
    plt.plot(history.get('train_rewards', []))
    plt.title('Training Rewards')
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.grid(True)
    
    # Plot evaluation rewards
    plt.subplot(2, 2, 2)
    eval_iterations = np.arange(len(history.get('eval_grpo_rewards', []))) * \
                    history.get('eval_interval', 50)
    plt.plot(eval_iterations, history.get('eval_grpo_rewards', []), 'r-', label='GRPO')
    plt.plot(eval_iterations, history.get('eval_sap_rewards', []), 'b-', label='SAP')
    plt.title('Evaluation Rewards')
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot relative performance
    plt.subplot(2, 2, 3)
    plt.plot(eval_iterations, history.get('relative_performance', []), 'g-')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
    plt.title('GRPO vs SAP Performance Gap')
    plt.xlabel('Iteration')
    plt.ylabel('Reward Gap')
    plt.grid(True)
    
    # Plot loss
    plt.subplot(2, 2, 4)
    plt.plot(history.get('loss_history', []), 'r-', alpha=0.7)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png")
    plt.close()
    
    print(f"Training history visualization saved to {output_dir}/training_history.png")


def visualize_comparison_results(comparison_results, output_dir="./results"):
    """
    Visualize comparison results across different topologies.
    
    Args:
        comparison_results: Dictionary of comparison results
        output_dir: Directory to save visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create chart
    plt.figure(figsize=(15, 10))
    
    # Extract data
    topologies = list(comparison_results.keys())
    scenarios = ['standard', 'complex']
    metrics = ['reward', 'success']
    agents = ['grpo', 'sap', 'rand']
    
    n_topologies = len(topologies)
    n_scenarios = len(scenarios)
    
    # Plot reward comparison
    plt.subplot(2, 1, 1)
    bar_width = 0.2
    index = np.arange(n_topologies)
    
    for i, scenario in enumerate(scenarios):
        offset = (i - 0.5) * bar_width * 3
        
        grpo_values = [comparison_results[topo][scenario]['grpo_reward'] for topo in topologies]
        sap_values = [comparison_results[topo][scenario]['sap_reward'] for topo in topologies]
        rand_values = [comparison_results[topo][scenario]['rand_reward'] for topo in topologies]
        
        plt.bar(index + offset, grpo_values, bar_width, label=f'GRPO ({scenario})', 
                alpha=0.8, color=['#1f77b4', '#2ca02c'][i])
        plt.bar(index + offset + bar_width, sap_values, bar_width, label=f'SAP ({scenario})', 
                alpha=0.8, color=['#ff7f0e', '#d62728'][i])
        plt.bar(index + offset + 2*bar_width, rand_values, bar_width, label=f'RAND ({scenario})', 
                alpha=0.8, color=['#9467bd', '#8c564b'][i])
    
    plt.xlabel('Topology')
    plt.ylabel('Average Reward')
    plt.title('Agent Performance Comparison Across Topologies (Reward)')
    plt.xticks(index, topologies)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot success rate comparison
    plt.subplot(2, 1, 2)
    
    for i, scenario in enumerate(scenarios):
        offset = (i - 0.5) * bar_width * 3
        
        grpo_values = [comparison_results[topo][scenario]['grpo_success'] for topo in topologies]
        sap_values = [comparison_results[topo][scenario]['sap_success'] for topo in topologies]
        rand_values = [comparison_results[topo][scenario]['rand_success'] for topo in topologies]
        
        plt.bar(index + offset, grpo_values, bar_width, label=f'GRPO ({scenario})', 
                alpha=0.8, color=['#1f77b4', '#2ca02c'][i])
        plt.bar(index + offset + bar_width, sap_values, bar_width, label=f'SAP ({scenario})', 
                alpha=0.8, color=['#ff7f0e', '#d62728'][i])
        plt.bar(index + offset + 2*bar_width, rand_values, bar_width, label=f'RAND ({scenario})', 
                alpha=0.8, color=['#9467bd', '#8c564b'][i])
    
    plt.xlabel('Topology')
    plt.ylabel('Average Success Rate')
    plt.title('Agent Performance Comparison Across Topologies (Success Rate)')
    plt.xticks(index, topologies)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/topology_comparison.png")
    plt.close()
    
    # Plot GRPO advantage over SAP
    plt.figure(figsize=(12, 6))
    
    for i, scenario in enumerate(scenarios):
        advantages = [comparison_results[topo][scenario]['grpo_reward'] - 
                     comparison_results[topo][scenario]['sap_reward'] for topo in topologies]
        
        plt.bar(index + i*bar_width, advantages, bar_width, 
                label=f'{scenario.capitalize()} scenario', alpha=0.8, 
                color=['#1f77b4', '#2ca02c'][i])
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.5)
    plt.xlabel('Topology')
    plt.ylabel('GRPO vs SAP Advantage (Reward Gap)')
    plt.title('GRPO Performance Advantage over SAP')
    plt.xticks(index + bar_width/2, topologies)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/grpo_advantage.png")
    plt.close()
    
    print(f"Comparison results visualization saved to {output_dir}")


def visualize_network_topology(env, output_dir="./results"):
    """
    Visualize network topology with node and link capacities.
    
    Args:
        env: Network environment
        output_dir: Directory to save visualization
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    
    # Create graph layout
    pos = {}
    for node in env.nodes:
        # Position nodes in a circle layout
        angle = 2 * np.pi * node / len(env.nodes)
        pos[node] = [np.cos(angle), np.sin(angle)]
    
    # Plot nodes
    node_sizes = env.node_capacity / np.max(env.node_capacity) * 1000
    node_colors = env.node_load / (env.node_capacity + 1e-10)  # Normalize to [0,1]
    
    # Plot edges
    for edge in env.edges:
        i, j = edge
        plt.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]], 'k-', alpha=0.3, linewidth=1)
    
    # Plot nodes
    sc = plt.scatter([pos[i][0] for i in env.nodes], [pos[i][1] for i in env.nodes], 
                    s=node_sizes, c=node_colors, cmap='coolwarm', vmin=0, vmax=1)
    
    # Add node labels
    for node in env.nodes:
        plt.text(pos[node][0], pos[node][1], str(node), 
                ha='center', va='center', fontweight='bold')
    
    plt.colorbar(sc, label='Node Load Ratio')
    
    # Show topology type in title
    topology_names = {0: "Small Test Network", 1: "NSFNET", 2: "GEANT2", 3: "GBN"}
    topology_name = topology_names.get(env.topology_type, "Custom")
    
    plt.title(f"{topology_name} Topology\n{env.numNodes} nodes, {env.numEdges} links")
    plt.axis('equal')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/network_topology.png")
    plt.close()
    
    print(f"Network topology visualization saved to {output_dir}/network_topology.png")