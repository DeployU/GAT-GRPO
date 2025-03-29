import numpy as np
import os
import json
from tqdm import tqdm

def evaluate_agents(grpo_agent, sap_agent, rand_agent, env, num_episodes=10, 
                   eval_mode='standard', print_log=True):
    """
    Evaluate performance of three agents: GRPO, SAP, and RAND.
    
    Args:
        grpo_agent: Graph-structured Policy Optimization agent
        sap_agent: Shortest Available Path agent
        rand_agent: Random agent
        env: Network environment
        num_episodes: Number of episodes to evaluate
        eval_mode: Evaluation mode ('standard', 'complex', or 'simple')
        print_log: Whether to print evaluation logs
        
    Returns:
        dict: Evaluation results
    """
    # Based on evaluation mode, select different environment configuration
    if eval_mode == 'complex':
        # Complex scenario evaluation - higher bandwidth demands, more complex topology, higher load factors
        from src.environment import NetworkEnvironment, TOPOLOGY_GEANT2
        
        eval_env = NetworkEnvironment(topology_type=TOPOLOGY_GEANT2, 
                                    demand_values=[16, 32, 64])
        
        # Increase node load factors to make task execution more challenging
        eval_env.load_factors = np.random.uniform(1.5, 2.5, eval_env.numNodes)
        
        # Pre-load network to create a more complex decision environment
        state, _, _, _ = eval_env.reset()
        
        # Simulate some existing traffic allocations and task executions
        pre_allocate_count = min(5, eval_env.numNodes // 2)
        for _ in range(pre_allocate_count):
            # Randomly select source-destination nodes and demand
            src = np.random.choice(eval_env.nodes)
            dst = np.random.choice([n for n in eval_env.nodes if n != src])
            demand_val = np.random.choice(eval_env.listofDemands)
            
            # Try to allocate path and task
            path_key = f"{src}:{dst}"
            if path_key in eval_env.allPaths and eval_env.allPaths[path_key]:
                path = eval_env.allPaths[path_key][0]  # Use shortest path
                
                # Try to allocate bandwidth for path
                can_allocate_path = True
                for i in range(len(path) - 1):
                    edge_key = f"{path[i]}:{path[i+1]}"
                    if edge_key not in eval_env.edgesDict:
                        can_allocate_path = False
                        break
                    edge_idx = eval_env.edgesDict[edge_key]
                    if eval_env.capacity[edge_idx] < demand_val:
                        can_allocate_path = False
                        break
                
                if can_allocate_path:
                    # Allocate bandwidth for path
                    for i in range(len(path) - 1):
                        edge_key = f"{path[i]}:{path[i+1]}"
                        edge_idx = eval_env.edgesDict[edge_key]
                        eval_env.capacity[edge_idx] -= demand_val
                        eval_env.bw_allocated[edge_idx] = demand_val
                    
                    # Select intermediate node to execute task
                    valid_nodes = list(range(1, len(path) - 1))
                    if valid_nodes:
                        task_node_idx = valid_nodes[np.random.randint(0, len(valid_nodes))]
                        task_node = path[task_node_idx]
                        task_load = demand_val * eval_env.load_factors[task_node]
                        
                        # Reduce node capacity
                        if eval_env.node_capacity[task_node] >= task_load:
                            eval_env.node_capacity[task_node] -= task_load
                            eval_env.node_load[task_node] += task_load
                
    elif eval_mode == 'simple':
        # Simple scenario evaluation
        from src.environment import NetworkEnvironment, TOPOLOGY_SMALL
        
        eval_env = NetworkEnvironment(topology_type=TOPOLOGY_SMALL, 
                                    demand_values=[8, 16, 32])
        # Use lower load factors
        eval_env.load_factors = np.random.uniform(1.0, 1.5, eval_env.numNodes)
        # Increase node capacity
        eval_env.node_capacity = np.full(eval_env.numNodes, 500.0)
    else:  # standard
        # Standard evaluation
        eval_env = env
    
    # Evaluation results
    grpo_rewards = []
    sap_rewards = []
    rand_rewards = []
    
    # Track success rates (proportion of successful bandwidth and task allocations)
    grpo_success = 0
    sap_success = 0
    rand_success = 0
    
    for ep in range(num_episodes):
        if print_log and ep % 5 == 0:
            print(f"Evaluating episode {ep}/{num_episodes}...")
        
        # Use exactly the same starting state for all agents
        state, demand, source, destination = eval_env.reset()
        
        # Create three independent environment copies
        envs = {}
        states = {}
        
        for agent_name in ['grpo', 'sap', 'rand']:
            # Create environment copy
            agent_env = eval_env.__class__(topology_type=eval_env.topology_type, 
                                         demand_values=eval_env.listofDemands)
            
            # Copy environment state
            agent_env.capacity = np.copy(eval_env.capacity)
            agent_env.bw_allocated = np.copy(eval_env.bw_allocated)
            agent_env.node_capacity = np.copy(eval_env.node_capacity)
            agent_env.node_load = np.copy(eval_env.node_load)
            agent_env.load_factors = np.copy(eval_env.load_factors)
            
            # Save environment and initial state
            envs[agent_name] = agent_env
            
            # Copy state
            if isinstance(state, dict):
                states[agent_name] = {
                    'link_state': np.copy(state['link_state']),
                    'node_state': np.copy(state['node_state'])
                }
            else:
                states[agent_name] = np.copy(state)
        
        # Evaluate each agent
        agent_data = {
            'grpo': {'agent': grpo_agent, 'reward': 0, 'steps': 0, 'done': False, 
                    'demand': demand, 'source': source, 'destination': destination,
                    'rewards': [], 'success_count': 0},
            'sap': {'agent': sap_agent, 'reward': 0, 'steps': 0, 'done': False, 
                  'demand': demand, 'source': source, 'destination': destination,
                  'rewards': [], 'success_count': 0},
            'rand': {'agent': rand_agent, 'reward': 0, 'steps': 0, 'done': False, 
                   'demand': demand, 'source': source, 'destination': destination,
                   'rewards': [], 'success_count': 0}
        }
        
        # Maximum step limit
        max_steps = 50
        
        # Multiple evaluation rounds
        while any(not data['done'] and data['steps'] < max_steps for data in agent_data.values()):
            for agent_name, data in agent_data.items():
                if data['done'] or data['steps'] >= max_steps:
                    continue
                
                # Get agent action
                if agent_name == 'grpo':
                    # GRPO agent returns joint action and probabilities
                    joint_action, _ = data['agent'].act(
                        states[agent_name], 
                        data['demand'], 
                        data['source'], 
                        data['destination'], 
                        deterministic=True
                    )
                else:
                    # SAP and RAND agents directly return joint action
                    joint_action = data['agent'].act(
                        envs[agent_name], 
                        states[agent_name], 
                        data['demand'], 
                        data['source'], 
                        data['destination']
                    )
                
                # Execute action
                next_state, reward, done, next_demand, next_source, next_destination = \
                    envs[agent_name].make_step(
                        states[agent_name], 
                        joint_action, 
                        data['demand'], 
                        data['source'], 
                        data['destination']
                    )
                
                # Update agent data
                data['reward'] += reward
                data['rewards'].append(reward)
                data['steps'] += 1
                data['done'] = done
                
                # Record successful allocation
                if reward > 0:
                    data['success_count'] += 1
                
                # Update state and demand
                states[agent_name] = next_state
                data['demand'] = next_demand
                data['source'] = next_source
                data['destination'] = next_destination
        
        # Collect total rewards for each agent
        grpo_rewards.append(agent_data['grpo']['reward'])
        sap_rewards.append(agent_data['sap']['reward'])
        rand_rewards.append(agent_data['rand']['reward'])
        
        # Calculate success rates
        if agent_data['grpo']['steps'] > 0:
            grpo_success += agent_data['grpo']['success_count'] / agent_data['grpo']['steps']
        if agent_data['sap']['steps'] > 0:
            sap_success += agent_data['sap']['success_count'] / agent_data['sap']['steps']
        if agent_data['rand']['steps'] > 0:
            rand_success += agent_data['rand']['success_count'] / agent_data['rand']['steps']
    
    # Calculate average data
    avg_grpo_reward = np.mean(grpo_rewards) if grpo_rewards else 0
    avg_sap_reward = np.mean(sap_rewards) if sap_rewards else 0
    avg_rand_reward = np.mean(rand_rewards) if rand_rewards else 0
    
    avg_grpo_success = grpo_success / num_episodes if num_episodes > 0 else 0
    avg_sap_success = sap_success / num_episodes if num_episodes > 0 else 0
    avg_rand_success = rand_success / num_episodes if num_episodes > 0 else 0
    
    # Calculate standard deviation
    std_grpo_reward = np.std(grpo_rewards) if len(grpo_rewards) > 1 else 0
    std_sap_reward = np.std(sap_rewards) if len(sap_rewards) > 1 else 0
    std_rand_reward = np.std(rand_rewards) if len(rand_rewards) > 1 else 0
    
    # Output results
    if print_log:
        print(f"\n{eval_mode.capitalize()} environment evaluation results:")
        print(f"GRPO agent: Avg reward {avg_grpo_reward:.4f} ± {std_grpo_reward:.4f}, Success rate {avg_grpo_success:.2f}")
        print(f"SAP agent:  Avg reward {avg_sap_reward:.4f} ± {std_sap_reward:.4f}, Success rate {avg_sap_success:.2f}")
        print(f"RAND agent: Avg reward {avg_rand_reward:.4f} ± {std_rand_reward:.4f}, Success rate {avg_rand_success:.2f}")
        print(f"GRPO vs SAP advantage: {avg_grpo_reward - avg_sap_reward:.4f}")
    
    # Return results
    return {
        'grpo_reward': avg_grpo_reward,
        'sap_reward': avg_sap_reward,
        'rand_reward': avg_rand_reward,
        'grpo_success': avg_grpo_success,
        'sap_success': avg_sap_success,
        'rand_success': avg_rand_success,
        'grpo_std': std_grpo_reward,
        'sap_std': std_sap_reward,
        'rand_std': std_rand_reward
    }


def compare_agents_across_topologies(grpo_agent, env, num_episodes=20):
    """
    Compare agent performance across different network topologies.
    
    Args:
        grpo_agent: GRPO agent
        env: Network environment
        num_episodes: Number of evaluation episodes per topology
        
    Returns:
        dict: Comparison results across topologies
    """
    print("Starting agent comparison across different topologies...")
    
    # Create baseline agents
    from src.agents.sap_agent import SAPAgent
    from src.agents.rand_agent import RANDAgent
    sap_agent = SAPAgent()
    rand_agent = RANDAgent()
    
    # Define different topologies and configurations
    from src.environment import NetworkEnvironment, TOPOLOGY_SMALL, TOPOLOGY_NSFNET, TOPOLOGY_GEANT2
    
    topology_configs = [
        ("Standard", env.topology_type, env.listofDemands),
        ("Simple", TOPOLOGY_SMALL, [8, 16, 32]),      # Small topology, low bandwidth demands
        ("NSFNET", TOPOLOGY_NSFNET, [8, 32, 64]),     # NSFNET topology, medium bandwidth demands
        ("GEANT2", TOPOLOGY_GEANT2, [16, 64, 128])    # GEANT2 topology, high bandwidth demands
    ]
    
    comparison_results = {}
    
    # Evaluate on each topology
    for topo_name, topo_type, demand_values in topology_configs:
        try:
            # Create environment for this topology
            topo_env = NetworkEnvironment(topology_type=topo_type, demand_values=demand_values)
            print(f"\nEvaluating {topo_name} topology (Nodes:{topo_env.numNodes}, Edges:{topo_env.numEdges}, Demands:{demand_values})")
            
            # Evaluate agents
            standard_results = evaluate_agents(
                grpo_agent, sap_agent, rand_agent, topo_env, 
                num_episodes=num_episodes, eval_mode='standard'
            )
            
            complex_results = evaluate_agents(
                grpo_agent, sap_agent, rand_agent, topo_env, 
                num_episodes=num_episodes//2, eval_mode='complex'
            )
            
            # Save results
            comparison_results[topo_name] = {
                'standard': standard_results,
                'complex': complex_results
            }
            
            # Output comprehensive performance
            print(f"\n{topo_name} topology comprehensive performance:")
            for scenario in ['standard', 'complex']:
                results = comparison_results[topo_name][scenario]
                print(f"  {scenario.capitalize()} scenario:")
                print(f"    GRPO: {results['grpo_reward']:.4f}, Success rate: {results['grpo_success']:.2f}")
                print(f"    SAP:  {results['sap_reward']:.4f}, Success rate: {results['sap_success']:.2f}")
                print(f"    RAND: {results['rand_reward']:.4f}, Success rate: {results['rand_success']:.2f}")
                print(f"    GRPO vs SAP advantage: {results['grpo_reward'] - results['sap_reward']:.4f}")
            
        except Exception as e:
            print(f"Error during {topo_name} topology evaluation: {e}")
    
    # Return all results
    return comparison_results


def evaluate_both_agents(agent, sap_agent, env, num_episodes=10, eval_mode='standard', print_log=True):
    """
    Simultaneously evaluate GRPO and SAP agents.
    
    Args:
        agent: GRPO agent
        sap_agent: SAP agent
        env: Network environment
        num_episodes: Number of episodes to evaluate
        eval_mode: Evaluation mode
        print_log: Whether to print logs
        
    Returns:
        tuple: (avg_grpo_reward, avg_sap_reward)
    """
    # Configure evaluation environment based on mode
    if eval_mode == 'complex':
        # Complex scenario evaluation
        from src.environment import NetworkEnvironment, TOPOLOGY_GEANT2
        
        eval_env = NetworkEnvironment(topology_type=TOPOLOGY_GEANT2, demand_values=[16, 64, 128])
        
        # Increase node load factor variation
        eval_env.load_factors = np.random.uniform(1.5, 2.5, eval_env.numNodes)
        
        # Pre-load network to create more complex decision environment
        state, _, _, _ = eval_env.reset()
        for _ in range(10):  # Randomly allocate 10 traffic requests
            src = np.random.choice(eval_env.nodes)
            dst = np.random.choice([n for n in eval_env.nodes if n != src])
            demand_val = np.random.choice(eval_env.listofDemands)
            
            path_key = f"{src}:{dst}"
            if path_key in eval_env.allPaths and len(eval_env.allPaths[path_key]) > 0:
                path = eval_env.allPaths[path_key][0]
                
                # Randomly select a task execution node
                valid_nodes = list(range(1, len(path) - 1))  # Exclude first and last nodes
                if valid_nodes:
                    task_node_idx = np.random.choice(valid_nodes)
                    task_node = path[task_node_idx]
                    task_load = demand_val * eval_env.load_factors[task_node]
                    
                    # Decrease node available capacity
                    if eval_env.node_capacity[task_node] >= task_load:
                        eval_env.node_capacity[task_node] -= task_load
                        eval_env.node_load[task_node] += task_load
                
                # Allocate link bandwidth
                can_allocate = True
                for i in range(len(path) - 1):
                    edge_key = f"{path[i]}:{path[i+1]}"
                    edge_idx = eval_env.edgesDict[edge_key]
                    if eval_env.capacity[edge_idx] < demand_val:
                        can_allocate = False
                        break
                
                if can_allocate:
                    for i in range(len(path) - 1):
                        edge_key = f"{path[i]}:{path[i+1]}"
                        edge_idx = eval_env.edgesDict[edge_key]
                        eval_env.capacity[edge_idx] -= demand_val
                        eval_env.bw_allocated[edge_idx] = demand_val
                        
    elif eval_mode == 'simple':
        # Simple scenario evaluation
        from src.environment import NetworkEnvironment, TOPOLOGY_SMALL
        
        eval_env = NetworkEnvironment(topology_type=TOPOLOGY_SMALL, demand_values=[8, 16, 32])
        # Use lower load factors
        eval_env.load_factors = np.random.uniform(1.0, 1.5, eval_env.numNodes)
    else:  # standard
        # Standard evaluation
        eval_env = env
    
    # Evaluate two agents
    grpo_rewards = []
    sap_rewards = []
    
    for ep in range(num_episodes):
        # Ensure two agents use exactly the same starting state
        state, demand, source, destination = eval_env.reset()
        
        # Copy initial state
        if isinstance(state, dict):
            initial_state = {
                'link_state': np.copy(state['link_state']),
                'node_state': np.copy(state['node_state'])
            }
        else:
            initial_state = np.copy(state)
        
        # Create independent environment for SAP
        from src.environment import NetworkEnvironment
        sap_env = NetworkEnvironment(topology_type=eval_env.topology_type, demand_values=eval_env.listofDemands)
        sap_env.capacity = np.copy(eval_env.capacity)
        sap_env.bw_allocated = np.copy(eval_env.bw_allocated)
        sap_env.node_capacity = np.copy(eval_env.node_capacity)
        sap_env.node_load = np.copy(eval_env.node_load)
        sap_env.load_factors = np.copy(eval_env.load_factors)
        
        # GRPO evaluation
        episode_reward_grpo = 0
        grpo_done = False
        grpo_steps = 0
        max_steps = 100
        
        if isinstance(state, dict):
            grpo_state = {
                'link_state': np.copy(state['link_state']),
                'node_state': np.copy(state['node_state'])
            }
        else:
            grpo_state = np.copy(state)
            
        grpo_demand = demand
        grpo_source = source
        grpo_destination = destination
        
        while not grpo_done and grpo_steps < max_steps:
            # Get joint action: (route selection, task execution node)
            joint_action, _ = agent.act(grpo_state, grpo_demand, grpo_source, grpo_destination, deterministic=True)
            next_state, reward, done, next_demand, next_source, next_destination = \
                eval_env.make_step(grpo_state, joint_action, grpo_demand, grpo_source, grpo_destination)
            
            episode_reward_grpo += reward
            grpo_steps += 1
            
            grpo_state = next_state
            grpo_demand = next_demand
            grpo_source = next_source
            grpo_destination = next_destination
            grpo_done = done
        
        grpo_rewards.append(episode_reward_grpo)
        
        # SAP evaluation - using same initial conditions
        episode_reward_sap = 0
        sap_done = False
        sap_steps = 0
        
        sap_state = initial_state
        sap_demand = demand
        sap_source = source
        sap_destination = destination
        
        while not sap_done and sap_steps < max_steps:
            # Get SAP's joint action: (route selection, task execution node)
            joint_action = sap_agent.act(sap_env, sap_state, sap_demand, sap_source, sap_destination)
            next_state, reward, done, next_demand, next_source, next_destination = \
                sap_env.make_step(sap_state, joint_action, sap_demand, sap_source, sap_destination)
            
            episode_reward_sap += reward
            sap_steps += 1
            
            sap_state = next_state
            sap_demand = next_demand
            sap_source = next_source
            sap_destination = next_destination
            sap_done = done
        
        sap_rewards.append(episode_reward_sap)
    
    # Calculate average rewards and standard deviations
    avg_grpo = np.mean(grpo_rewards)
    avg_sap = np.mean(sap_rewards)
    std_grpo = np.std(grpo_rewards)
    std_sap = np.std(sap_rewards)
    
    # Output comparison results
    if print_log:
        print(f"{eval_mode.capitalize()} environment evaluation:")
        print(f"  GRPO: {avg_grpo:.4f} ± {std_grpo:.4f}")
        print(f"  SAP: {avg_sap:.4f} ± {std_sap:.4f}")
        print(f"  GRPO vs SAP: {avg_grpo - avg_sap:.4f}")
    
    return avg_grpo, avg_sap