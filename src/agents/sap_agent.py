import numpy as np

class SAPAgent:
    """Shortest Available Path agent - joint optimization version."""
    
    def __init__(self):
        """Initialize SAP agent."""
        self.K = 4  # Number of shortest paths to consider
    
    def act(self, env, state, demand, source, destination):
        """
        Select the shortest available path and a random task execution node.
        
        Args:
            env: Network environment
            state: Current network state
            demand: Bandwidth demand
            source: Source node
            destination: Destination node
            
        Returns:
            tuple: Joint action (route_action, node_idx)
        """
        path_key = f"{source}:{destination}"
        if path_key not in env.allPaths:
            return (0, 0)  # If no path found, return default joint action
        
        path_list = env.allPaths[path_key]
        if not path_list:
            return (0, 0)  # If path list is empty, return default action
        
        # Iterate through paths, find the first path that can satisfy link demand
        selected_route = 0
        selected_path = None
        
        for path_idx, path in enumerate(path_list[:min(self.K, len(path_list))]):
            # Check if path is available (sufficient link capacity)
            can_allocate_path = True
            for i in range(len(path) - 1):
                edge_key = f"{path[i]}:{path[i+1]}"
                if edge_key not in env.edgesDict:
                    can_allocate_path = False
                    break
                edge_idx = env.edgesDict[edge_key]
                
                link_state = state['link_state'] if isinstance(state, dict) else state
                if edge_idx >= link_state.shape[0] or link_state[edge_idx, 0] < demand:
                    can_allocate_path = False
                    break
            
            if can_allocate_path:
                selected_route = path_idx
                selected_path = path
                break
        
        # If no path satisfying link demand is found, use the first path
        if selected_path is None:
            selected_path = path_list[0]
            selected_route = 0
        
        # Randomly select a task execution node (excluding source and destination nodes)
        valid_nodes = list(range(1, len(selected_path) - 1))  # Exclude first and last nodes
        
        # If no intermediate nodes, select source node
        if not valid_nodes:
            selected_node = 0
        else:
            # Check each node to see if it has sufficient capacity to execute task
            valid_task_nodes = []
            
            for node_idx in valid_nodes:
                node = selected_path[node_idx]
                if node >= len(env.load_factors):
                    continue  # Skip invalid node index
                
                task_load = demand * env.load_factors[node]
                
                if node < len(env.node_capacity) and env.node_capacity[node] >= task_load:
                    valid_task_nodes.append(node_idx)
            
            # If there are available task nodes, randomly select one; otherwise select the first intermediate node
            if valid_task_nodes:
                selected_node = np.random.choice(valid_task_nodes)
            else:
                selected_node = valid_nodes[0] if valid_nodes else 0
        
        return (selected_route, selected_node)