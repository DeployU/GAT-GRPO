import numpy as np

class RANDAgent:
    """Random path selection agent - joint optimization version."""
    
    def __init__(self):
        """Initialize RAND agent."""
        self.K = 4  # Number of shortest paths to consider
    
    def act(self, env, state, demand, source, destination):
        """
        Randomly select an available path and task execution node.
        
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
        
        # First check which paths are available (sufficient link capacity)
        available_routes = []
        available_paths = []
        
        for path_idx, path in enumerate(path_list[:min(self.K, len(path_list))]):
            can_allocate = True
            for i in range(len(path) - 1):
                edge_key = f"{path[i]}:{path[i+1]}"
                if edge_key not in env.edgesDict:
                    can_allocate = False
                    break
                edge_idx = env.edgesDict[edge_key]
                
                link_state = state['link_state'] if isinstance(state, dict) else state
                if edge_idx >= link_state.shape[0] or link_state[edge_idx, 0] < demand:
                    can_allocate = False
                    break
            
            if can_allocate:
                available_routes.append(path_idx)
                available_paths.append(path)
        
        # If there are available paths, randomly select one; otherwise randomly select any path (even if unavailable)
        if available_routes:
            random_idx = np.random.randint(0, len(available_routes))
            selected_route = available_routes[random_idx]
            selected_path = available_paths[random_idx]
        else:
            selected_route = np.random.randint(0, min(self.K, len(path_list)))
            selected_path = path_list[selected_route] if selected_route < len(path_list) else path_list[0]
        
        # Randomly select a task execution node (excluding source and destination nodes)
        valid_nodes = list(range(1, len(selected_path) - 1))  # Exclude first and last nodes
        
        if not valid_nodes:  # If no intermediate nodes, select source node
            selected_node = 0
        else:
            # Completely random selection of intermediate node, regardless of node capacity constraint
            selected_node = np.random.choice(valid_nodes)
        
        return (selected_route, selected_node)