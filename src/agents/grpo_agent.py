import tensorflow as tf
import numpy as np
import os

from src.models.gat import GAT

class GRPOAgent:
    """Graph-structured Policy Optimization agent for joint optimization."""
    
    def __init__(self, env, hparams):
        """
        Initialize GRPO agent.
        
        Args:
            env: Network environment
            hparams: Hyperparameters dictionary
        """
        self.env = env
        self.hparams = hparams
        
        # Create policy network and reference network
        self.policy_net = GAT(hparams)
        self.reference_net = GAT(hparams)
        
        # Explicitly build network architecture
        self.policy_net.build()
        self.reference_net.build()
        
        # Initialize networks with a forward pass
        dummy_state = self.env._get_state()
        dummy_input = self.env.get_graph_features(dummy_state)
        
        # Create correct graph IDs
        dummy_graph_ids = tf.zeros([tf.shape(dummy_input['link_state'])[0]], dtype=tf.int32)
        
        # Forward pass to initialize
        self.policy_net(
            dummy_input['link_state'],
            dummy_input['node_state'],
            dummy_graph_ids,
            dummy_input['first'],
            dummy_input['second'],
            num_edges=dummy_input['num_edges'],
            num_nodes=dummy_input['num_nodes'],
            training=False
        )
        
        # Ensure reference network has the same weights as policy network
        self.reference_net.set_weights(self.policy_net.get_weights())
        
        # Set optimizer
        self.current_lr = hparams['learning_rate']
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.current_lr)
        
        # Training metrics
        self.step_counter = 0
        self.episode_counter = 0
        self.best_reward = -float('inf')
    
    def act(self, state, demand, source, destination, deterministic=False):
        """
        Select a joint action (routing + task execution node) according to current policy.
        
        Args:
            state: Current network state
            demand: Bandwidth demand
            source: Source node
            destination: Destination node
            deterministic: Whether to select action deterministically
            
        Returns:
            action: Joint action (route_action, node_idx)
            action_probs: Probabilities of the selected actions
        """
        # Get graph features
        graph_features = self.env.get_graph_features(state)
        
        # Create correct graph IDs - ensure length matches link state
        graph_ids = tf.zeros([tf.shape(graph_features['link_state'])[0]], dtype=tf.int32)
        
        # Forward pass on policy network
        route_probs, node_probs = self.policy_net(
            graph_features['link_state'],
            graph_features['node_state'],
            graph_ids,
            graph_features['first'],
            graph_features['second'],
            num_edges=graph_features['num_edges'],
            num_nodes=graph_features['num_nodes'],
            training=False
        )
        
        route_probs = route_probs.numpy()[0]
        node_probs = node_probs.numpy()[0]
        
        # Ensure path key exists
        path_key = f"{source}:{destination}"
        if path_key not in self.env.allPaths or len(self.env.allPaths[path_key]) == 0:
            return (0, 0), (route_probs[0] if len(route_probs) > 0 else 1.0, 1.0)
        
        # Route selection
        num_paths = min(4, len(self.env.allPaths[path_key]))
        
        if len(route_probs) < num_paths:
            num_paths = len(route_probs)
        
        if num_paths == 0:
            return (0, 0), (1.0, 1.0)  # If no paths available, return default action
        
        if deterministic:
            # Evaluation mode - select route with highest probability
            route_action = np.argmax(route_probs[:num_paths])
        else:
            # Training mode - sample route from probability distribution
            valid_route_probs = route_probs[:num_paths]
            valid_route_probs = valid_route_probs / np.sum(valid_route_probs)  # Re-normalize
            route_action = np.random.choice(num_paths, p=valid_route_probs)
        
        # Ensure route action is valid
        if route_action >= num_paths:
            route_action = 0  # If action is invalid, use default action
        
        # Get selected path
        selected_path = self.env.allPaths[path_key][route_action]
        
        # Node selection - select a node along the path to execute task
        # Exclude source and destination nodes
        valid_nodes = list(range(1, len(selected_path) - 1))  # Exclude first and last nodes
        
        if not valid_nodes:  # If no intermediate nodes (direct connection between two nodes)
            node_idx = 0  # Choose source node
        else:
            max_nodes = min(len(valid_nodes), 10)  # Consider at most 10 nodes
            
            if deterministic:
                # Evaluation mode - select node with highest probability
                node_action = np.argmax(node_probs[:max_nodes])
                node_idx = valid_nodes[node_action % len(valid_nodes)]
            else:
                # Training mode - sample node from probability distribution
                valid_node_probs = node_probs[:max_nodes]
                valid_node_probs = valid_node_probs / np.sum(valid_node_probs)  # Re-normalize
                node_action = np.random.choice(max_nodes, p=valid_node_probs)
                node_idx = valid_nodes[node_action % len(valid_nodes)]
        
        # Return joint action and corresponding probabilities
        return (route_action, node_idx), (route_probs[route_action], node_probs[node_idx % len(node_probs)])
    
    def collect_trajectory(self, group_size=16):
        """
        Collect a group of trajectory samples for GRPO learning.
        
        Args:
            group_size: Number of samples to collect
            
        Returns:
            dict: Group of trajectories including states, actions, rewards, etc.
        """
        group_states = []
        group_actions = []
        group_route_probs = []
        group_node_probs = []
        group_rewards = []
        group_dones = []
        group_next_states = []
        group_demands = []
        group_sources = []
        group_destinations = []
        
        # Reset environment
        state, demand, source, destination = self.env.reset()
        
        # Collect group_size samples
        samples_collected = 0
        
        while samples_collected < group_size:
            # Copy current state for independent exploration
            state_copy = {
                'link_state': np.copy(state['link_state']),
                'node_state': np.copy(state['node_state'])
            }
            demand_copy = demand
            source_copy = source
            destination_copy = destination
            
            # Select joint action
            action, action_probs = self.act(state_copy, demand_copy, source_copy, destination_copy)
            route_action, node_idx = action
            route_prob, node_prob = action_probs
            
            # Execute action
            next_state, reward, done, next_demand, next_source, next_destination = \
                self.env.make_step(state_copy, action, demand_copy, source_copy, destination_copy)
            
            # Reward scaling - amplify positive rewards to enhance learning signal
            if reward > 0:
                reward *= self.hparams['reward_scale']
            
            # Store sample
            group_states.append(state_copy)
            group_actions.append(action)
            group_route_probs.append(route_prob)
            group_node_probs.append(node_prob)
            group_rewards.append(reward)
            group_dones.append(done)
            group_next_states.append(next_state)
            group_demands.append(demand_copy)
            group_sources.append(source_copy)
            group_destinations.append(destination_copy)
            
            samples_collected += 1
            
            # Update state
            state = next_state
            demand = next_demand
            source = next_source
            destination = next_destination
            
            self.step_counter += 1
            
            if done:
                # If an episode is done, reset the environment
                state, demand, source, destination = self.env.reset()
                self.episode_counter += 1
        
        # Return the entire trajectory group
        return {
            'states': group_states,
            'actions': group_actions,
            'route_probs': group_route_probs,
            'node_probs': group_node_probs,
            'rewards': group_rewards,
            'dones': group_dones,
            'next_states': group_next_states,
            'demands': group_demands,
            'sources': group_sources,
            'destinations': group_destinations
        }
    
    def update_reference_model(self):
        """Update reference model weights."""
        self.reference_net.set_weights(self.policy_net.get_weights())

    def save_model(self, path):
        """
        Save model weights.
        
        Args:
            path: Path to save model weights
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save weights
        if not path.endswith('.weights.h5'):
            path = f"{path}.weights.h5"
        self.policy_net.save_weights(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load model weights.
        
        Args:
            path: Path to load model weights from
        """
        # Ensure path format is correct
        if not path.endswith('.weights.h5'):
            path = f"{path}.weights.h5"
        
        # Load weights
        self.policy_net.load_weights(path)
        self.reference_net.set_weights(self.policy_net.get_weights())
        print(f"Model loaded from {path}")
        
    def train_step(self, trajectory_group):
        """
        Execute one step of GRPO algorithm training - joint optimization version.
        
        Args:
            trajectory_group: Group of trajectories
            
        Returns:
            dict: Training metrics
        """
        states = trajectory_group['states']
        actions = trajectory_group['actions']
        old_route_probs = trajectory_group['route_probs']
        old_node_probs = trajectory_group['node_probs']
        rewards = trajectory_group['rewards']
        
        if len(states) == 0:
            print("Warning: Empty trajectory group, skipping training")
            return {
                'policy_loss': 0.0,
                'kl_div': 0.0,
                'total_loss': 0.0,
                'grad_norm': 0.0,
                'mean_reward': 0.0,
                'reward_std': 0.0,
                'entropy': 0.0,
                'learning_rate': self.current_lr
            }
        
        # Calculate relative rewards (advantages)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        mean_reward = tf.reduce_mean(rewards_tensor)
        std_reward = tf.math.reduce_std(rewards_tensor) + 1e-6  # Prevent division by zero
        advantages = (rewards_tensor - mean_reward) / std_reward
        
        # Prepare batch graph features
        batch_link_states = []
        batch_node_states = []
        batch_first_all = []
        batch_second_all = []
        batch_graph_ids = []
        
        # Use safe node offset handling
        max_node_id = 0
        
        for i, state in enumerate(states):
            # Get graph features
            features = self.env.get_graph_features(state)
            
            # Add to batch
            batch_link_states.append(features['link_state'])
            batch_node_states.append(features['node_state'])
            
            # Apply correct offset to edges for each graph
            first = features['first'] + max_node_id
            second = features['second'] + max_node_id
            batch_first_all.append(first)
            batch_second_all.append(second)
            
            # Assign graph ID to all edges of current graph
            edge_count = tf.shape(features['link_state'])[0]
            batch_graph_ids.append(tf.fill([edge_count], i))
            
            # Update max node ID for next graph
            max_node_id += features['num_nodes']
        
        # Merge into tensors
        batch_link_states = tf.concat(batch_link_states, axis=0)
        batch_node_states = tf.concat(batch_node_states, axis=0)
        batch_first = tf.concat(batch_first_all, axis=0)
        batch_second = tf.concat(batch_second_all, axis=0)
        batch_graph_ids = tf.concat(batch_graph_ids, axis=0)
        
        # Extract actions
        route_actions = [a[0] for a in actions]
        node_idxs = [a[1] for a in actions]
        
        # Use GradientTape to record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Calculate action probabilities under current policy
            new_route_probs_dist, new_node_probs_dist = self.policy_net(
                batch_link_states,
                batch_node_states,
                batch_graph_ids,
                batch_first,
                batch_second,
                num_edges=max([features['num_edges'] for features in [self.env.get_graph_features(s) for s in states]]),
                num_nodes=max([features['num_nodes'] for features in [self.env.get_graph_features(s) for s in states]]),
                training=True
            )
            
            # Extract current policy's route action probabilities
            if tf.shape(new_route_probs_dist)[0] < len(route_actions):
                # Truncate actions and advantages to match
                route_actions = route_actions[:tf.shape(new_route_probs_dist)[0]]
                node_idxs = node_idxs[:tf.shape(new_route_probs_dist)[0]]
                advantages = advantages[:tf.shape(new_route_probs_dist)[0]]
                old_route_probs = old_route_probs[:tf.shape(new_route_probs_dist)[0]]
                old_node_probs = old_node_probs[:tf.shape(new_route_probs_dist)[0]]
            
            # Extract probability for each state's corresponding route action
            route_indices = [[i, route_actions[i]] for i in range(len(route_actions))]
            selected_route_probs = tf.gather_nd(new_route_probs_dist, route_indices)
            
            # Extract probability for each state's corresponding node action
            node_indices = [[i, min(node_idxs[i], tf.shape(new_node_probs_dist)[1]-1)] for i in range(len(node_idxs))]
            selected_node_probs = tf.gather_nd(new_node_probs_dist, node_indices)
            
            # Calculate joint action probability (route probability * node probability)
            selected_probs = selected_route_probs * selected_node_probs
            old_action_probs_tensor = tf.convert_to_tensor(
                [old_route_probs[i] * old_node_probs[i] for i in range(len(old_route_probs))], 
                dtype=tf.float32
            )
            
            # Calculate policy ratio
            ratio = selected_probs / (old_action_probs_tensor + 1e-8)  # Prevent division by zero
            
            # Calculate clipped objective
            clip_ratio = self.hparams['clip_ratio']
            clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
            surrogate1 = ratio * advantages
            surrogate2 = clipped_ratio * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            
            # Calculate KL divergence constraint
            ref_route_probs_dist, ref_node_probs_dist = self.reference_net(
                batch_link_states,
                batch_node_states,
                batch_graph_ids,
                batch_first,
                batch_second,
                num_edges=max([features['num_edges'] for features in [self.env.get_graph_features(s) for s in states]]),
                num_nodes=max([features['num_nodes'] for features in [self.env.get_graph_features(s) for s in states]]),
                training=False
            )
            
            # Safely calculate KL divergence
            epsilon = 1e-8
            route_kl_div = tf.reduce_mean(
                tf.reduce_sum(
                    ref_route_probs_dist * tf.math.log((ref_route_probs_dist + epsilon) / (new_route_probs_dist + epsilon)),
                    axis=1
                )
            )
            
            node_kl_div = tf.reduce_mean(
                tf.reduce_sum(
                    ref_node_probs_dist * tf.math.log((ref_node_probs_dist + epsilon) / (new_node_probs_dist + epsilon)),
                    axis=1
                )
            )
            
            # Combined KL divergence (route and node selection)
            kl_div = 0.6 * route_kl_div + 0.4 * node_kl_div
            
            # Calculate entropy regularization term
            route_entropy = -tf.reduce_mean(
                tf.reduce_sum(
                    new_route_probs_dist * tf.math.log(new_route_probs_dist + epsilon),
                    axis=1
                )
            )
            
            node_entropy = -tf.reduce_mean(
                tf.reduce_sum(
                    new_node_probs_dist * tf.math.log(new_node_probs_dist + epsilon),
                    axis=1
                )
            )
            
            # Combined entropy (route and node selection)
            entropy = 0.6 * route_entropy + 0.4 * node_entropy
            
            # Ensure KL divergence has minimum value
            min_kl = self.hparams['min_kl']
            kl_penalty = tf.maximum(0.0, (min_kl - kl_div) * 10.0)
            
            # Calculate total loss
            scale_factor = self.hparams['scale_factor']
            entropy_coef = self.hparams['entropy_coef']
            
            policy_loss_scaled = scale_factor * policy_loss
            entropy_loss = -entropy_coef * entropy
            kl_loss = self.hparams['kl_coef'] * tf.maximum(kl_div, min_kl)
            
            total_loss = policy_loss_scaled + scale_factor * (entropy_loss + kl_loss + kl_penalty)
        
        # Calculate gradients
        gradients = tape.gradient(total_loss, self.policy_net.trainable_variables)
        
        # Check if gradients are valid
        valid_gradients = True
        for grad in gradients:
            if grad is None:
                valid_gradients = False
                break
            
            if tf.reduce_any(tf.math.is_nan(grad)) or tf.reduce_any(tf.math.is_inf(grad)):
                valid_gradients = False
                break
        
        if valid_gradients:
            # Gradient clipping
            gradients, grad_norm = tf.clip_by_global_norm(gradients, self.hparams['max_grad_norm'])
            
            # Apply gradients
            self.optimizer.apply_gradients(zip(gradients, self.policy_net.trainable_variables))
            
            # Update current learning rate
            if hasattr(self.optimizer, 'learning_rate'):
                lr = self.optimizer.learning_rate
                if hasattr(lr, 'numpy'):
                    self.current_lr = lr.numpy()
                else:
                    self.current_lr = float(lr)
            
            # Return training metrics
            return {
                'policy_loss': policy_loss.numpy(),
                'kl_div': kl_div.numpy(),
                'total_loss': total_loss.numpy(),
                'grad_norm': grad_norm.numpy(),
                'mean_reward': mean_reward.numpy(),
                'reward_std': std_reward.numpy(),
                'entropy': entropy.numpy(),
                'learning_rate': self.current_lr
            }
        else:
            print("Warning: Invalid gradients detected, skipping update")
            
            # Return empty metrics when gradients are invalid
            return {
                'policy_loss': 0.0,
                'kl_div': 0.0,
                'total_loss': 0.0,
                'grad_norm': 0.0,
                'mean_reward': np.mean(rewards) if len(rewards) > 0 else 0.0,
                'reward_std': np.std(rewards) if len(rewards) > 0 else 0.0,
                'entropy': 0.0,
                'learning_rate': self.current_lr
            }