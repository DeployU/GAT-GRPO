import tensorflow as tf
import numpy as np

class GAT(tf.keras.Model):
    """Graph Attention Network for joint optimization of routing and task execution."""
    
    def __init__(self, hparams):
        """
        Initialize the GAT model.
        
        Args:
            hparams: Dict of hyperparameters
        """
        super(GAT, self).__init__()
        self.hparams = hparams
        
        # Calculate dimensions per attention head
        self.link_head_dim = self.hparams['link_state_dim'] // self.hparams['attention_heads']
        self.node_head_dim = self.hparams['node_state_dim'] // self.hparams['attention_heads']
        
        # Link feature transformation layer
        self.link_feature_transform = tf.keras.layers.Dense(
            self.hparams['link_state_dim'],
            activation='elu',
            kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
            name="LinkFeatureTransform"
        )
        
        # Node feature transformation layer
        self.node_feature_transform = tf.keras.layers.Dense(
            self.hparams['node_state_dim'],
            activation='elu',
            kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
            name="NodeFeatureTransform"
        )
        
        # Link multi-head attention
        self.link_attention_q = []
        self.link_attention_k = []
        self.link_attention_v = []
        
        for i in range(self.hparams['attention_heads']):
            self.link_attention_q.append(tf.keras.layers.Dense(
                self.link_head_dim, activation=None, use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
                name=f"LinkAttentionQ_{i}"
            ))
            
            self.link_attention_k.append(tf.keras.layers.Dense(
                self.link_head_dim, activation=None, use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
                name=f"LinkAttentionK_{i}"
            ))
            
            self.link_attention_v.append(tf.keras.layers.Dense(
                self.link_head_dim, activation=None, use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
                name=f"LinkAttentionV_{i}"
            ))
        
        # Node multi-head attention
        self.node_attention_q = []
        self.node_attention_k = []
        self.node_attention_v = []
        
        for i in range(self.hparams['attention_heads']):
            self.node_attention_q.append(tf.keras.layers.Dense(
                self.node_head_dim, activation=None, use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
                name=f"NodeAttentionQ_{i}"
            ))
            
            self.node_attention_k.append(tf.keras.layers.Dense(
                self.node_head_dim, activation=None, use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
                name=f"NodeAttentionK_{i}"
            ))
            
            self.node_attention_v.append(tf.keras.layers.Dense(
                self.node_head_dim, activation=None, use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
                name=f"NodeAttentionV_{i}"
            ))
        
        # Link output projection layer
        self.link_output_projection = tf.keras.layers.Dense(
            self.hparams['link_state_dim'],
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
            name="LinkOutputProjection"
        )
        
        # Node output projection layer
        self.node_output_projection = tf.keras.layers.Dense(
            self.hparams['node_state_dim'],
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
            name="NodeOutputProjection"
        )
        
        # Link feed-forward network
        self.link_ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.hparams['link_state_dim'] * 2,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
                name="LinkFFN1"
            ),
            tf.keras.layers.Dropout(self.hparams['dropout_rate']),
            tf.keras.layers.Dense(
                self.hparams['link_state_dim'],
                activation=None,
                kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
                name="LinkFFN2"
            )
        ])
        
        # Node feed-forward network
        self.node_ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.hparams['node_state_dim'] * 2,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
                name="NodeFFN1"
            ),
            tf.keras.layers.Dropout(self.hparams['dropout_rate']),
            tf.keras.layers.Dense(
                self.hparams['node_state_dim'],
                activation=None,
                kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
                name="NodeFFN2"
            )
        ])
        
        # Layer normalization
        self.link_layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="LinkLayerNorm1")
        self.link_layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="LinkLayerNorm2")
        self.node_layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="NodeLayerNorm1")
        self.node_layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="NodeLayerNorm2")
        
        # Link-to-node attention connection
        self.link_to_node_projection = tf.keras.layers.Dense(
            self.hparams['node_state_dim'],
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
            name="LinkToNodeProjection"
        )
        
        # Readout network - route selection
        self.route_readout_net = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.hparams['readout_units'],
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
                name="RouteReadoutLayer1"
            ),
            tf.keras.layers.Dropout(rate=self.hparams['dropout_rate']),
            tf.keras.layers.Dense(
                self.hparams['readout_units'] // 2,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
                name="RouteReadoutLayer2"
            ),
            tf.keras.layers.Dropout(rate=self.hparams['dropout_rate']),
            tf.keras.layers.Dense(
                4,  # Output probabilities for 4 paths
                activation='softmax',
                kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
                name="RouteReadoutLayer3"
            )
        ])
        
        # Readout network - node selection (along path)
        self.node_readout_net = tf.keras.Sequential([
            tf.keras.layers.Dense(
                self.hparams['readout_units'] // 2,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
                name="NodeReadoutLayer1"
            ),
            tf.keras.layers.Dropout(rate=self.hparams['dropout_rate']),
            tf.keras.layers.Dense(
                self.hparams['readout_units'] // 4,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
                name="NodeReadoutLayer2"
            ),
            tf.keras.layers.Dropout(rate=self.hparams['dropout_rate']),
            tf.keras.layers.Dense(
                10,  # Consider up to 10 nodes in a path
                activation='softmax',
                kernel_regularizer=tf.keras.regularizers.l2(self.hparams['l2_reg']),
                name="NodeReadoutLayer3"
            )
        ])
    
    def build(self, input_shape=None):
        """
        Explicitly build the model to ensure all submodules are initialized.
        
        Args:
            input_shape: Optional input shape (not used, but required by Keras build method)
        """
        # Create example inputs
        batch_size = 2
        num_nodes = 5
        edge_per_node = 2
        num_edges = num_nodes * edge_per_node
        
        # Link state example input
        dummy_link_states = tf.random.normal([num_edges, self.hparams['link_state_dim']])
        # Node state example input
        dummy_node_states = tf.random.normal([num_nodes, self.hparams['node_state_dim']])
        
        dummy_graph_ids = tf.zeros([num_edges], dtype=tf.int32)
        dummy_first = tf.random.uniform([num_edges*2], maxval=num_nodes, dtype=tf.int32)
        dummy_second = tf.random.uniform([num_edges*2], maxval=num_nodes, dtype=tf.int32)
        
        # Ensure source and target node indices are valid
        dummy_first = tf.math.mod(dummy_first, num_nodes)
        dummy_second = tf.math.mod(dummy_second, num_nodes)
        
        # Forward pass to initialize all layers
        _ = self.call(
            dummy_link_states, 
            dummy_node_states,
            dummy_graph_ids,
            dummy_first,
            dummy_second,
            num_edges=num_edges,
            num_nodes=num_nodes,
            training=False
        )
        
        self.built = True
        return
    
    def compute_link_message_passing(self, link_features, edge_index, training=False):
        """
        Perform graph attention message passing for link layer.
        
        Args:
            link_features: Link features
            edge_index: Edge indices
            training: Whether in training mode
            
        Returns:
            Graph attention output for links
        """
        # Extract source and target nodes
        source_indices = edge_index[0]
        target_indices = edge_index[1]
        
        # Ensure indices are in valid range
        num_links = tf.shape(link_features)[0]
        source_indices = tf.math.mod(source_indices, num_links)
        target_indices = tf.math.mod(target_indices, num_links)
        
        # Multi-head attention computation
        multi_head_output = []
        
        for head_idx in range(self.hparams['attention_heads']):
            # Transform source and target node features
            source_features = tf.gather(link_features, source_indices)
            target_features = tf.gather(link_features, target_indices)
            
            # Compute attention scores - using scaled dot-product attention
            query = self.link_attention_q[head_idx](source_features)
            key = self.link_attention_k[head_idx](target_features)
            value = self.link_attention_v[head_idx](target_features)
            
            # Calculate attention weights
            scale = tf.math.sqrt(tf.cast(self.link_head_dim, tf.float32))
            attention_logits = tf.reduce_sum(query * key, axis=-1) / scale
            
            # Apply softmax to each source node's neighbors
            attention_weights = tf.zeros_like(attention_logits)
            
            # Group by source nodes and apply softmax
            unique_sources, _ = tf.unique(source_indices)
            
            for src in unique_sources:
                # Find all edges for current source node
                mask = tf.equal(source_indices, src)
                
                # Compute softmax only for current source node's edges
                src_logits = tf.boolean_mask(attention_logits, mask)
                src_weights = tf.nn.softmax(src_logits)
                
                # Put results back to original positions
                indices = tf.where(mask)
                for i, idx in enumerate(tf.range(tf.shape(indices)[0])):
                    pos = indices[idx][0]
                    attention_weights = tf.tensor_scatter_nd_update(
                        attention_weights, 
                        [[pos]], 
                        [src_weights[i]]
                    )
            
            # Weight target node features
            weighted_values = value * tf.expand_dims(attention_weights, axis=-1)
            
            # Aggregate features by target node
            output = tf.zeros([num_links, self.link_head_dim], dtype=link_features.dtype)
            
            # Safely aggregate features
            for i in range(tf.shape(target_indices)[0]):
                target_idx = target_indices[i]
                output = tf.tensor_scatter_nd_add(
                    output,
                    [[target_idx]],
                    [weighted_values[i]]
                )
            
            multi_head_output.append(output)
        
        # Concatenate multi-head outputs
        if self.hparams['attention_heads'] > 1:
            concat_output = tf.concat(multi_head_output, axis=-1)
        else:
            concat_output = multi_head_output[0]
        
        # Project back to original dimension
        projected_output = self.link_output_projection(concat_output)
        
        return projected_output
    
    def compute_node_message_passing(self, node_features, edge_index, training=False):
        """
        Perform graph attention message passing for node layer.
        
        Args:
            node_features: Node features
            edge_index: Edge indices
            training: Whether in training mode
            
        Returns:
            Graph attention output for nodes
        """
        # Extract source and target nodes
        source_indices = edge_index[0]
        target_indices = edge_index[1]
        
        # Ensure indices are in valid range
        num_nodes = tf.shape(node_features)[0]
        source_indices = tf.math.mod(source_indices, num_nodes)
        target_indices = tf.math.mod(target_indices, num_nodes)
        
        # Multi-head attention computation
        multi_head_output = []
        
        for head_idx in range(self.hparams['attention_heads']):
            # Transform source and target node features
            source_features = tf.gather(node_features, source_indices)
            target_features = tf.gather(node_features, target_indices)
            
            # Compute attention scores - using scaled dot-product attention
            query = self.node_attention_q[head_idx](source_features)
            key = self.node_attention_k[head_idx](target_features)
            value = self.node_attention_v[head_idx](target_features)
            
            # Calculate attention weights
            scale = tf.math.sqrt(tf.cast(self.node_head_dim, tf.float32))
            attention_logits = tf.reduce_sum(query * key, axis=-1) / scale
            
            # Apply softmax to each source node's neighbors
            attention_weights = tf.zeros_like(attention_logits)
            
            # Group by source nodes and apply softmax
            unique_sources, _ = tf.unique(source_indices)
            
            for src in unique_sources:
                # Find all edges for current source node
                mask = tf.equal(source_indices, src)
                
                # Compute softmax only for current source node's edges
                src_logits = tf.boolean_mask(attention_logits, mask)
                src_weights = tf.nn.softmax(src_logits)
                
                # Put results back to original positions
                indices = tf.where(mask)
                for i, idx in enumerate(tf.range(tf.shape(indices)[0])):
                    pos = indices[idx][0]
                    attention_weights = tf.tensor_scatter_nd_update(
                        attention_weights, 
                        [[pos]], 
                        [src_weights[i]]
                    )
            
            # Weight target node features
            weighted_values = value * tf.expand_dims(attention_weights, axis=-1)
            
            # Aggregate features by target node
            output = tf.zeros([num_nodes, self.node_head_dim], dtype=node_features.dtype)
            
            # Safely aggregate features
            for i in range(tf.shape(target_indices)[0]):
                target_idx = target_indices[i]
                output = tf.tensor_scatter_nd_add(
                    output,
                    [[target_idx]],
                    [weighted_values[i]]
                )
            
            multi_head_output.append(output)
        
        # Concatenate multi-head outputs
        if self.hparams['attention_heads'] > 1:
            concat_output = tf.concat(multi_head_output, axis=-1)
        else:
            concat_output = multi_head_output[0]
        
        # Project back to original dimension
        projected_output = self.node_output_projection(concat_output)
        
        return projected_output
    
    def link_to_node_aggregation(self, link_features, edge_index, num_nodes):
        """
        Aggregate link features to node features.
        
        Args:
            link_features: Link features
            edge_index: Edge indices
            num_nodes: Number of nodes
            
        Returns:
            Node features aggregated from link features
        """
        # Extract source and target nodes
        source_indices = edge_index[0]
        target_indices = edge_index[1]
        
        # Ensure indices are in valid range
        source_indices = tf.math.mod(source_indices, num_nodes)
        target_indices = tf.math.mod(target_indices, num_nodes)
        
        # Project link features to node feature space
        link_projected = self.link_to_node_projection(link_features)
        
        # Initialize node features
        node_aggregated = tf.zeros([num_nodes, self.hparams['node_state_dim']], dtype=link_features.dtype)
        
        # Aggregate to source nodes
        for i in range(tf.shape(source_indices)[0]):
            src = source_indices[i]
            node_aggregated = tf.tensor_scatter_nd_add(
                node_aggregated,
                [[src]],
                [link_projected[i]]
            )
        
        # Aggregate to target nodes
        for i in range(tf.shape(target_indices)[0]):
            tgt = target_indices[i]
            node_aggregated = tf.tensor_scatter_nd_add(
                node_aggregated,
                [[tgt]],
                [link_projected[i]]
            )
        
        # Normalize - calculate degree of each node
        degrees = tf.zeros([num_nodes], dtype=tf.float32)
        for idx in source_indices:
            degrees = tf.tensor_scatter_nd_add(degrees, [[idx]], [1.0])
        for idx in target_indices:
            degrees = tf.tensor_scatter_nd_add(degrees, [[idx]], [1.0])
        
        # Avoid division by zero
        degrees = tf.maximum(degrees, 1.0)
        
        # Normalize by degree
        for i in range(num_nodes):
            node_aggregated = tf.tensor_scatter_nd_update(
                node_aggregated,
                [[i]],
                [node_aggregated[i] / degrees[i]]
            )
        
        return node_aggregated

    def call(self, link_states, node_states, graph_ids, first, second, num_edges=None, num_nodes=None, training=False):
        """
        Forward pass - supporting joint optimization.
        
        Args:
            link_states: Link state features
            node_states: Node state features
            graph_ids: Graph IDs for each link
            first: Source node indices
            second: Target node indices
            num_edges: Number of edges
            num_nodes: Number of nodes
            training: Whether in training mode
            
        Returns:
            route_probs: Route selection probabilities
            node_probs: Node selection probabilities
        """
        # Save input for residual connections
        link_identity = link_states
        node_identity = node_states
        
        # Edge index for message passing
        edge_index = tf.stack([first, second], axis=0)
        
        # Feature transformation
        link_x = self.link_feature_transform(link_states)
        node_x = self.node_feature_transform(node_states)
        
        # 1. Link layer message passing
        link_attn_output = self.compute_link_message_passing(link_x, edge_index, training)
        
        # Add dropout
        if training:
            link_attn_output = tf.nn.dropout(link_attn_output, rate=self.hparams['dropout_rate'])
        
        # Link layer residual connection and layer normalization
        num_nodes_actual = tf.shape(node_x)[0]
        if tf.shape(link_identity)[0] > tf.shape(link_attn_output)[0]:
            link_identity_matched = link_identity[:tf.shape(link_attn_output)[0]]
        elif tf.shape(link_identity)[0] < tf.shape(link_attn_output)[0]:
            padding = tf.shape(link_attn_output)[0] - tf.shape(link_identity)[0]
            link_identity_matched = tf.pad(link_identity, [[0, padding], [0, 0]])
        else:
            link_identity_matched = link_identity
            
        link_x = self.link_layer_norm1(link_attn_output + link_identity_matched)
        
        # Link layer feed-forward network
        link_ffn_output = self.link_ffn(link_x)
        
        # Link layer second residual connection and layer normalization
        link_x = self.link_layer_norm2(link_x + link_ffn_output)
        
        # 2. Aggregate from link features to node features
        link_to_node = self.link_to_node_aggregation(link_x, edge_index, num_nodes_actual)
        
        # Merge link-aggregated information with original node features
        node_x = node_x + link_to_node * 0.5  # Use half-connection to avoid over-influencing node features
        
        # 3. Node layer message passing
        node_attn_output = self.compute_node_message_passing(node_x, edge_index, training)
        
        # Add dropout
        if training:
            node_attn_output = tf.nn.dropout(node_attn_output, rate=self.hparams['dropout_rate'])
        
        # Node layer residual connection and layer normalization
        if tf.shape(node_identity)[0] > tf.shape(node_attn_output)[0]:
            node_identity_matched = node_identity[:tf.shape(node_attn_output)[0]]
        elif tf.shape(node_identity)[0] < tf.shape(node_attn_output)[0]:
            padding = tf.shape(node_attn_output)[0] - tf.shape(node_identity)[0]
            node_identity_matched = tf.pad(node_identity, [[0, padding], [0, 0]])
        else:
            node_identity_matched = node_identity
            
        node_x = self.node_layer_norm1(node_attn_output + node_identity_matched)
        
        # Node layer feed-forward network
        node_ffn_output = self.node_ffn(node_x)
        
        # Node layer second residual connection and layer normalization
        node_x = self.node_layer_norm2(node_x + node_ffn_output)
        
        # Aggregate features to graph level by graph ID
        # Ensure graph_ids matches link_x length
        if tf.shape(graph_ids)[0] != tf.shape(link_x)[0]:
            if tf.shape(graph_ids)[0] > tf.shape(link_x)[0]:
                graph_ids_matched = graph_ids[:tf.shape(link_x)[0]]
            else:
                padding = tf.shape(link_x)[0] - tf.shape(graph_ids)[0]
                graph_ids_matched = tf.pad(graph_ids, [[0, padding]], constant_values=0)
        else:
            graph_ids_matched = graph_ids
        
        # Get unique graph IDs
        unique_graph_ids, _ = tf.unique(graph_ids_matched)
        num_graphs = tf.shape(unique_graph_ids)[0]
        
        # Initialize graph embeddings
        graph_link_embeddings = tf.zeros([num_graphs, self.hparams['link_state_dim']], dtype=link_x.dtype)
        graph_node_embeddings = tf.zeros([num_graphs, self.hparams['node_state_dim']], dtype=node_x.dtype)
        
        # Aggregate link features by graph ID
        for i in range(num_graphs):
            graph_id = unique_graph_ids[i]
            mask = tf.equal(graph_ids_matched, graph_id)
            graph_links = tf.boolean_mask(link_x, mask)
            graph_link_embedding = tf.reduce_mean(graph_links, axis=0)
            
            graph_link_embeddings = tf.tensor_scatter_nd_update(
                graph_link_embeddings,
                [[i]],
                [graph_link_embedding]
            )
            
            # Corresponding node features (use all nodes)
            graph_node_embedding = tf.reduce_mean(node_x, axis=0)
            
            graph_node_embeddings = tf.tensor_scatter_nd_update(
                graph_node_embeddings,
                [[i]],
                [graph_node_embedding]
            )
        
        # Generate route action probabilities through readout network
        route_probs = self.route_readout_net(graph_link_embeddings, training=training)
        
        # Generate node selection probabilities through readout network
        node_probs = self.node_readout_net(graph_node_embeddings, training=training)
        
        # Return joint action probabilities
        return route_probs, node_probs