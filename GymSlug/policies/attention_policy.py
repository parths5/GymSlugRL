"""
Custom Attention Policy for SAC
Based on Stable Baselines SAC FeedForwardPolicy with self-attention mechanism
"""

import tensorflow as tf
import numpy as np
from stable_baselines.sac.policies import FeedForwardPolicy, LOG_STD_MIN, LOG_STD_MAX
from stable_baselines.sac.policies import gaussian_likelihood, gaussian_entropy, apply_squashing_func
from stable_baselines.common.tf_layers import linear, ortho_init


class AttentionPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network with attention mechanism.
    
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """
    
    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, 
                 attention_heads=4, attention_dim=64, **kwargs):
        # Store attention parameters before calling super
        self.attention_heads = attention_heads
        self.attention_dim = attention_dim
        
        # Ensure feature_extraction is 'mlp' not 'cnn' for 1D observations
        kwargs.setdefault('feature_extraction', 'mlp')
        
        # Call parent init - FeedForwardPolicy handles the basic setup
        super(AttentionPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, **kwargs)
    
    def _attention_layer(self, inputs, name, reuse=None):
        """
        Multi-head self-attention layer for 2D inputs [batch, features]
        Treats each feature dimension as a separate token for self-attention
        
        :param inputs: (tf.Tensor) Input tensor of shape [batch_size, input_dim]
        :param name: (str) Name scope
        :param reuse: (bool) Whether to reuse variables
        :return: (tf.Tensor) Output tensor after attention [batch_size, input_dim]
        """
        with tf.variable_scope(name, reuse=reuse):
            input_dim = inputs.get_shape()[-1].value
            batch_size = tf.shape(inputs)[0]
            seq_len = input_dim  # Each feature dimension is a token
            
            # Treat each feature as a token: [batch, input_dim] -> [batch, input_dim, 1]
            # Then embed to attention_dim: [batch, seq_len, embed_dim]
            inputs_reshaped = tf.expand_dims(inputs, 2)  # [batch, seq_len, 1]
            embed_dim = self.attention_dim
            embedded = tf.layers.dense(inputs_reshaped, embed_dim,
                                      kernel_initializer=ortho_init(np.sqrt(2)), name='embed')
            # embedded: [batch, seq_len, embed_dim]
            
            # Linear projections for Q, K, V: [batch, seq_len, heads * head_dim]
            Q = tf.layers.dense(embedded, self.attention_dim * self.attention_heads, 
                               kernel_initializer=ortho_init(np.sqrt(2)), name='query')
            K = tf.layers.dense(embedded, self.attention_dim * self.attention_heads,
                               kernel_initializer=ortho_init(np.sqrt(2)), name='key')
            V = tf.layers.dense(embedded, self.attention_dim * self.attention_heads,
                               kernel_initializer=ortho_init(np.sqrt(2)), name='value')
            
            # Reshape for multi-head attention: [batch, seq_len, heads, head_dim]
            Q = tf.reshape(Q, [batch_size, seq_len, self.attention_heads, self.attention_dim])
            K = tf.reshape(K, [batch_size, seq_len, self.attention_heads, self.attention_dim])
            V = tf.reshape(V, [batch_size, seq_len, self.attention_heads, self.attention_dim])
            
            # Transpose to [batch, heads, seq_len, head_dim]
            Q = tf.transpose(Q, [0, 2, 1, 3])
            K = tf.transpose(K, [0, 2, 1, 3])
            V = tf.transpose(V, [0, 2, 1, 3])
            
            # Scaled dot-product attention
            # Q @ K^T: [batch, heads, seq_len, seq_len] - each token attends to all tokens
            scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(self.attention_dim))
            attention_weights = tf.nn.softmax(scores, axis=-1)
            
            # Apply attention to values: [batch, heads, seq_len, head_dim]
            attended = tf.matmul(attention_weights, V)
            
            # Concatenate heads: [batch, seq_len, heads * head_dim]
            attended = tf.transpose(attended, [0, 2, 1, 3])  # [batch, seq_len, heads, head_dim]
            attended = tf.reshape(attended, [batch_size, seq_len, self.attention_heads * self.attention_dim])
            
            # Output projection: [batch, seq_len, embed_dim]
            output = tf.layers.dense(attended, embed_dim,
                                    kernel_initializer=ortho_init(np.sqrt(2)), name='output_proj')
            
            # Residual connection: add embedded input
            output = output + embedded
            
            # Layer normalization
            output = tf.contrib.layers.layer_norm(output)
            
            # Project back to 1D per token: [batch, seq_len, embed_dim] -> [batch, seq_len, 1]
            output = tf.layers.dense(output, 1,
                                    kernel_initializer=ortho_init(np.sqrt(2)), name='final_proj')
            
            # Squeeze to remove last dimension: [batch, seq_len, 1] -> [batch, seq_len]
            output = tf.squeeze(output, 2)
            
            # Final residual connection with original input
            output = output + inputs
            
            # Final layer normalization
            output = tf.contrib.layers.layer_norm(output)
            
            return output
    
    def make_actor(self, obs=None, reuse=False, scope="pi"):
        """
        Creates an actor (policy) network with attention mechanism.
        This overrides the parent class method to add attention.
        
        :param obs: (tf.Tensor) The observation placeholder (can be None for default shape)
        :param reuse: (bool) Whether or not to reuse parameters
        :param scope: (str) the scope name of the actor
        :return: (tf.Tensor, tf.Tensor, tf.Tensor) deterministic_policy, policy, logp_pi
        """
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            # Flatten observations if needed
            pi_h = tf.layers.flatten(obs)
            
            # Extract features using attention network
            pi_h = self._attention_network(pi_h, 'attention_features')

            # Output mean and log_std for actions
            self.act_mu = mu_ = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)
            log_std = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)

        # Clip log_std to valid range
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        self.std = std = tf.exp(log_std)
        # Reparameterization trick
        pi_ = mu_ + tf.random_normal(tf.shape(mu_)) * std
        logp_pi = gaussian_likelihood(pi_, mu_, log_std)
        self.entropy = gaussian_entropy(log_std)
        
        # Apply squashing and account for it in the probability
        deterministic_policy, policy, logp_pi = apply_squashing_func(mu_, pi_, logp_pi)
        self.policy = policy
        self.deterministic_policy = deterministic_policy

        return deterministic_policy, policy, logp_pi
    
    def _attention_network(self, obs, name):
        """
        Feature extraction network with attention mechanism
        Applies attention to the original observation dimensions (7) before MLP expansion
        
        :param obs: (tf.Tensor) Processed observations
        :param name: (str) Name scope
        :return: (tf.Tensor) Extracted features
        """
        with tf.variable_scope(name):
            # Apply attention to original observation dimensions first (much cheaper: 7x7 vs 256x256)
            # This learns relationships between the 7 observation features
            attended_obs = self._attention_layer(obs, 'attention_obs')
            
            # Then expand with MLP layers
            pi_l1 = tf.layers.dense(attended_obs, 256, activation=tf.nn.tanh,
                                   kernel_initializer=ortho_init(np.sqrt(2)), name='pi_l1')
            pi_l1_norm = tf.contrib.layers.layer_norm(pi_l1)
            
            # Final feature extraction
            pi_l2 = tf.layers.dense(pi_l1_norm, 256, activation=tf.nn.tanh,
                                   kernel_initializer=ortho_init(np.sqrt(2)), name='pi_l2')
            pi_l2_norm = tf.contrib.layers.layer_norm(pi_l2)
            
            return pi_l2_norm

