import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.diffusion_conditioning = DiffusionConditioning(config)
        self.atom_attention_encoder = AtomAttentionEncoder(config)
        self.diffusion_transformer_blocks = nn.ModuleList([
            DiffusionTransformerBlock(config) for _ in range(config.num_blocks)
        ])
        self.atom_attention_decoder = AtomAttentionDecoder(config)
        
        self.post_attn_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_atom_cond_layernorm = nn.LayerNorm(config.atom_feature_size)
        self.to_pos_updates = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 3)
        )

    def forward(self, token_single_initial_repr, token_pair_initial_repr, token_single_trunk_repr, 
                token_pair_trunk_repr, atom_single_input_feats, atom_block_pair_input_feats, 
                atom_single_mask, atom_block_pair_mask, token_single_mask, block_indices_h, 
                block_indices_w, atom_noised_coords, noise_sigma, atom_token_indices):
        
        diffusion_cond = self.diffusion_conditioning(
            token_single_initial_repr, token_pair_initial_repr, 
            token_single_trunk_repr, token_pair_trunk_repr, noise_sigma
        )
        
        atom_encoding = self.atom_attention_encoder(
            atom_single_input_feats, atom_block_pair_input_feats,
            atom_single_mask, atom_block_pair_mask, atom_noised_coords
        )
        
        hidden_states = diffusion_cond
        for block in self.diffusion_transformer_blocks:
            hidden_states = block(hidden_states, atom_encoding)
        
        atom_decoding = self.atom_attention_decoder(hidden_states, atom_encoding)
        
        hidden_states = self.post_attn_layernorm(hidden_states)
        atom_decoding = self.post_atom_cond_layernorm(atom_decoding)
        
        pos_updates = self.to_pos_updates(atom_decoding)
        
        noise_scale = self.get_noise_scale(noise_sigma)
        pos_updates = pos_updates * noise_scale
        
        final_coords = atom_noised_coords + pos_updates
        
        return final_coords

    def get_noise_scale(self, noise_sigma):
        return 16.0 * noise_sigma / torch.sqrt(noise_sigma**2 + 256.0)

class DiffusionConditioning(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_pair_proj = nn.Sequential(
            nn.LayerNorm(config.token_pair_size * 2),
            nn.Linear(config.token_pair_size * 2, config.hidden_size)
        )
        self.token_in_proj = nn.Sequential(
            nn.LayerNorm(config.token_single_size * 2),
            nn.Linear(config.token_single_size * 2, config.hidden_size)
        )
        self.single_trans1 = TransformerLayer(config)
        self.single_trans2 = TransformerLayer(config)
        self.pair_trans1 = TransformerLayer(config)
        self.pair_trans2 = TransformerLayer(config)
        self.fourier_embedding = FourierEmbedding(config)
        self.fourier_proj = nn.Sequential(
            nn.Linear(config.fourier_size, config.hidden_size),
            nn.ReLU()
        )
        self.single_ln = nn.LayerNorm(config.hidden_size)
        self.pair_ln = nn.LayerNorm(config.hidden_size)

    def forward(self, token_single_initial_repr, token_pair_initial_repr, 
                token_single_trunk_repr, token_pair_trunk_repr, noise_sigma):
        pair_repr = self.token_pair_proj(torch.cat([token_pair_initial_repr, token_pair_trunk_repr], dim=-1))
        single_repr = self.token_in_proj(torch.cat([token_single_initial_repr, token_single_trunk_repr], dim=-1))
        
        pair_repr = self.pair_trans1(pair_repr)
        pair_repr = self.pair_trans2(pair_repr)
        
        fourier_emb = self.fourier_embedding(noise_sigma)
        fourier_proj = self.fourier_proj(fourier_emb)
        
        single_repr = single_repr + fourier_proj
        single_repr = self.single_trans1(single_repr)
        single_repr = self.single_trans2(single_repr)
        
        single_repr = self.single_ln(single_repr)
        pair_repr = self.pair_ln(pair_repr)
        
        return single_repr, pair_repr

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        residual = x
        x = self.layer_norm1(x)
        x = self.attention(x, x, x)
        x = x + residual

        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = x + residual
        return x

class FourierEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(config.fourier_size // 2))
        self.bias = nn.Parameter(torch.randn(config.fourier_size // 2))

    def forward(self, x):
        x = x.unsqueeze(-1) * self.weights + self.bias
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

class AtomAttentionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.to_atom_cond = nn.Linear(config.atom_feature_size, config.hidden_size)
        self.token_to_atom_single = nn.Sequential(
            nn.LayerNorm(config.token_single_size),
            nn.Linear(config.token_single_size, config.hidden_size)
        )
        self.prev_pos_embed = nn.Embedding(config.max_atoms, config.hidden_size)
        self.pair_update_block = PairUpdateBlock(config)
        self.atom_transformer = AtomTransformer(config)
        self.to_token_single = nn.Linear(config.hidden_size, config.token_single_size)
        self.token_pair_to_atom_pair = nn.Sequential(
            nn.Linear(config.token_pair_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.pair_feature_size)
        )

    def forward(self, atom_single_input_feats, atom_block_pair_input_feats,
                atom_single_mask, atom_block_pair_mask, atom_noised_coords):
        atom_cond = self.to_atom_cond(atom_single_input_feats)
        token_atom_single = self.token_to_atom_single(atom_single_input_feats)
        
        prev_pos_embed = self.prev_pos_embed(torch.arange(atom_noised_coords.size(1), device=atom_noised_coords.device))
        atom_cond = atom_cond + prev_pos_embed

        pair_feats = self.pair_update_block(atom_cond, atom_block_pair_input_feats)
        
        atom_encoding = self.atom_transformer(atom_cond, pair_feats, atom_single_mask, atom_block_pair_mask)
        
        token_single_out = self.to_token_single(atom_encoding)
        token_pair_out = self.token_pair_to_atom_pair(pair_feats)
        
        return atom_encoding, token_single_out, token_pair_out

class PairUpdateBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.atom_single_to_atom_pair_proj_h = nn.Linear(config.hidden_size, config.pair_feature_size)
        self.atom_single_to_atom_pair_proj_w = nn.Linear(config.hidden_size, config.pair_feature_size)
        self.atom_pair_mlp = nn.Sequential(
            nn.Linear(config.pair_feature_size, config.pair_feature_size),
            nn.ReLU(),
            nn.Linear(config.pair_feature_size, config.pair_feature_size)
        )

    def forward(self, atom_single, atom_pair):
        proj_h = self.atom_single_to_atom_pair_proj_h(atom_single).unsqueeze(2)
        proj_w = self.atom_single_to_atom_pair_proj_w(atom_single).unsqueeze(1)
        pair_feats = atom_pair + proj_h + proj_w
        pair_feats = self.atom_pair_mlp(pair_feats)
        return pair_feats

class AtomTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.local_diffn_transformer = LocalDiffnTransformer(config)

    def forward(self, atom_single, pair_feats, atom_single_mask, atom_block_pair_mask):
        return self.local_diffn_transformer(atom_single, pair_feats, atom_single_mask, atom_block_pair_mask)

class LocalDiffnTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transitions = nn.ModuleList([TransitionLayer(config) for _ in range(config.num_transitions)])
        self.local_attentions = nn.ModuleList([LocalAttention(config) for _ in range(config.num_local_attentions)])
        self.blocked_pairs2blocked_bias = nn.Sequential(
            nn.Linear(config.pair_feature_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1)
        )

    def forward(self, atom_single, pair_feats, atom_single_mask, atom_block_pair_mask):
        for transition, local_attention in zip(self.transitions, self.local_attentions):
            atom_single = transition(atom_single)
            blocked_bias = self.blocked_pairs2blocked_bias(pair_feats).squeeze(-1)
            atom_single = local_attention(atom_single, blocked_bias, atom_single_mask, atom_block_pair_mask)
        return atom_single

class TransitionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.SiLU()

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x + residual

class LocalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x, blocked_bias, atom_single_mask, atom_block_pair_mask):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        attn_weights = attn_weights + blocked_bias.unsqueeze(1)

        attn_mask = atom_single_mask.unsqueeze(1) & atom_block_pair_mask.unsqueeze(2)
        attn_weights = attn_weights.masked_fill(~attn_mask, float('-inf'))

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.transition = TransitionLayer(config)
        self.gate_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Sigmoid()
        )
        self.pair_layer_norm = nn.LayerNorm(config.pair_feature_size)
        self.pair_linear = nn.Linear(config.pair_feature_size, config.hidden_size)

    def forward(self, x, context):
        # Self-attention
        residual = x
        x = self.layer_norm1(x)
        x = self.attention(x, x, x)
        x = x + residual

        # Feed-forward
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = x + residual

        # Transition
        x = self.transition(x)

        # Gating
        gate = self.gate_proj(x)
        x = x * gate

        # Pair interaction
        pair_context = self.pair_layer_norm(context)
        pair_context = self.pair_linear(pair_context)
        x = x + pair_context

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.size()

        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class AtomAttentionDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_to_atom = nn.Linear(config.hidden_size, config.atom_feature_size)
        self.atom_transformer = AtomTransformer(config)
        self.to_pos_updates = nn.Sequential(
            nn.LayerNorm(config.atom_feature_size),
            nn.Linear(config.atom_feature_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 3)
        )

    def forward(self, hidden_states, atom_encoding):
        atom_features = self.token_to_atom(hidden_states)
        atom_features = atom_features + atom_encoding
        
        decoded_atoms = self.atom_transformer(atom_features, atom_encoding)
        
        pos_updates = self.to_pos_updates(decoded_atoms)
        
        return pos_updates

# Additional helper function that might be useful
def get_noise_scale(noise_sigma):
    return 16.0 * noise_sigma / torch.sqrt(noise_sigma**2 + 256.0)

# Configuration class to hold all the parameters
class DiffusionConfig:
    def __init__(self):
        self.hidden_size = 768
        self.intermediate_size = 3072
        self.num_attention_heads = 12
        self.num_hidden_layers = 12
        self.max_atoms = 1000
        self.atom_feature_size = 128
        self.pair_feature_size = 16
        self.token_single_size = 384
        self.token_pair_size = 256
        self.fourier_size = 256
        self.num_transitions = 3
        self.num_local_attentions = 3
        self.num_blocks = 12


config = DiffusionConfig()
model = DiffusionModule(config)

# Then use the model in your training or inference loop
output = model(token_single_initial_repr, token_pair_initial_repr, token_single_trunk_repr, 
               token_pair_trunk_repr, atom_single_input_feats, atom_block_pair_input_feats, 
               atom_single_mask, atom_block_pair_mask, token_single_mask, block_indices_h, 
               block_indices_w, atom_noised_coords, noise_sigma, atom_token_indices)
