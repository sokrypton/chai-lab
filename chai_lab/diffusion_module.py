def forward(self,
            token_single_initial_repr,
            token_pair_initial_repr,
            token_single_trunk_repr,
            token_pair_trunk_repr,
            atom_single_input_feats,
            atom_block_pair_input_feats,
            atom_single_mask,
            atom_block_pair_mask,
            token_single_mask,
            block_indices_h,
            block_indices_w,
            atom_noised_coords,
            noise_sigma,
            atom_token_indices,
            num_diffn_samples=5):
    arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, = fx_pytree.tree_flatten_spec(([], {'token_single_initial_repr':token_single_initial_repr, 'token_pair_initial_repr':token_pair_initial_repr, 'token_single_trunk_repr':token_single_trunk_repr, 'token_pair_trunk_repr':token_pair_trunk_repr, 'atom_single_input_feats':atom_single_input_feats, 'atom_block_pair_input_feats':atom_block_pair_input_feats, 'atom_single_mask':atom_single_mask, 'atom_block_pair_mask':atom_block_pair_mask, 'token_single_mask':token_single_mask, 'block_indices_h':block_indices_h, 'block_indices_w':block_indices_w, 'atom_noised_coords':atom_noised_coords, 'noise_sigma':noise_sigma, 'atom_token_indices':atom_token_indices}), self._in_spec)
    diffusion_conditioning_token_pair_proj_0_weight = getattr(self.diffusion_conditioning.token_pair_proj, "0").weight
    diffusion_conditioning_token_pair_proj_0_bias = getattr(self.diffusion_conditioning.token_pair_proj, "0").bias
    diffusion_conditioning_token_pair_proj_1_weight = getattr(self.diffusion_conditioning.token_pair_proj, "1").weight
    diffusion_conditioning_token_in_proj_0_weight = getattr(self.diffusion_conditioning.token_in_proj, "0").weight
    diffusion_conditioning_token_in_proj_0_bias = getattr(self.diffusion_conditioning.token_in_proj, "0").bias
    diffusion_conditioning_token_in_proj_1_weight = getattr(self.diffusion_conditioning.token_in_proj, "1").weight
    diffusion_conditioning_single_trans1_layer_norm_weight = self.diffusion_conditioning.single_trans1.layer_norm.weight
    diffusion_conditioning_single_trans1_layer_norm_bias = self.diffusion_conditioning.single_trans1.layer_norm.bias
    diffusion_conditioning_single_trans1_linear_no_bias_ab_weight = self.diffusion_conditioning.single_trans1.linear_no_bias_ab.weight
    diffusion_conditioning_single_trans1_linear_out_weight = self.diffusion_conditioning.single_trans1.linear_out.weight
    diffusion_conditioning_pair_trans1_layer_norm_weight = self.diffusion_conditioning.pair_trans1.layer_norm.weight
    diffusion_conditioning_pair_trans1_layer_norm_bias = self.diffusion_conditioning.pair_trans1.layer_norm.bias
    diffusion_conditioning_pair_trans1_linear_no_bias_ab_weight = self.diffusion_conditioning.pair_trans1.linear_no_bias_ab.weight
    diffusion_conditioning_pair_trans1_linear_out_weight = self.diffusion_conditioning.pair_trans1.linear_out.weight
    diffusion_conditioning_single_trans2_layer_norm_weight = self.diffusion_conditioning.single_trans2.layer_norm.weight
    diffusion_conditioning_single_trans2_layer_norm_bias = self.diffusion_conditioning.single_trans2.layer_norm.bias
    diffusion_conditioning_single_trans2_linear_no_bias_ab_weight = self.diffusion_conditioning.single_trans2.linear_no_bias_ab.weight
    diffusion_conditioning_single_trans2_linear_out_weight = self.diffusion_conditioning.single_trans2.linear_out.weight
    diffusion_conditioning_pair_trans2_layer_norm_weight = self.diffusion_conditioning.pair_trans2.layer_norm.weight
    diffusion_conditioning_pair_trans2_layer_norm_bias = self.diffusion_conditioning.pair_trans2.layer_norm.bias
    diffusion_conditioning_pair_trans2_linear_no_bias_ab_weight = self.diffusion_conditioning.pair_trans2.linear_no_bias_ab.weight
    diffusion_conditioning_pair_trans2_linear_out_weight = self.diffusion_conditioning.pair_trans2.linear_out.weight
    diffusion_conditioning_fourier_embedding_weights = self.diffusion_conditioning.fourier_embedding.weights
    diffusion_conditioning_fourier_embedding_bias = self.diffusion_conditioning.fourier_embedding.bias
    diffusion_conditioning_fourier_proj_0_weight = getattr(self.diffusion_conditioning.fourier_proj, "0").weight
    diffusion_conditioning_fourier_proj_0_bias = getattr(self.diffusion_conditioning.fourier_proj, "0").bias
    diffusion_conditioning_fourier_proj_1_weight = getattr(self.diffusion_conditioning.fourier_proj, "1").weight
    diffusion_conditioning_single_ln_weight = self.diffusion_conditioning.single_ln.weight
    diffusion_conditioning_single_ln_bias = self.diffusion_conditioning.single_ln.bias
    diffusion_conditioning_pair_ln_weight = self.diffusion_conditioning.pair_ln.weight
    diffusion_conditioning_pair_ln_bias = self.diffusion_conditioning.pair_ln.bias
    atom_attention_encoder_to_atom_cond_weight = self.atom_attention_encoder.to_atom_cond.weight
    atom_attention_encoder_token_to_atom_single_0_weight = getattr(self.atom_attention_encoder.token_to_atom_single, "0").weight
    atom_attention_encoder_token_to_atom_single_0_bias = getattr(self.atom_attention_encoder.token_to_atom_single, "0").bias
    atom_attention_encoder_token_to_atom_single_1_weight = getattr(self.atom_attention_encoder.token_to_atom_single, "1").weight
    atom_attention_encoder_prev_pos_embed_weight = self.atom_attention_encoder.prev_pos_embed.weight
    atom_attention_encoder_pair_update_block_atom_single_to_atom_pair_proj_h_1_weight = getattr(self.atom_attention_encoder.pair_update_block.atom_single_to_atom_pair_proj_h, "1").weight
    atom_attention_encoder_pair_update_block_atom_single_to_atom_pair_proj_w_1_weight = getattr(self.atom_attention_encoder.pair_update_block.atom_single_to_atom_pair_proj_w, "1").weight
    atom_attention_encoder_pair_update_block_atom_pair_mlp_0_weight = getattr(self.atom_attention_encoder.pair_update_block.atom_pair_mlp, "0").weight
    atom_attention_encoder_pair_update_block_atom_pair_mlp_2_weight = getattr(self.atom_attention_encoder.pair_update_block.atom_pair_mlp, "2").weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_0_ada_ln_lin_s_merged_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "0").ada_ln.lin_s_merged.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_0_linear_a_nobias_double_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_a_nobias_double.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_0_linear_b_nobias_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_b_nobias.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_0_linear_s_biasinit_m2_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_s_biasinit_m2.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_0_linear_s_biasinit_m2_bias = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_s_biasinit_m2.bias
    atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_1_ada_ln_lin_s_merged_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "1").ada_ln.lin_s_merged.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_1_linear_a_nobias_double_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_a_nobias_double.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_1_linear_b_nobias_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_b_nobias.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_1_linear_s_biasinit_m2_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_s_biasinit_m2.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_1_linear_s_biasinit_m2_bias = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_s_biasinit_m2.bias
    atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_2_ada_ln_lin_s_merged_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "2").ada_ln.lin_s_merged.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_2_linear_a_nobias_double_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_a_nobias_double.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_2_linear_b_nobias_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_b_nobias.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_2_linear_s_biasinit_m2_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_s_biasinit_m2.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_2_linear_s_biasinit_m2_bias = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_s_biasinit_m2.bias
    atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_0_q_bias = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "0").q_bias
    atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_0_single_layer_norm_lin_s_merged_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "0").single_layer_norm.lin_s_merged.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_0_to_qkv_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "0").to_qkv.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_0_out_proj_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "0").out_proj.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_0_out_proj_bias = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "0").out_proj.bias
    atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_1_q_bias = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "1").q_bias
    atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_1_single_layer_norm_lin_s_merged_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "1").single_layer_norm.lin_s_merged.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_1_to_qkv_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "1").to_qkv.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_1_out_proj_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "1").out_proj.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_1_out_proj_bias = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "1").out_proj.bias
    atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_2_q_bias = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "2").q_bias
    atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_2_single_layer_norm_lin_s_merged_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "2").single_layer_norm.lin_s_merged.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_2_to_qkv_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "2").to_qkv.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_2_out_proj_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "2").out_proj.weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_2_out_proj_bias = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.local_attentions, "2").out_proj.bias
    atom_attention_encoder_atom_transformer_local_diffn_transformer_blocked_pairs2blocked_bias_0_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.blocked_pairs2blocked_bias, "0").weight
    atom_attention_encoder_atom_transformer_local_diffn_transformer_blocked_pairs2blocked_bias_0_bias = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.blocked_pairs2blocked_bias, "0").bias
    atom_attention_encoder_atom_transformer_local_diffn_transformer_blocked_pairs2blocked_bias_1_weight = getattr(self.atom_attention_encoder.atom_transformer.local_diffn_transformer.blocked_pairs2blocked_bias, "1").weight
    atom_attention_encoder_to_token_single_0_weight = getattr(self.atom_attention_encoder.to_token_single, "0").weight
    atom_attention_encoder_token_pair_to_atom_pair_0_weight = getattr(self.atom_attention_encoder.token_pair_to_atom_pair, "0").weight
    atom_attention_encoder_token_pair_to_atom_pair_0_bias = getattr(self.atom_attention_encoder.token_pair_to_atom_pair, "0").bias
    atom_attention_encoder_token_pair_to_atom_pair_1_weight = getattr(self.atom_attention_encoder.token_pair_to_atom_pair, "1").weight
    diffusion_transformer_blocks_0_q_bias = getattr(self.diffusion_transformer.blocks, "0").q_bias
    diffusion_transformer_blocks_0_transition_ada_ln_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "0").transition.ada_ln.lin_s_merged.weight
    diffusion_transformer_blocks_0_transition_linear_a_nobias_double_weight = getattr(self.diffusion_transformer.blocks, "0").transition.linear_a_nobias_double.weight
    diffusion_transformer_blocks_0_transition_linear_b_nobias_weight = getattr(self.diffusion_transformer.blocks, "0").transition.linear_b_nobias.weight
    diffusion_transformer_blocks_0_transition_linear_s_biasinit_m2_weight = getattr(self.diffusion_transformer.blocks, "0").transition.linear_s_biasinit_m2.weight
    diffusion_transformer_blocks_0_transition_linear_s_biasinit_m2_bias = getattr(self.diffusion_transformer.blocks, "0").transition.linear_s_biasinit_m2.bias
    diffusion_transformer_blocks_0_norm_in_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "0").norm_in.lin_s_merged.weight
    diffusion_transformer_blocks_0_to_qkv_weight = getattr(self.diffusion_transformer.blocks, "0").to_qkv.weight
    diffusion_transformer_blocks_0_gate_proj_0_weight = getattr(getattr(self.diffusion_transformer.blocks, "0").gate_proj, "0").weight
    diffusion_transformer_blocks_0_gate_proj_0_bias = getattr(getattr(self.diffusion_transformer.blocks, "0").gate_proj, "0").bias
    diffusion_transformer_blocks_0_pair_layer_norm_weight = getattr(self.diffusion_transformer.blocks, "0").pair_layer_norm.weight
    diffusion_transformer_blocks_0_pair_layer_norm_bias = getattr(self.diffusion_transformer.blocks, "0").pair_layer_norm.bias
    diffusion_transformer_blocks_0_pair_linear_weight = getattr(self.diffusion_transformer.blocks, "0").pair_linear.weight
    diffusion_transformer_blocks_0_to_out_weight = getattr(self.diffusion_transformer.blocks, "0").to_out.weight
    diffusion_transformer_blocks_1_q_bias = getattr(self.diffusion_transformer.blocks, "1").q_bias
    diffusion_transformer_blocks_1_transition_ada_ln_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "1").transition.ada_ln.lin_s_merged.weight
    diffusion_transformer_blocks_1_transition_linear_a_nobias_double_weight = getattr(self.diffusion_transformer.blocks, "1").transition.linear_a_nobias_double.weight
    diffusion_transformer_blocks_1_transition_linear_b_nobias_weight = getattr(self.diffusion_transformer.blocks, "1").transition.linear_b_nobias.weight
    diffusion_transformer_blocks_1_transition_linear_s_biasinit_m2_weight = getattr(self.diffusion_transformer.blocks, "1").transition.linear_s_biasinit_m2.weight
    diffusion_transformer_blocks_1_transition_linear_s_biasinit_m2_bias = getattr(self.diffusion_transformer.blocks, "1").transition.linear_s_biasinit_m2.bias
    diffusion_transformer_blocks_1_norm_in_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "1").norm_in.lin_s_merged.weight
    diffusion_transformer_blocks_1_to_qkv_weight = getattr(self.diffusion_transformer.blocks, "1").to_qkv.weight
    diffusion_transformer_blocks_1_gate_proj_0_weight = getattr(getattr(self.diffusion_transformer.blocks, "1").gate_proj, "0").weight
    diffusion_transformer_blocks_1_gate_proj_0_bias = getattr(getattr(self.diffusion_transformer.blocks, "1").gate_proj, "0").bias
    diffusion_transformer_blocks_1_pair_layer_norm_weight = getattr(self.diffusion_transformer.blocks, "1").pair_layer_norm.weight
    diffusion_transformer_blocks_1_pair_layer_norm_bias = getattr(self.diffusion_transformer.blocks, "1").pair_layer_norm.bias
    diffusion_transformer_blocks_1_pair_linear_weight = getattr(self.diffusion_transformer.blocks, "1").pair_linear.weight
    diffusion_transformer_blocks_1_to_out_weight = getattr(self.diffusion_transformer.blocks, "1").to_out.weight
    diffusion_transformer_blocks_2_q_bias = getattr(self.diffusion_transformer.blocks, "2").q_bias
    diffusion_transformer_blocks_2_transition_ada_ln_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "2").transition.ada_ln.lin_s_merged.weight
    diffusion_transformer_blocks_2_transition_linear_a_nobias_double_weight = getattr(self.diffusion_transformer.blocks, "2").transition.linear_a_nobias_double.weight
    diffusion_transformer_blocks_2_transition_linear_b_nobias_weight = getattr(self.diffusion_transformer.blocks, "2").transition.linear_b_nobias.weight
    diffusion_transformer_blocks_2_transition_linear_s_biasinit_m2_weight = getattr(self.diffusion_transformer.blocks, "2").transition.linear_s_biasinit_m2.weight
    diffusion_transformer_blocks_2_transition_linear_s_biasinit_m2_bias = getattr(self.diffusion_transformer.blocks, "2").transition.linear_s_biasinit_m2.bias
    diffusion_transformer_blocks_2_norm_in_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "2").norm_in.lin_s_merged.weight
    diffusion_transformer_blocks_2_to_qkv_weight = getattr(self.diffusion_transformer.blocks, "2").to_qkv.weight
    diffusion_transformer_blocks_2_gate_proj_0_weight = getattr(getattr(self.diffusion_transformer.blocks, "2").gate_proj, "0").weight
    diffusion_transformer_blocks_2_gate_proj_0_bias = getattr(getattr(self.diffusion_transformer.blocks, "2").gate_proj, "0").bias
    diffusion_transformer_blocks_2_pair_layer_norm_weight = getattr(self.diffusion_transformer.blocks, "2").pair_layer_norm.weight
    diffusion_transformer_blocks_2_pair_layer_norm_bias = getattr(self.diffusion_transformer.blocks, "2").pair_layer_norm.bias
    diffusion_transformer_blocks_2_pair_linear_weight = getattr(self.diffusion_transformer.blocks, "2").pair_linear.weight
    diffusion_transformer_blocks_2_to_out_weight = getattr(self.diffusion_transformer.blocks, "2").to_out.weight
    diffusion_transformer_blocks_3_q_bias = getattr(self.diffusion_transformer.blocks, "3").q_bias
    diffusion_transformer_blocks_3_transition_ada_ln_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "3").transition.ada_ln.lin_s_merged.weight
    diffusion_transformer_blocks_3_transition_linear_a_nobias_double_weight = getattr(self.diffusion_transformer.blocks, "3").transition.linear_a_nobias_double.weight
    diffusion_transformer_blocks_3_transition_linear_b_nobias_weight = getattr(self.diffusion_transformer.blocks, "3").transition.linear_b_nobias.weight
    diffusion_transformer_blocks_3_transition_linear_s_biasinit_m2_weight = getattr(self.diffusion_transformer.blocks, "3").transition.linear_s_biasinit_m2.weight
    diffusion_transformer_blocks_3_transition_linear_s_biasinit_m2_bias = getattr(self.diffusion_transformer.blocks, "3").transition.linear_s_biasinit_m2.bias
    diffusion_transformer_blocks_3_norm_in_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "3").norm_in.lin_s_merged.weight
    diffusion_transformer_blocks_3_to_qkv_weight = getattr(self.diffusion_transformer.blocks, "3").to_qkv.weight
    diffusion_transformer_blocks_3_gate_proj_0_weight = getattr(getattr(self.diffusion_transformer.blocks, "3").gate_proj, "0").weight
    diffusion_transformer_blocks_3_gate_proj_0_bias = getattr(getattr(self.diffusion_transformer.blocks, "3").gate_proj, "0").bias
    diffusion_transformer_blocks_3_pair_layer_norm_weight = getattr(self.diffusion_transformer.blocks, "3").pair_layer_norm.weight
    diffusion_transformer_blocks_3_pair_layer_norm_bias = getattr(self.diffusion_transformer.blocks, "3").pair_layer_norm.bias
    diffusion_transformer_blocks_3_pair_linear_weight = getattr(self.diffusion_transformer.blocks, "3").pair_linear.weight
    diffusion_transformer_blocks_3_to_out_weight = getattr(self.diffusion_transformer.blocks, "3").to_out.weight
    diffusion_transformer_blocks_4_q_bias = getattr(self.diffusion_transformer.blocks, "4").q_bias
    diffusion_transformer_blocks_4_transition_ada_ln_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "4").transition.ada_ln.lin_s_merged.weight
    diffusion_transformer_blocks_4_transition_linear_a_nobias_double_weight = getattr(self.diffusion_transformer.blocks, "4").transition.linear_a_nobias_double.weight
    diffusion_transformer_blocks_4_transition_linear_b_nobias_weight = getattr(self.diffusion_transformer.blocks, "4").transition.linear_b_nobias.weight
    diffusion_transformer_blocks_4_transition_linear_s_biasinit_m2_weight = getattr(self.diffusion_transformer.blocks, "4").transition.linear_s_biasinit_m2.weight
    diffusion_transformer_blocks_4_transition_linear_s_biasinit_m2_bias = getattr(self.diffusion_transformer.blocks, "4").transition.linear_s_biasinit_m2.bias
    diffusion_transformer_blocks_4_norm_in_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "4").norm_in.lin_s_merged.weight
    diffusion_transformer_blocks_4_to_qkv_weight = getattr(self.diffusion_transformer.blocks, "4").to_qkv.weight
    diffusion_transformer_blocks_4_gate_proj_0_weight = getattr(getattr(self.diffusion_transformer.blocks, "4").gate_proj, "0").weight
    diffusion_transformer_blocks_4_gate_proj_0_bias = getattr(getattr(self.diffusion_transformer.blocks, "4").gate_proj, "0").bias
    diffusion_transformer_blocks_4_pair_layer_norm_weight = getattr(self.diffusion_transformer.blocks, "4").pair_layer_norm.weight
    diffusion_transformer_blocks_4_pair_layer_norm_bias = getattr(self.diffusion_transformer.blocks, "4").pair_layer_norm.bias
    diffusion_transformer_blocks_4_pair_linear_weight = getattr(self.diffusion_transformer.blocks, "4").pair_linear.weight
    diffusion_transformer_blocks_4_to_out_weight = getattr(self.diffusion_transformer.blocks, "4").to_out.weight
    diffusion_transformer_blocks_5_q_bias = getattr(self.diffusion_transformer.blocks, "5").q_bias
    diffusion_transformer_blocks_5_transition_ada_ln_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "5").transition.ada_ln.lin_s_merged.weight
    diffusion_transformer_blocks_5_transition_linear_a_nobias_double_weight = getattr(self.diffusion_transformer.blocks, "5").transition.linear_a_nobias_double.weight
    diffusion_transformer_blocks_5_transition_linear_b_nobias_weight = getattr(self.diffusion_transformer.blocks, "5").transition.linear_b_nobias.weight
    diffusion_transformer_blocks_5_transition_linear_s_biasinit_m2_weight = getattr(self.diffusion_transformer.blocks, "5").transition.linear_s_biasinit_m2.weight
    diffusion_transformer_blocks_5_transition_linear_s_biasinit_m2_bias = getattr(self.diffusion_transformer.blocks, "5").transition.linear_s_biasinit_m2.bias
    diffusion_transformer_blocks_5_norm_in_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "5").norm_in.lin_s_merged.weight
    diffusion_transformer_blocks_5_to_qkv_weight = getattr(self.diffusion_transformer.blocks, "5").to_qkv.weight
    diffusion_transformer_blocks_5_gate_proj_0_weight = getattr(getattr(self.diffusion_transformer.blocks, "5").gate_proj, "0").weight
    diffusion_transformer_blocks_5_gate_proj_0_bias = getattr(getattr(self.diffusion_transformer.blocks, "5").gate_proj, "0").bias
    diffusion_transformer_blocks_5_pair_layer_norm_weight = getattr(self.diffusion_transformer.blocks, "5").pair_layer_norm.weight
    diffusion_transformer_blocks_5_pair_layer_norm_bias = getattr(self.diffusion_transformer.blocks, "5").pair_layer_norm.bias
    diffusion_transformer_blocks_5_pair_linear_weight = getattr(self.diffusion_transformer.blocks, "5").pair_linear.weight
    diffusion_transformer_blocks_5_to_out_weight = getattr(self.diffusion_transformer.blocks, "5").to_out.weight
    diffusion_transformer_blocks_6_q_bias = getattr(self.diffusion_transformer.blocks, "6").q_bias
    diffusion_transformer_blocks_6_transition_ada_ln_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "6").transition.ada_ln.lin_s_merged.weight
    diffusion_transformer_blocks_6_transition_linear_a_nobias_double_weight = getattr(self.diffusion_transformer.blocks, "6").transition.linear_a_nobias_double.weight
    diffusion_transformer_blocks_6_transition_linear_b_nobias_weight = getattr(self.diffusion_transformer.blocks, "6").transition.linear_b_nobias.weight
    diffusion_transformer_blocks_6_transition_linear_s_biasinit_m2_weight = getattr(self.diffusion_transformer.blocks, "6").transition.linear_s_biasinit_m2.weight
    diffusion_transformer_blocks_6_transition_linear_s_biasinit_m2_bias = getattr(self.diffusion_transformer.blocks, "6").transition.linear_s_biasinit_m2.bias
    diffusion_transformer_blocks_6_norm_in_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "6").norm_in.lin_s_merged.weight
    diffusion_transformer_blocks_6_to_qkv_weight = getattr(self.diffusion_transformer.blocks, "6").to_qkv.weight
    diffusion_transformer_blocks_6_gate_proj_0_weight = getattr(getattr(self.diffusion_transformer.blocks, "6").gate_proj, "0").weight
    diffusion_transformer_blocks_6_gate_proj_0_bias = getattr(getattr(self.diffusion_transformer.blocks, "6").gate_proj, "0").bias
    diffusion_transformer_blocks_6_pair_layer_norm_weight = getattr(self.diffusion_transformer.blocks, "6").pair_layer_norm.weight
    diffusion_transformer_blocks_6_pair_layer_norm_bias = getattr(self.diffusion_transformer.blocks, "6").pair_layer_norm.bias
    diffusion_transformer_blocks_6_pair_linear_weight = getattr(self.diffusion_transformer.blocks, "6").pair_linear.weight
    diffusion_transformer_blocks_6_to_out_weight = getattr(self.diffusion_transformer.blocks, "6").to_out.weight
    diffusion_transformer_blocks_7_q_bias = getattr(self.diffusion_transformer.blocks, "7").q_bias
    diffusion_transformer_blocks_7_transition_ada_ln_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "7").transition.ada_ln.lin_s_merged.weight
    diffusion_transformer_blocks_7_transition_linear_a_nobias_double_weight = getattr(self.diffusion_transformer.blocks, "7").transition.linear_a_nobias_double.weight
    diffusion_transformer_blocks_7_transition_linear_b_nobias_weight = getattr(self.diffusion_transformer.blocks, "7").transition.linear_b_nobias.weight
    diffusion_transformer_blocks_7_transition_linear_s_biasinit_m2_weight = getattr(self.diffusion_transformer.blocks, "7").transition.linear_s_biasinit_m2.weight
    diffusion_transformer_blocks_7_transition_linear_s_biasinit_m2_bias = getattr(self.diffusion_transformer.blocks, "7").transition.linear_s_biasinit_m2.bias
    diffusion_transformer_blocks_7_norm_in_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "7").norm_in.lin_s_merged.weight
    diffusion_transformer_blocks_7_to_qkv_weight = getattr(self.diffusion_transformer.blocks, "7").to_qkv.weight
    diffusion_transformer_blocks_7_gate_proj_0_weight = getattr(getattr(self.diffusion_transformer.blocks, "7").gate_proj, "0").weight
    diffusion_transformer_blocks_7_gate_proj_0_bias = getattr(getattr(self.diffusion_transformer.blocks, "7").gate_proj, "0").bias
    diffusion_transformer_blocks_7_pair_layer_norm_weight = getattr(self.diffusion_transformer.blocks, "7").pair_layer_norm.weight
    diffusion_transformer_blocks_7_pair_layer_norm_bias = getattr(self.diffusion_transformer.blocks, "7").pair_layer_norm.bias
    diffusion_transformer_blocks_7_pair_linear_weight = getattr(self.diffusion_transformer.blocks, "7").pair_linear.weight
    diffusion_transformer_blocks_7_to_out_weight = getattr(self.diffusion_transformer.blocks, "7").to_out.weight
    diffusion_transformer_blocks_8_q_bias = getattr(self.diffusion_transformer.blocks, "8").q_bias
    diffusion_transformer_blocks_8_transition_ada_ln_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "8").transition.ada_ln.lin_s_merged.weight
    diffusion_transformer_blocks_8_transition_linear_a_nobias_double_weight = getattr(self.diffusion_transformer.blocks, "8").transition.linear_a_nobias_double.weight
    diffusion_transformer_blocks_8_transition_linear_b_nobias_weight = getattr(self.diffusion_transformer.blocks, "8").transition.linear_b_nobias.weight
    diffusion_transformer_blocks_8_transition_linear_s_biasinit_m2_weight = getattr(self.diffusion_transformer.blocks, "8").transition.linear_s_biasinit_m2.weight
    diffusion_transformer_blocks_8_transition_linear_s_biasinit_m2_bias = getattr(self.diffusion_transformer.blocks, "8").transition.linear_s_biasinit_m2.bias
    diffusion_transformer_blocks_8_norm_in_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "8").norm_in.lin_s_merged.weight
    diffusion_transformer_blocks_8_to_qkv_weight = getattr(self.diffusion_transformer.blocks, "8").to_qkv.weight
    diffusion_transformer_blocks_8_gate_proj_0_weight = getattr(getattr(self.diffusion_transformer.blocks, "8").gate_proj, "0").weight
    diffusion_transformer_blocks_8_gate_proj_0_bias = getattr(getattr(self.diffusion_transformer.blocks, "8").gate_proj, "0").bias
    diffusion_transformer_blocks_8_pair_layer_norm_weight = getattr(self.diffusion_transformer.blocks, "8").pair_layer_norm.weight
    diffusion_transformer_blocks_8_pair_layer_norm_bias = getattr(self.diffusion_transformer.blocks, "8").pair_layer_norm.bias
    diffusion_transformer_blocks_8_pair_linear_weight = getattr(self.diffusion_transformer.blocks, "8").pair_linear.weight
    diffusion_transformer_blocks_8_to_out_weight = getattr(self.diffusion_transformer.blocks, "8").to_out.weight
    diffusion_transformer_blocks_9_q_bias = getattr(self.diffusion_transformer.blocks, "9").q_bias
    diffusion_transformer_blocks_9_transition_ada_ln_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "9").transition.ada_ln.lin_s_merged.weight
    diffusion_transformer_blocks_9_transition_linear_a_nobias_double_weight = getattr(self.diffusion_transformer.blocks, "9").transition.linear_a_nobias_double.weight
    diffusion_transformer_blocks_9_transition_linear_b_nobias_weight = getattr(self.diffusion_transformer.blocks, "9").transition.linear_b_nobias.weight
    diffusion_transformer_blocks_9_transition_linear_s_biasinit_m2_weight = getattr(self.diffusion_transformer.blocks, "9").transition.linear_s_biasinit_m2.weight
    diffusion_transformer_blocks_9_transition_linear_s_biasinit_m2_bias = getattr(self.diffusion_transformer.blocks, "9").transition.linear_s_biasinit_m2.bias
    diffusion_transformer_blocks_9_norm_in_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "9").norm_in.lin_s_merged.weight
    diffusion_transformer_blocks_9_to_qkv_weight = getattr(self.diffusion_transformer.blocks, "9").to_qkv.weight
    diffusion_transformer_blocks_9_gate_proj_0_weight = getattr(getattr(self.diffusion_transformer.blocks, "9").gate_proj, "0").weight
    diffusion_transformer_blocks_9_gate_proj_0_bias = getattr(getattr(self.diffusion_transformer.blocks, "9").gate_proj, "0").bias
    diffusion_transformer_blocks_9_pair_layer_norm_weight = getattr(self.diffusion_transformer.blocks, "9").pair_layer_norm.weight
    diffusion_transformer_blocks_9_pair_layer_norm_bias = getattr(self.diffusion_transformer.blocks, "9").pair_layer_norm.bias
    diffusion_transformer_blocks_9_pair_linear_weight = getattr(self.diffusion_transformer.blocks, "9").pair_linear.weight
    diffusion_transformer_blocks_9_to_out_weight = getattr(self.diffusion_transformer.blocks, "9").to_out.weight
    diffusion_transformer_blocks_10_q_bias = getattr(self.diffusion_transformer.blocks, "10").q_bias
    diffusion_transformer_blocks_10_transition_ada_ln_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "10").transition.ada_ln.lin_s_merged.weight
    diffusion_transformer_blocks_10_transition_linear_a_nobias_double_weight = getattr(self.diffusion_transformer.blocks, "10").transition.linear_a_nobias_double.weight
    diffusion_transformer_blocks_10_transition_linear_b_nobias_weight = getattr(self.diffusion_transformer.blocks, "10").transition.linear_b_nobias.weight
    diffusion_transformer_blocks_10_transition_linear_s_biasinit_m2_weight = getattr(self.diffusion_transformer.blocks, "10").transition.linear_s_biasinit_m2.weight
    diffusion_transformer_blocks_10_transition_linear_s_biasinit_m2_bias = getattr(self.diffusion_transformer.blocks, "10").transition.linear_s_biasinit_m2.bias
    diffusion_transformer_blocks_10_norm_in_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "10").norm_in.lin_s_merged.weight
    diffusion_transformer_blocks_10_to_qkv_weight = getattr(self.diffusion_transformer.blocks, "10").to_qkv.weight
    diffusion_transformer_blocks_10_gate_proj_0_weight = getattr(getattr(self.diffusion_transformer.blocks, "10").gate_proj, "0").weight
    diffusion_transformer_blocks_10_gate_proj_0_bias = getattr(getattr(self.diffusion_transformer.blocks, "10").gate_proj, "0").bias
    diffusion_transformer_blocks_10_pair_layer_norm_weight = getattr(self.diffusion_transformer.blocks, "10").pair_layer_norm.weight
    diffusion_transformer_blocks_10_pair_layer_norm_bias = getattr(self.diffusion_transformer.blocks, "10").pair_layer_norm.bias
    diffusion_transformer_blocks_10_pair_linear_weight = getattr(self.diffusion_transformer.blocks, "10").pair_linear.weight
    diffusion_transformer_blocks_10_to_out_weight = getattr(self.diffusion_transformer.blocks, "10").to_out.weight
    diffusion_transformer_blocks_11_q_bias = getattr(self.diffusion_transformer.blocks, "11").q_bias
    diffusion_transformer_blocks_11_transition_ada_ln_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "11").transition.ada_ln.lin_s_merged.weight
    diffusion_transformer_blocks_11_transition_linear_a_nobias_double_weight = getattr(self.diffusion_transformer.blocks, "11").transition.linear_a_nobias_double.weight
    diffusion_transformer_blocks_11_transition_linear_b_nobias_weight = getattr(self.diffusion_transformer.blocks, "11").transition.linear_b_nobias.weight
    diffusion_transformer_blocks_11_transition_linear_s_biasinit_m2_weight = getattr(self.diffusion_transformer.blocks, "11").transition.linear_s_biasinit_m2.weight
    diffusion_transformer_blocks_11_transition_linear_s_biasinit_m2_bias = getattr(self.diffusion_transformer.blocks, "11").transition.linear_s_biasinit_m2.bias
    diffusion_transformer_blocks_11_norm_in_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "11").norm_in.lin_s_merged.weight
    diffusion_transformer_blocks_11_to_qkv_weight = getattr(self.diffusion_transformer.blocks, "11").to_qkv.weight
    diffusion_transformer_blocks_11_gate_proj_0_weight = getattr(getattr(self.diffusion_transformer.blocks, "11").gate_proj, "0").weight
    diffusion_transformer_blocks_11_gate_proj_0_bias = getattr(getattr(self.diffusion_transformer.blocks, "11").gate_proj, "0").bias
    diffusion_transformer_blocks_11_pair_layer_norm_weight = getattr(self.diffusion_transformer.blocks, "11").pair_layer_norm.weight
    diffusion_transformer_blocks_11_pair_layer_norm_bias = getattr(self.diffusion_transformer.blocks, "11").pair_layer_norm.bias
    diffusion_transformer_blocks_11_pair_linear_weight = getattr(self.diffusion_transformer.blocks, "11").pair_linear.weight
    diffusion_transformer_blocks_11_to_out_weight = getattr(self.diffusion_transformer.blocks, "11").to_out.weight
    diffusion_transformer_blocks_12_q_bias = getattr(self.diffusion_transformer.blocks, "12").q_bias
    diffusion_transformer_blocks_12_transition_ada_ln_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "12").transition.ada_ln.lin_s_merged.weight
    diffusion_transformer_blocks_12_transition_linear_a_nobias_double_weight = getattr(self.diffusion_transformer.blocks, "12").transition.linear_a_nobias_double.weight
    diffusion_transformer_blocks_12_transition_linear_b_nobias_weight = getattr(self.diffusion_transformer.blocks, "12").transition.linear_b_nobias.weight
    diffusion_transformer_blocks_12_transition_linear_s_biasinit_m2_weight = getattr(self.diffusion_transformer.blocks, "12").transition.linear_s_biasinit_m2.weight
    diffusion_transformer_blocks_12_transition_linear_s_biasinit_m2_bias = getattr(self.diffusion_transformer.blocks, "12").transition.linear_s_biasinit_m2.bias
    diffusion_transformer_blocks_12_norm_in_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "12").norm_in.lin_s_merged.weight
    diffusion_transformer_blocks_12_to_qkv_weight = getattr(self.diffusion_transformer.blocks, "12").to_qkv.weight
    diffusion_transformer_blocks_12_gate_proj_0_weight = getattr(getattr(self.diffusion_transformer.blocks, "12").gate_proj, "0").weight
    diffusion_transformer_blocks_12_gate_proj_0_bias = getattr(getattr(self.diffusion_transformer.blocks, "12").gate_proj, "0").bias
    diffusion_transformer_blocks_12_pair_layer_norm_weight = getattr(self.diffusion_transformer.blocks, "12").pair_layer_norm.weight
    diffusion_transformer_blocks_12_pair_layer_norm_bias = getattr(self.diffusion_transformer.blocks, "12").pair_layer_norm.bias
    diffusion_transformer_blocks_12_pair_linear_weight = getattr(self.diffusion_transformer.blocks, "12").pair_linear.weight
    diffusion_transformer_blocks_12_to_out_weight = getattr(self.diffusion_transformer.blocks, "12").to_out.weight
    diffusion_transformer_blocks_13_q_bias = getattr(self.diffusion_transformer.blocks, "13").q_bias
    diffusion_transformer_blocks_13_transition_ada_ln_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "13").transition.ada_ln.lin_s_merged.weight
    diffusion_transformer_blocks_13_transition_linear_a_nobias_double_weight = getattr(self.diffusion_transformer.blocks, "13").transition.linear_a_nobias_double.weight
    diffusion_transformer_blocks_13_transition_linear_b_nobias_weight = getattr(self.diffusion_transformer.blocks, "13").transition.linear_b_nobias.weight
    diffusion_transformer_blocks_13_transition_linear_s_biasinit_m2_weight = getattr(self.diffusion_transformer.blocks, "13").transition.linear_s_biasinit_m2.weight
    diffusion_transformer_blocks_13_transition_linear_s_biasinit_m2_bias = getattr(self.diffusion_transformer.blocks, "13").transition.linear_s_biasinit_m2.bias
    diffusion_transformer_blocks_13_norm_in_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "13").norm_in.lin_s_merged.weight
    diffusion_transformer_blocks_13_to_qkv_weight = getattr(self.diffusion_transformer.blocks, "13").to_qkv.weight
    diffusion_transformer_blocks_13_gate_proj_0_weight = getattr(getattr(self.diffusion_transformer.blocks, "13").gate_proj, "0").weight
    diffusion_transformer_blocks_13_gate_proj_0_bias = getattr(getattr(self.diffusion_transformer.blocks, "13").gate_proj, "0").bias
    diffusion_transformer_blocks_13_pair_layer_norm_weight = getattr(self.diffusion_transformer.blocks, "13").pair_layer_norm.weight
    diffusion_transformer_blocks_13_pair_layer_norm_bias = getattr(self.diffusion_transformer.blocks, "13").pair_layer_norm.bias
    diffusion_transformer_blocks_13_pair_linear_weight = getattr(self.diffusion_transformer.blocks, "13").pair_linear.weight
    diffusion_transformer_blocks_13_to_out_weight = getattr(self.diffusion_transformer.blocks, "13").to_out.weight
    diffusion_transformer_blocks_14_q_bias = getattr(self.diffusion_transformer.blocks, "14").q_bias
    diffusion_transformer_blocks_14_transition_ada_ln_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "14").transition.ada_ln.lin_s_merged.weight
    diffusion_transformer_blocks_14_transition_linear_a_nobias_double_weight = getattr(self.diffusion_transformer.blocks, "14").transition.linear_a_nobias_double.weight
    diffusion_transformer_blocks_14_transition_linear_b_nobias_weight = getattr(self.diffusion_transformer.blocks, "14").transition.linear_b_nobias.weight
    diffusion_transformer_blocks_14_transition_linear_s_biasinit_m2_weight = getattr(self.diffusion_transformer.blocks, "14").transition.linear_s_biasinit_m2.weight
    diffusion_transformer_blocks_14_transition_linear_s_biasinit_m2_bias = getattr(self.diffusion_transformer.blocks, "14").transition.linear_s_biasinit_m2.bias
    diffusion_transformer_blocks_14_norm_in_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "14").norm_in.lin_s_merged.weight
    diffusion_transformer_blocks_14_to_qkv_weight = getattr(self.diffusion_transformer.blocks, "14").to_qkv.weight
    diffusion_transformer_blocks_14_gate_proj_0_weight = getattr(getattr(self.diffusion_transformer.blocks, "14").gate_proj, "0").weight
    diffusion_transformer_blocks_14_gate_proj_0_bias = getattr(getattr(self.diffusion_transformer.blocks, "14").gate_proj, "0").bias
    diffusion_transformer_blocks_14_pair_layer_norm_weight = getattr(self.diffusion_transformer.blocks, "14").pair_layer_norm.weight
    diffusion_transformer_blocks_14_pair_layer_norm_bias = getattr(self.diffusion_transformer.blocks, "14").pair_layer_norm.bias
    diffusion_transformer_blocks_14_pair_linear_weight = getattr(self.diffusion_transformer.blocks, "14").pair_linear.weight
    diffusion_transformer_blocks_14_to_out_weight = getattr(self.diffusion_transformer.blocks, "14").to_out.weight
    diffusion_transformer_blocks_15_q_bias = getattr(self.diffusion_transformer.blocks, "15").q_bias
    diffusion_transformer_blocks_15_transition_ada_ln_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "15").transition.ada_ln.lin_s_merged.weight
    diffusion_transformer_blocks_15_transition_linear_a_nobias_double_weight = getattr(self.diffusion_transformer.blocks, "15").transition.linear_a_nobias_double.weight
    diffusion_transformer_blocks_15_transition_linear_b_nobias_weight = getattr(self.diffusion_transformer.blocks, "15").transition.linear_b_nobias.weight
    diffusion_transformer_blocks_15_transition_linear_s_biasinit_m2_weight = getattr(self.diffusion_transformer.blocks, "15").transition.linear_s_biasinit_m2.weight
    diffusion_transformer_blocks_15_transition_linear_s_biasinit_m2_bias = getattr(self.diffusion_transformer.blocks, "15").transition.linear_s_biasinit_m2.bias
    diffusion_transformer_blocks_15_norm_in_lin_s_merged_weight = getattr(self.diffusion_transformer.blocks, "15").norm_in.lin_s_merged.weight
    diffusion_transformer_blocks_15_to_qkv_weight = getattr(self.diffusion_transformer.blocks, "15").to_qkv.weight
    diffusion_transformer_blocks_15_gate_proj_0_weight = getattr(getattr(self.diffusion_transformer.blocks, "15").gate_proj, "0").weight
    diffusion_transformer_blocks_15_gate_proj_0_bias = getattr(getattr(self.diffusion_transformer.blocks, "15").gate_proj, "0").bias
    diffusion_transformer_blocks_15_pair_layer_norm_weight = getattr(self.diffusion_transformer.blocks, "15").pair_layer_norm.weight
    diffusion_transformer_blocks_15_pair_layer_norm_bias = getattr(self.diffusion_transformer.blocks, "15").pair_layer_norm.bias
    diffusion_transformer_blocks_15_pair_linear_weight = getattr(self.diffusion_transformer.blocks, "15").pair_linear.weight
    diffusion_transformer_blocks_15_to_out_weight = getattr(self.diffusion_transformer.blocks, "15").to_out.weight
    atom_attention_decoder_token_to_atom_weight = self.atom_attention_decoder.token_to_atom.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_0_ada_ln_lin_s_merged_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "0").ada_ln.lin_s_merged.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_0_linear_a_nobias_double_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_a_nobias_double.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_0_linear_b_nobias_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_b_nobias.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_0_linear_s_biasinit_m2_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_s_biasinit_m2.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_0_linear_s_biasinit_m2_bias = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "0").linear_s_biasinit_m2.bias
    atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_1_ada_ln_lin_s_merged_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "1").ada_ln.lin_s_merged.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_1_linear_a_nobias_double_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_a_nobias_double.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_1_linear_b_nobias_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_b_nobias.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_1_linear_s_biasinit_m2_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_s_biasinit_m2.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_1_linear_s_biasinit_m2_bias = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "1").linear_s_biasinit_m2.bias
    atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_2_ada_ln_lin_s_merged_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "2").ada_ln.lin_s_merged.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_2_linear_a_nobias_double_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_a_nobias_double.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_2_linear_b_nobias_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_b_nobias.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_2_linear_s_biasinit_m2_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_s_biasinit_m2.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_2_linear_s_biasinit_m2_bias = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.transitions, "2").linear_s_biasinit_m2.bias
    atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_0_q_bias = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "0").q_bias
    atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_0_single_layer_norm_lin_s_merged_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "0").single_layer_norm.lin_s_merged.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_0_to_qkv_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "0").to_qkv.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_0_out_proj_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "0").out_proj.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_0_out_proj_bias = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "0").out_proj.bias
    atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_1_q_bias = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "1").q_bias
    atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_1_single_layer_norm_lin_s_merged_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "1").single_layer_norm.lin_s_merged.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_1_to_qkv_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "1").to_qkv.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_1_out_proj_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "1").out_proj.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_1_out_proj_bias = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "1").out_proj.bias
    atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_2_q_bias = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "2").q_bias
    atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_2_single_layer_norm_lin_s_merged_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "2").single_layer_norm.lin_s_merged.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_2_to_qkv_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "2").to_qkv.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_2_out_proj_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "2").out_proj.weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_2_out_proj_bias = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.local_attentions, "2").out_proj.bias
    atom_attention_decoder_atom_transformer_local_diffn_transformer_blocked_pairs2blocked_bias_0_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.blocked_pairs2blocked_bias, "0").weight
    atom_attention_decoder_atom_transformer_local_diffn_transformer_blocked_pairs2blocked_bias_0_bias = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.blocked_pairs2blocked_bias, "0").bias
    atom_attention_decoder_atom_transformer_local_diffn_transformer_blocked_pairs2blocked_bias_1_weight = getattr(self.atom_attention_decoder.atom_transformer.local_diffn_transformer.blocked_pairs2blocked_bias, "1").weight
    atom_attention_decoder_to_pos_updates_0_weight = getattr(self.atom_attention_decoder.to_pos_updates, "0").weight
    atom_attention_decoder_to_pos_updates_0_bias = getattr(self.atom_attention_decoder.to_pos_updates, "0").bias
    atom_attention_decoder_to_pos_updates_1_weight = getattr(self.atom_attention_decoder.to_pos_updates, "1").weight
    structure_cond_to_token_structure_proj_weight = self.structure_cond_to_token_structure_proj.weight
    post_attn_layernorm_weight = self.post_attn_layernorm.weight
    post_attn_layernorm_bias = self.post_attn_layernorm.bias
    post_atom_cond_layernorm_weight = self.post_atom_cond_layernorm.weight
    post_atom_cond_layernorm_bias = self.post_atom_cond_layernorm.bias
    cat = torch.ops.aten.cat.default([arg346_1, arg344_1], dim = -1);  arg346_1 = arg344_1 = None
    native_layer_norm_default = torch.ops.aten.native_layer_norm.default(cat, [512], diffusion_conditioning_token_pair_proj_0_weight, diffusion_conditioning_token_pair_proj_0_bias, 1e-05);  cat = diffusion_conditioning_token_pair_proj_0_weight = diffusion_conditioning_token_pair_proj_0_bias = None
    getitem = native_layer_norm_default[0];  native_layer_norm_default = None
    t = torch.ops.aten.t.default(diffusion_conditioning_token_pair_proj_1_weight);  diffusion_conditioning_token_pair_proj_1_weight = None
    view = torch.ops.aten.view.default(getitem, [147456, 512]);  getitem = None
    mm = torch.ops.aten.mm.default(view, t);  view = t = None
    view_1 = torch.ops.aten.view.default(mm, [1, 384, 384, 256]);  mm = None
    split_tensor = torch.ops.aten.split.Tensor(view_1, 384, dim = -2)
    getitem_3 = split_tensor[0];  split_tensor = None
    native_layer_norm_default_1 = torch.ops.aten.native_layer_norm.default(getitem_3, [256], diffusion_conditioning_pair_trans1_layer_norm_weight, diffusion_conditioning_pair_trans1_layer_norm_bias, 1e-05);  getitem_3 = diffusion_conditioning_pair_trans1_layer_norm_weight = diffusion_conditioning_pair_trans1_layer_norm_bias = None
    getitem_4 = native_layer_norm_default_1[0];  native_layer_norm_default_1 = None
    t_1 = torch.ops.aten.t.default(diffusion_conditioning_pair_trans1_linear_no_bias_ab_weight);  diffusion_conditioning_pair_trans1_linear_no_bias_ab_weight = None
    view_2 = torch.ops.aten.view.default(getitem_4, [147456, 256]);  getitem_4 = None
    mm_1 = torch.ops.aten.mm.default(view_2, t_1);  view_2 = t_1 = None
    view_3 = torch.ops.aten.view.default(mm_1, [1, 384, 384, 1024]);  mm_1 = None
    split_tensor_1 = torch.ops.aten.split.Tensor(view_3, 512, dim = -1);  view_3 = None
    getitem_7 = split_tensor_1[0]
    getitem_8 = split_tensor_1[1];  split_tensor_1 = None
    silu = torch.ops.aten.silu.default(getitem_7);  getitem_7 = None
    mul = torch.ops.aten.mul.Tensor(silu, getitem_8);  silu = getitem_8 = None
    t_2 = torch.ops.aten.t.default(diffusion_conditioning_pair_trans1_linear_out_weight);  diffusion_conditioning_pair_trans1_linear_out_weight = None
    view_5 = torch.ops.aten.view.default(mul, [147456, 512]);  mul = None
    mm_2 = torch.ops.aten.mm.default(view_5, t_2);  view_5 = t_2 = None
    view_6 = torch.ops.aten.view.default(mm_2, [1, 384, 384, 256]);  mm_2 = None
    add = torch.ops.aten.add.Tensor(view_1, view_6);  view_1 = view_6 = None
    split_tensor_2 = torch.ops.aten.split.Tensor(add, 384, dim = -2)
    getitem_9 = split_tensor_2[0];  split_tensor_2 = None
    native_layer_norm_default_2 = torch.ops.aten.native_layer_norm.default(getitem_9, [256], diffusion_conditioning_pair_trans2_layer_norm_weight, diffusion_conditioning_pair_trans2_layer_norm_bias, 1e-05);  getitem_9 = diffusion_conditioning_pair_trans2_layer_norm_weight = diffusion_conditioning_pair_trans2_layer_norm_bias = None
    getitem_10 = native_layer_norm_default_2[0];  native_layer_norm_default_2 = None
    t_3 = torch.ops.aten.t.default(diffusion_conditioning_pair_trans2_linear_no_bias_ab_weight);  diffusion_conditioning_pair_trans2_linear_no_bias_ab_weight = None
    view_7 = torch.ops.aten.view.default(getitem_10, [147456, 256]);  getitem_10 = None
    mm_3 = torch.ops.aten.mm.default(view_7, t_3);  view_7 = t_3 = None
    view_8 = torch.ops.aten.view.default(mm_3, [1, 384, 384, 1024]);  mm_3 = None
    split_tensor_3 = torch.ops.aten.split.Tensor(view_8, 512, dim = -1);  view_8 = None
    getitem_13 = split_tensor_3[0]
    getitem_14 = split_tensor_3[1];  split_tensor_3 = None
    silu_1 = torch.ops.aten.silu.default(getitem_13);  getitem_13 = None
    mul_1 = torch.ops.aten.mul.Tensor(silu_1, getitem_14);  silu_1 = getitem_14 = None
    t_4 = torch.ops.aten.t.default(diffusion_conditioning_pair_trans2_linear_out_weight);  diffusion_conditioning_pair_trans2_linear_out_weight = None
    view_10 = torch.ops.aten.view.default(mul_1, [147456, 512]);  mul_1 = None
    mm_4 = torch.ops.aten.mm.default(view_10, t_4);  view_10 = t_4 = None
    view_11 = torch.ops.aten.view.default(mm_4, [1, 384, 384, 256]);  mm_4 = None
    add_1 = torch.ops.aten.add.Tensor(add, view_11);  add = view_11 = None
    cat_1 = torch.ops.aten.cat.default([arg343_1, arg345_1], dim = -1);  arg343_1 = None
    native_layer_norm_default_3 = torch.ops.aten.native_layer_norm.default(cat_1, [768], diffusion_conditioning_token_in_proj_0_weight, diffusion_conditioning_token_in_proj_0_bias, 1e-05);  cat_1 = diffusion_conditioning_token_in_proj_0_weight = diffusion_conditioning_token_in_proj_0_bias = None
    getitem_15 = native_layer_norm_default_3[0];  native_layer_norm_default_3 = None
    t_5 = torch.ops.aten.t.default(diffusion_conditioning_token_in_proj_1_weight);  diffusion_conditioning_token_in_proj_1_weight = None
    view_12 = torch.ops.aten.view.default(getitem_15, [384, 768]);  getitem_15 = None
    mm_5 = torch.ops.aten.mm.default(view_12, t_5);  view_12 = t_5 = None
    view_13 = torch.ops.aten.view.default(mm_5, [1, 384, 384]);  mm_5 = None
    clamp_min = torch.ops.aten.clamp_min.default(arg355_1, 1.1920928955078125e-07)
    log = torch.ops.aten.log.default(clamp_min);  clamp_min = None
    mul_2 = torch.ops.aten.mul.Tensor(log, 0.25);  log = None
    view_14 = torch.ops.aten.view.default(mul_2, [5]);  mul_2 = None
    view_15 = torch.ops.aten.view.default(view_14, [5, 1]);  view_14 = None
    mul_3 = torch.ops.aten.mul.Tensor(view_15, diffusion_conditioning_fourier_embedding_weights);  view_15 = diffusion_conditioning_fourier_embedding_weights = None
    add_2 = torch.ops.aten.add.Tensor(mul_3, diffusion_conditioning_fourier_embedding_bias);  mul_3 = diffusion_conditioning_fourier_embedding_bias = None
    mul_4 = torch.ops.aten.mul.Tensor(add_2, 6.283185307179586);  add_2 = None
    cos = torch.ops.aten.cos.default(mul_4);  mul_4 = None
    view_16 = torch.ops.aten.view.default(cos, [5, 1, 256]);  cos = None
    native_layer_norm_default_4 = torch.ops.aten.native_layer_norm.default(view_16, [256], diffusion_conditioning_fourier_proj_0_weight, diffusion_conditioning_fourier_proj_0_bias, 1e-05);  view_16 = diffusion_conditioning_fourier_proj_0_weight = diffusion_conditioning_fourier_proj_0_bias = None
    getitem_18 = native_layer_norm_default_4[0];  native_layer_norm_default_4 = None
    t_6 = torch.ops.aten.t.default(diffusion_conditioning_fourier_proj_1_weight);  diffusion_conditioning_fourier_proj_1_weight = None
    view_17 = torch.ops.aten.view.default(getitem_18, [5, 256]);  getitem_18 = None
    mm_6 = torch.ops.aten.mm.default(view_17, t_6);  view_17 = t_6 = None
    view_18 = torch.ops.aten.view.default(mm_6, [5, 1, 384]);  mm_6 = None
    view_19 = torch.ops.aten.view.default(view_13, [1, 1, 384, 384]);  view_13 = None
    view_20 = torch.ops.aten.view.default(view_18, [1, 5, 1, 384]);  view_18 = None
    add_3 = torch.ops.aten.add.Tensor(view_19, view_20);  view_19 = view_20 = None
    split_tensor_4 = torch.ops.aten.split.Tensor(add_3, 384, dim = -2)
    getitem_21 = split_tensor_4[0];  split_tensor_4 = None
    native_layer_norm_default_5 = torch.ops.aten.native_layer_norm.default(getitem_21, [384], diffusion_conditioning_single_trans1_layer_norm_weight, diffusion_conditioning_single_trans1_layer_norm_bias, 1e-05);  getitem_21 = diffusion_conditioning_single_trans1_layer_norm_weight = diffusion_conditioning_single_trans1_layer_norm_bias = None
    getitem_22 = native_layer_norm_default_5[0];  native_layer_norm_default_5 = None
    t_7 = torch.ops.aten.t.default(diffusion_conditioning_single_trans1_linear_no_bias_ab_weight);  diffusion_conditioning_single_trans1_linear_no_bias_ab_weight = None
    view_21 = torch.ops.aten.view.default(getitem_22, [1920, 384]);  getitem_22 = None
    mm_7 = torch.ops.aten.mm.default(view_21, t_7);  view_21 = t_7 = None
    view_22 = torch.ops.aten.view.default(mm_7, [1, 5, 384, 1536]);  mm_7 = None
    split_tensor_5 = torch.ops.aten.split.Tensor(view_22, 768, dim = -1);  view_22 = None
    getitem_25 = split_tensor_5[0]
    getitem_26 = split_tensor_5[1];  split_tensor_5 = None
    silu_2 = torch.ops.aten.silu.default(getitem_25);  getitem_25 = None
    mul_5 = torch.ops.aten.mul.Tensor(silu_2, getitem_26);  silu_2 = getitem_26 = None
    t_8 = torch.ops.aten.t.default(diffusion_conditioning_single_trans1_linear_out_weight);  diffusion_conditioning_single_trans1_linear_out_weight = None
    view_24 = torch.ops.aten.view.default(mul_5, [1920, 768]);  mul_5 = None
    mm_8 = torch.ops.aten.mm.default(view_24, t_8);  view_24 = t_8 = None
    view_25 = torch.ops.aten.view.default(mm_8, [1, 5, 384, 384]);  mm_8 = None
    add_4 = torch.ops.aten.add.Tensor(add_3, view_25);  add_3 = view_25 = None
    split_tensor_6 = torch.ops.aten.split.Tensor(add_4, 384, dim = -2)
    getitem_27 = split_tensor_6[0];  split_tensor_6 = None
    native_layer_norm_default_6 = torch.ops.aten.native_layer_norm.default(getitem_27, [384], diffusion_conditioning_single_trans2_layer_norm_weight, diffusion_conditioning_single_trans2_layer_norm_bias, 1e-05);  getitem_27 = diffusion_conditioning_single_trans2_layer_norm_weight = diffusion_conditioning_single_trans2_layer_norm_bias = None
    getitem_28 = native_layer_norm_default_6[0];  native_layer_norm_default_6 = None
    t_9 = torch.ops.aten.t.default(diffusion_conditioning_single_trans2_linear_no_bias_ab_weight);  diffusion_conditioning_single_trans2_linear_no_bias_ab_weight = None
    view_26 = torch.ops.aten.view.default(getitem_28, [1920, 384]);  getitem_28 = None
    mm_9 = torch.ops.aten.mm.default(view_26, t_9);  view_26 = t_9 = None
    view_27 = torch.ops.aten.view.default(mm_9, [1, 5, 384, 1536]);  mm_9 = None
    split_tensor_7 = torch.ops.aten.split.Tensor(view_27, 768, dim = -1);  view_27 = None
    getitem_31 = split_tensor_7[0]
    getitem_32 = split_tensor_7[1];  split_tensor_7 = None
    silu_3 = torch.ops.aten.silu.default(getitem_31);  getitem_31 = None
    mul_6 = torch.ops.aten.mul.Tensor(silu_3, getitem_32);  silu_3 = getitem_32 = None
    t_10 = torch.ops.aten.t.default(diffusion_conditioning_single_trans2_linear_out_weight);  diffusion_conditioning_single_trans2_linear_out_weight = None
    view_29 = torch.ops.aten.view.default(mul_6, [1920, 768]);  mul_6 = None
    mm_10 = torch.ops.aten.mm.default(view_29, t_10);  view_29 = t_10 = None
    view_30 = torch.ops.aten.view.default(mm_10, [1, 5, 384, 384]);  mm_10 = None
    add_5 = torch.ops.aten.add.Tensor(add_4, view_30);  add_4 = view_30 = None
    native_layer_norm_default_7 = torch.ops.aten.native_layer_norm.default(add_5, [384], diffusion_conditioning_single_ln_weight, diffusion_conditioning_single_ln_bias, 1e-05);  add_5 = diffusion_conditioning_single_ln_weight = diffusion_conditioning_single_ln_bias = None
    getitem_33 = native_layer_norm_default_7[0];  native_layer_norm_default_7 = None
    native_layer_norm_default_8 = torch.ops.aten.native_layer_norm.default(add_1, [256], diffusion_conditioning_pair_ln_weight, diffusion_conditioning_pair_ln_bias, 1e-05);  add_1 = diffusion_conditioning_pair_ln_weight = diffusion_conditioning_pair_ln_bias = None
    getitem_36 = native_layer_norm_default_8[0];  native_layer_norm_default_8 = None
    view_31 = torch.ops.aten.view.default(arg355_1, [1, 5, 1, 1]);  arg355_1 = None
    pow_1 = torch.ops.aten.pow.Tensor_Scalar(view_31, 2)
    add_6 = torch.ops.aten.add.Tensor(pow_1, 256.0);  pow_1 = None
    pow_2 = torch.ops.aten.pow.Tensor_Scalar(add_6, -0.5);  add_6 = None
    mul_7 = torch.ops.aten.mul.Tensor(arg354_1, pow_2);  pow_2 = None
    t_11 = torch.ops.aten.t.default(atom_attention_encoder_to_atom_cond_weight);  atom_attention_encoder_to_atom_cond_weight = None
    view_32 = torch.ops.aten.view.default(arg347_1, [8832, 128]);  arg347_1 = None
    mm_11 = torch.ops.aten.mm.default(view_32, t_11);  view_32 = t_11 = None
    view_33 = torch.ops.aten.view.default(mm_11, [1, 8832, 128]);  mm_11 = None
    unsqueeze = torch.ops.aten.unsqueeze.default(view_33, 1)
    expand = torch.ops.aten.expand.default(unsqueeze, [-1, 1, -1, -1]);  unsqueeze = None
    native_layer_norm_default_9 = torch.ops.aten.native_layer_norm.default(arg345_1, [384], atom_attention_encoder_token_to_atom_single_0_weight, atom_attention_encoder_token_to_atom_single_0_bias, 1e-05);  arg345_1 = atom_attention_encoder_token_to_atom_single_0_weight = atom_attention_encoder_token_to_atom_single_0_bias = None
    getitem_39 = native_layer_norm_default_9[0];  native_layer_norm_default_9 = None
    t_12 = torch.ops.aten.t.default(atom_attention_encoder_token_to_atom_single_1_weight);  atom_attention_encoder_token_to_atom_single_1_weight = None
    view_34 = torch.ops.aten.view.default(getitem_39, [384, 384]);  getitem_39 = None
    mm_12 = torch.ops.aten.mm.default(view_34, t_12);  view_34 = t_12 = None
    view_35 = torch.ops.aten.view.default(mm_12, [1, 384, 128]);  mm_12 = None
    arange = torch.ops.aten.arange.default(1, device = device(type='cuda', index=0), pin_memory = False)
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(arange, 1);  arange = None
    index = torch.ops.aten.index.Tensor(view_35, [unsqueeze_1, arg356_1]);  view_35 = unsqueeze_1 = None
    clone = torch.ops.aten.clone.default(index);  index = None
    add_7 = torch.ops.aten.add.Tensor(view_33, clone);  view_33 = clone = None
    native_layer_norm_default_10 = torch.ops.aten.native_layer_norm.default(add_7, [128], None, None, 1e-05);  add_7 = None
    getitem_42 = native_layer_norm_default_10[0];  native_layer_norm_default_10 = None
    t_13 = torch.ops.aten.t.default(atom_attention_encoder_prev_pos_embed_weight);  atom_attention_encoder_prev_pos_embed_weight = None
    view_36 = torch.ops.aten.view.default(mul_7, [44160, 3]);  mul_7 = None
    mm_13 = torch.ops.aten.mm.default(view_36, t_13);  view_36 = t_13 = None
    view_37 = torch.ops.aten.view.default(mm_13, [1, 5, 8832, 128]);  mm_13 = None
    add_8 = torch.ops.aten.add.Tensor(expand, view_37);  expand = view_37 = None
    native_layer_norm_default_11 = torch.ops.aten.native_layer_norm.default(getitem_36, [256], atom_attention_encoder_token_pair_to_atom_pair_0_weight, atom_attention_encoder_token_pair_to_atom_pair_0_bias, 1e-05);  atom_attention_encoder_token_pair_to_atom_pair_0_weight = atom_attention_encoder_token_pair_to_atom_pair_0_bias = None
    getitem_45 = native_layer_norm_default_11[0];  native_layer_norm_default_11 = None
    t_14 = torch.ops.aten.t.default(atom_attention_encoder_token_pair_to_atom_pair_1_weight);  atom_attention_encoder_token_pair_to_atom_pair_1_weight = None
    view_38 = torch.ops.aten.view.default(getitem_45, [147456, 256]);  getitem_45 = None
    mm_14 = torch.ops.aten.mm.default(view_38, t_14);  view_38 = t_14 = None
    view_39 = torch.ops.aten.view.default(mm_14, [1, 384, 384, 16]);  mm_14 = None
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(getitem_42, 1)
    expand_1 = torch.ops.aten.expand.default(unsqueeze_2, [-1, 5, -1, -1]);  unsqueeze_2 = None
    slice_1 = torch.ops.aten.slice.Tensor(expand_1, dim = 0, start = 0, end = 9223372036854775807)
    slice_2 = torch.ops.aten.slice.Tensor(slice_1, dim = 1, start = 0, end = 9223372036854775807);  slice_1 = None
    index_1 = torch.ops.aten.index.Tensor(slice_2, [None, None, arg352_1]);  slice_2 = None
    slice_3 = torch.ops.aten.slice.Tensor(expand_1, dim = 0, start = 0, end = 9223372036854775807);  expand_1 = None
    slice_4 = torch.ops.aten.slice.Tensor(slice_3, dim = 1, start = 0, end = 9223372036854775807);  slice_3 = None
    index_2 = torch.ops.aten.index.Tensor(slice_4, [None, None, arg353_1]);  slice_4 = None
    slice_5 = torch.ops.aten.slice.Tensor(arg356_1, dim = 0, start = 0, end = 9223372036854775807)
    index_3 = torch.ops.aten.index.Tensor(slice_5, [None, arg352_1]);  slice_5 = arg352_1 = None
    slice_6 = torch.ops.aten.slice.Tensor(arg356_1, dim = 0, start = 0, end = 9223372036854775807)
    index_4 = torch.ops.aten.index.Tensor(slice_6, [None, arg353_1]);  slice_6 = arg353_1 = None
    view_40 = torch.ops.aten.view.default(index_3, [1, 276, 32, 1]);  index_3 = None
    view_41 = torch.ops.aten.view.default(index_4, [1, 276, 1, 128]);  index_4 = None
    arange_1 = torch.ops.aten.arange.default(1, device = device(type='cuda', index=0), pin_memory = False)
    view_42 = torch.ops.aten.view.default(arange_1, [-1, 1, 1, 1]);  arange_1 = None
    index_5 = torch.ops.aten.index.Tensor(view_39, [view_42, view_40, view_41]);  view_39 = view_42 = view_40 = view_41 = None
    add_9 = torch.ops.aten.add.Tensor(arg348_1, index_5);  arg348_1 = index_5 = None
    relu = torch.ops.aten.relu.default(index_1);  index_1 = None
    t_15 = torch.ops.aten.t.default(atom_attention_encoder_pair_update_block_atom_single_to_atom_pair_proj_h_1_weight);  atom_attention_encoder_pair_update_block_atom_single_to_atom_pair_proj_h_1_weight = None
    view_43 = torch.ops.aten.view.default(relu, [44160, 128]);  relu = None
    mm_15 = torch.ops.aten.mm.default(view_43, t_15);  view_43 = t_15 = None
    view_44 = torch.ops.aten.view.default(mm_15, [1, 5, 276, 32, 16]);  mm_15 = None
    relu_1 = torch.ops.aten.relu.default(index_2);  index_2 = None
    t_16 = torch.ops.aten.t.default(atom_attention_encoder_pair_update_block_atom_single_to_atom_pair_proj_w_1_weight);  atom_attention_encoder_pair_update_block_atom_single_to_atom_pair_proj_w_1_weight = None
    view_45 = torch.ops.aten.view.default(relu_1, [176640, 128]);  relu_1 = None
    mm_16 = torch.ops.aten.mm.default(view_45, t_16);  view_45 = t_16 = None
    view_46 = torch.ops.aten.view.default(mm_16, [1, 5, 276, 128, 16]);  mm_16 = None
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(add_9, 1);  add_9 = None
    view_47 = torch.ops.aten.view.default(view_44, [1, 5, 276, 32, 1, 16]);  view_44 = None
    add_10 = torch.ops.aten.add.Tensor(unsqueeze_3, view_47);  unsqueeze_3 = view_47 = None
    view_48 = torch.ops.aten.view.default(view_46, [1, 5, 276, 1, 128, 16]);  view_46 = None
    add_11 = torch.ops.aten.add.Tensor(add_10, view_48);  add_10 = view_48 = None
    t_17 = torch.ops.aten.t.default(atom_attention_encoder_pair_update_block_atom_pair_mlp_0_weight);  atom_attention_encoder_pair_update_block_atom_pair_mlp_0_weight = None
    view_49 = torch.ops.aten.view.default(add_11, [5652480, 16])
    mm_17 = torch.ops.aten.mm.default(view_49, t_17);  view_49 = t_17 = None
    view_50 = torch.ops.aten.view.default(mm_17, [1, 5, 276, 32, 128, 16]);  mm_17 = None
    relu_2 = torch.ops.aten.relu.default(view_50);  view_50 = None
    view_51 = torch.ops.aten.view.default(relu_2, [5652480, 16]);  relu_2 = None
    t_18 = torch.ops.aten.t.default(atom_attention_encoder_pair_update_block_atom_pair_mlp_2_weight);  atom_attention_encoder_pair_update_block_atom_pair_mlp_2_weight = None
    view_54 = torch.ops.aten.view.default(view_51, [1, 5, 276, 32, 128, 16]);  view_51 = None
    view_55 = torch.ops.aten.view.default(view_54, [5652480, 16]);  view_54 = None
    mm_18 = torch.ops.aten.mm.default(view_55, t_18);  view_55 = t_18 = None
    view_56 = torch.ops.aten.view.default(mm_18, [1, 5, 276, 32, 128, 16]);  mm_18 = None
    add_12 = torch.ops.aten.add.Tensor(view_56, add_11);  view_56 = add_11 = None
    view_57 = torch.ops.aten.view.default(add_12, [5652480, 16]);  add_12 = None
    view_59 = torch.ops.aten.view.default(add_8, [5, 8832, 128]);  add_8 = None
    unsqueeze_4 = torch.ops.aten.unsqueeze.default(getitem_42, 1)
    expand_2 = torch.ops.aten.expand.default(unsqueeze_4, [-1, 5, -1, -1]);  unsqueeze_4 = None
    view_60 = torch.ops.aten.view.default(expand_2, [5, 8832, 128]);  expand_2 = None
    unsqueeze_5 = torch.ops.aten.unsqueeze.default(arg350_1, 1)
    expand_3 = torch.ops.aten.expand.default(unsqueeze_5, [-1, 5, -1, -1, -1]);  unsqueeze_5 = None
    view_62 = torch.ops.aten.view.default(expand_3, [5, 276, 32, 128]);  expand_3 = None
    unsqueeze_6 = torch.ops.aten.unsqueeze.default(arg349_1, 1)
    expand_4 = torch.ops.aten.expand.default(unsqueeze_6, [-1, 5, -1]);  unsqueeze_6 = None
    view_63 = torch.ops.aten.view.default(expand_4, [5, 8832]);  expand_4 = None
    arange_2 = torch.ops.aten.arange.default(8832, device = device(type='cuda', index=0), pin_memory = False)
    view_64 = torch.ops.aten.view.default(arange_2, [276, 32]);  arange_2 = None
    slice_7 = torch.ops.aten.slice.Tensor(view_64, dim = 0, start = 0, end = 9223372036854775807);  view_64 = None
    slice_8 = torch.ops.aten.slice.Tensor(slice_7, dim = 1, start = 0, end = 1);  slice_7 = None
    add_13 = torch.ops.aten.add.Tensor(slice_8, -48);  slice_8 = None
    arange_3 = torch.ops.aten.arange.default(128, device = device(type='cuda', index=0), pin_memory = False)
    add_14 = torch.ops.aten.add.Tensor(add_13, arange_3);  add_13 = arange_3 = None
    remainder = torch.ops.aten.remainder.Scalar(add_14, 8832);  add_14 = None
    view_65 = torch.ops.aten.view.default(view_57, [1, 5, 276, 32, 128, 16])
    view_66 = torch.ops.aten.view.default(view_65, [5, 276, 32, 128, 16]);  view_65 = None
    native_layer_norm_default_12 = torch.ops.aten.native_layer_norm.default(view_66, [16], atom_attention_encoder_atom_transformer_local_diffn_transformer_blocked_pairs2blocked_bias_0_weight, atom_attention_encoder_atom_transformer_local_diffn_transformer_blocked_pairs2blocked_bias_0_bias, 1e-05);  view_66 = atom_attention_encoder_atom_transformer_local_diffn_transformer_blocked_pairs2blocked_bias_0_weight = atom_attention_encoder_atom_transformer_local_diffn_transformer_blocked_pairs2blocked_bias_0_bias = None
    getitem_48 = native_layer_norm_default_12[0];  native_layer_norm_default_12 = None
    unbind_int = torch.ops.aten.unbind.int(atom_attention_encoder_atom_transformer_local_diffn_transformer_blocked_pairs2blocked_bias_1_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_blocked_pairs2blocked_bias_1_weight = None
    getitem_51 = unbind_int[0]
    getitem_52 = unbind_int[1]
    getitem_53 = unbind_int[2];  unbind_int = None
    unsqueeze_7 = torch.ops.aten.unsqueeze.default(view_63, -1)
    bitwise_not = torch.ops.aten.bitwise_not.default(unsqueeze_7);  unsqueeze_7 = None
    masked_fill = torch.ops.aten.masked_fill.Scalar(view_59, bitwise_not, 0.0);  view_59 = bitwise_not = None
    unsqueeze_8 = torch.ops.aten.unsqueeze.default(getitem_48, 5)
    permute = torch.ops.aten.permute.default(unsqueeze_8, [0, 5, 1, 2, 3, 4]);  unsqueeze_8 = None
    unsqueeze_9 = torch.ops.aten.unsqueeze.default(getitem_51, 2);  getitem_51 = None
    unsqueeze_10 = torch.ops.aten.unsqueeze.default(unsqueeze_9, 3);  unsqueeze_9 = None
    unsqueeze_11 = torch.ops.aten.unsqueeze.default(unsqueeze_10, 4);  unsqueeze_10 = None
    unsqueeze_12 = torch.ops.aten.unsqueeze.default(unsqueeze_11, 5);  unsqueeze_11 = None
    permute_1 = torch.ops.aten.permute.default(unsqueeze_12, [2, 0, 3, 4, 5, 1]);  unsqueeze_12 = None
    permute_2 = torch.ops.aten.permute.default(permute, [0, 2, 3, 4, 5, 1]);  permute = None
    view_67 = torch.ops.aten.view.default(permute_2, [1, 5652480, 16]);  permute_2 = None
    permute_3 = torch.ops.aten.permute.default(permute_1, [5, 1, 0, 2, 3, 4]);  permute_1 = None
    view_68 = torch.ops.aten.view.default(permute_3, [1, 16, 4]);  permute_3 = None
    bmm = torch.ops.aten.bmm.default(view_67, view_68);  view_67 = view_68 = None
    view_69 = torch.ops.aten.view.default(bmm, [5, 276, 32, 128, 1, 4]);  bmm = None
    permute_4 = torch.ops.aten.permute.default(view_69, [0, 5, 1, 2, 3, 4]);  view_69 = None
    view_70 = torch.ops.aten.view.default(permute_4, [5, 4, 276, 32, 128]);  permute_4 = None
    view_71 = torch.ops.aten.view.default(view_62, [5, 1, 276, 32, 128])
    bitwise_not_1 = torch.ops.aten.bitwise_not.default(view_71);  view_71 = None
    masked_fill_1 = torch.ops.aten.masked_fill.Scalar(view_70, bitwise_not_1, -10000);  view_70 = bitwise_not_1 = None
    native_layer_norm_default_13 = torch.ops.aten.native_layer_norm.default(masked_fill, [128], None, None, 0.1)
    getitem_54 = native_layer_norm_default_13[0];  native_layer_norm_default_13 = None
    t_19 = torch.ops.aten.t.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_0_single_layer_norm_lin_s_merged_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_0_single_layer_norm_lin_s_merged_weight = None
    clone_1 = torch.ops.aten.clone.default(view_60, memory_format = torch.contiguous_format)
    _unsafe_view = torch.ops.aten._unsafe_view.default(clone_1, [44160, 128]);  clone_1 = None
    mm_19 = torch.ops.aten.mm.default(_unsafe_view, t_19);  _unsafe_view = t_19 = None
    view_72 = torch.ops.aten.view.default(mm_19, [5, 8832, 256]);  mm_19 = None
    split_tensor_8 = torch.ops.aten.split.Tensor(view_72, 128, dim = -1);  view_72 = None
    getitem_57 = split_tensor_8[0]
    getitem_58 = split_tensor_8[1];  split_tensor_8 = None
    add_15 = torch.ops.aten.add.Tensor(getitem_57, 1);  getitem_57 = None
    mul_8 = torch.ops.aten.mul.Tensor(getitem_54, add_15);  getitem_54 = add_15 = None
    add_16 = torch.ops.aten.add.Tensor(mul_8, getitem_58);  mul_8 = getitem_58 = None
    unsqueeze_13 = torch.ops.aten.unsqueeze.default(add_16, 3);  add_16 = None
    unsqueeze_14 = torch.ops.aten.unsqueeze.default(unsqueeze_13, 4);  unsqueeze_13 = None
    unsqueeze_15 = torch.ops.aten.unsqueeze.default(unsqueeze_14, 5);  unsqueeze_14 = None
    permute_5 = torch.ops.aten.permute.default(unsqueeze_15, [3, 0, 4, 1, 5, 2]);  unsqueeze_15 = None
    unsqueeze_16 = torch.ops.aten.unsqueeze.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_0_to_qkv_weight, 4);  atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_0_to_qkv_weight = None
    unsqueeze_17 = torch.ops.aten.unsqueeze.default(unsqueeze_16, 5);  unsqueeze_16 = None
    permute_6 = torch.ops.aten.permute.default(unsqueeze_17, [0, 4, 1, 5, 2, 3]);  unsqueeze_17 = None
    permute_7 = torch.ops.aten.permute.default(permute_5, [1, 3, 5, 0, 2, 4]);  permute_5 = None
    view_73 = torch.ops.aten.view.default(permute_7, [1, 44160, 128]);  permute_7 = None
    permute_8 = torch.ops.aten.permute.default(permute_6, [5, 0, 2, 4, 1, 3]);  permute_6 = None
    view_74 = torch.ops.aten.view.default(permute_8, [1, 128, 384]);  permute_8 = None
    bmm_1 = torch.ops.aten.bmm.default(view_73, view_74);  view_73 = view_74 = None
    view_75 = torch.ops.aten.view.default(bmm_1, [5, 8832, 1, 3, 4, 32]);  bmm_1 = None
    permute_9 = torch.ops.aten.permute.default(view_75, [3, 0, 4, 1, 5, 2]);  view_75 = None
    view_76 = torch.ops.aten.view.default(permute_9, [3, 5, 4, 8832, 32]);  permute_9 = None
    clone_2 = torch.ops.aten.clone.default(view_76, memory_format = torch.contiguous_format);  view_76 = None
    _unsafe_view_1 = torch.ops.aten._unsafe_view.default(clone_2, [3, 20, 8832, 32]);  clone_2 = None
    unbind_int_1 = torch.ops.aten.unbind.int(_unsafe_view_1);  _unsafe_view_1 = None
    getitem_59 = unbind_int_1[0]
    getitem_60 = unbind_int_1[1]
    getitem_61 = unbind_int_1[2];  unbind_int_1 = None
    unsqueeze_18 = torch.ops.aten.unsqueeze.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_0_q_bias, 0);  atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_0_q_bias = None
    expand_5 = torch.ops.aten.expand.default(unsqueeze_18, [5, -1, -1]);  unsqueeze_18 = None
    clone_3 = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    _unsafe_view_2 = torch.ops.aten._unsafe_view.default(clone_3, [20, 1, 32]);  clone_3 = None
    add_17 = torch.ops.aten.add.Tensor(getitem_59, _unsafe_view_2);  getitem_59 = _unsafe_view_2 = None
    view_77 = torch.ops.aten.view.default(add_17, [20, 276, 32, 32]);  add_17 = None
    slice_9 = torch.ops.aten.slice.Tensor(getitem_60, dim = 0, start = 0, end = 9223372036854775807);  getitem_60 = None
    slice_10 = torch.ops.aten.slice.Tensor(slice_9, dim = 2, start = 0, end = 9223372036854775807);  slice_9 = None
    index_6 = torch.ops.aten.index.Tensor(slice_10, [None, remainder]);  slice_10 = None
    slice_11 = torch.ops.aten.slice.Tensor(getitem_61, dim = 0, start = 0, end = 9223372036854775807);  getitem_61 = None
    slice_12 = torch.ops.aten.slice.Tensor(slice_11, dim = 2, start = 0, end = 9223372036854775807);  slice_11 = None
    index_7 = torch.ops.aten.index.Tensor(slice_12, [None, remainder]);  slice_12 = None
    view_78 = torch.ops.aten.view.default(masked_fill_1, [20, 276, 32, 128]);  masked_fill_1 = None
    expand_6 = torch.ops.aten.expand.default(view_78, [20, 276, 32, 128]);  view_78 = None
    _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_77, index_6, index_7, expand_6, False);  view_77 = index_6 = index_7 = expand_6 = None
    getitem_62 = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
    view_79 = torch.ops.aten.view.default(getitem_62, [5, 4, 276, 32, 32]);  getitem_62 = None
    permute_10 = torch.ops.aten.permute.default(view_79, [0, 2, 3, 1, 4]);  view_79 = None
    clone_4 = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
    _unsafe_view_3 = torch.ops.aten._unsafe_view.default(clone_4, [5, 8832, 128]);  clone_4 = None
    t_20 = torch.ops.aten.t.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_0_out_proj_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_0_out_proj_weight = None
    clone_5 = torch.ops.aten.clone.default(view_60, memory_format = torch.contiguous_format)
    _unsafe_view_4 = torch.ops.aten._unsafe_view.default(clone_5, [44160, 128]);  clone_5 = None
    mm_20 = torch.ops.aten.mm.default(_unsafe_view_4, t_20);  _unsafe_view_4 = t_20 = None
    view_80 = torch.ops.aten.view.default(mm_20, [5, 8832, 128]);  mm_20 = None
    add_18 = torch.ops.aten.add.Tensor(view_80, atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_0_out_proj_bias);  view_80 = atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_0_out_proj_bias = None
    sigmoid = torch.ops.aten.sigmoid.default(add_18);  add_18 = None
    mul_9 = torch.ops.aten.mul.Tensor(_unsafe_view_3, sigmoid);  _unsafe_view_3 = sigmoid = None
    native_layer_norm_default_14 = torch.ops.aten.native_layer_norm.default(masked_fill, [128], None, None, 0.1)
    getitem_66 = native_layer_norm_default_14[0];  native_layer_norm_default_14 = None
    t_21 = torch.ops.aten.t.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_0_ada_ln_lin_s_merged_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_0_ada_ln_lin_s_merged_weight = None
    clone_6 = torch.ops.aten.clone.default(view_60, memory_format = torch.contiguous_format)
    _unsafe_view_5 = torch.ops.aten._unsafe_view.default(clone_6, [44160, 128]);  clone_6 = None
    mm_21 = torch.ops.aten.mm.default(_unsafe_view_5, t_21);  _unsafe_view_5 = t_21 = None
    view_81 = torch.ops.aten.view.default(mm_21, [5, 8832, 256]);  mm_21 = None
    split_tensor_9 = torch.ops.aten.split.Tensor(view_81, 128, dim = -1);  view_81 = None
    getitem_69 = split_tensor_9[0]
    getitem_70 = split_tensor_9[1];  split_tensor_9 = None
    add_19 = torch.ops.aten.add.Tensor(getitem_69, 1);  getitem_69 = None
    mul_10 = torch.ops.aten.mul.Tensor(getitem_66, add_19);  getitem_66 = add_19 = None
    add_20 = torch.ops.aten.add.Tensor(mul_10, getitem_70);  mul_10 = getitem_70 = None
    t_22 = torch.ops.aten.t.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_0_linear_a_nobias_double_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_0_linear_a_nobias_double_weight = None
    view_82 = torch.ops.aten.view.default(add_20, [44160, 128]);  add_20 = None
    mm_22 = torch.ops.aten.mm.default(view_82, t_22);  view_82 = t_22 = None
    view_83 = torch.ops.aten.view.default(mm_22, [5, 8832, 512]);  mm_22 = None
    split_tensor_10 = torch.ops.aten.split.Tensor(view_83, 256, dim = -1);  view_83 = None
    getitem_71 = split_tensor_10[0]
    getitem_72 = split_tensor_10[1];  split_tensor_10 = None
    silu_4 = torch.ops.aten.silu.default(getitem_71);  getitem_71 = None
    mul_11 = torch.ops.aten.mul.Tensor(silu_4, getitem_72);  silu_4 = getitem_72 = None
    t_23 = torch.ops.aten.t.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_0_linear_s_biasinit_m2_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_0_linear_s_biasinit_m2_weight = None
    clone_7 = torch.ops.aten.clone.default(view_60, memory_format = torch.contiguous_format)
    _unsafe_view_6 = torch.ops.aten._unsafe_view.default(clone_7, [44160, 128]);  clone_7 = None
    mm_23 = torch.ops.aten.mm.default(_unsafe_view_6, t_23);  _unsafe_view_6 = t_23 = None
    view_84 = torch.ops.aten.view.default(mm_23, [5, 8832, 128]);  mm_23 = None
    add_21 = torch.ops.aten.add.Tensor(view_84, atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_0_linear_s_biasinit_m2_bias);  view_84 = atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_0_linear_s_biasinit_m2_bias = None
    sigmoid_1 = torch.ops.aten.sigmoid.default(add_21);  add_21 = None
    t_24 = torch.ops.aten.t.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_0_linear_b_nobias_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_0_linear_b_nobias_weight = None
    view_85 = torch.ops.aten.view.default(mul_11, [44160, 256]);  mul_11 = None
    mm_24 = torch.ops.aten.mm.default(view_85, t_24);  view_85 = t_24 = None
    view_86 = torch.ops.aten.view.default(mm_24, [5, 8832, 128]);  mm_24 = None
    mul_12 = torch.ops.aten.mul.Tensor(sigmoid_1, view_86);  sigmoid_1 = view_86 = None
    add_22 = torch.ops.aten.add.Tensor(masked_fill, mul_12);  masked_fill = mul_12 = None
    add_23 = torch.ops.aten.add.Tensor(add_22, mul_9);  add_22 = mul_9 = None
    unsqueeze_19 = torch.ops.aten.unsqueeze.default(view_63, -1)
    bitwise_not_2 = torch.ops.aten.bitwise_not.default(unsqueeze_19);  unsqueeze_19 = None
    masked_fill_2 = torch.ops.aten.masked_fill.Scalar(add_23, bitwise_not_2, 0.0);  add_23 = bitwise_not_2 = None
    unsqueeze_20 = torch.ops.aten.unsqueeze.default(getitem_48, 5)
    permute_11 = torch.ops.aten.permute.default(unsqueeze_20, [0, 5, 1, 2, 3, 4]);  unsqueeze_20 = None
    unsqueeze_21 = torch.ops.aten.unsqueeze.default(getitem_52, 2);  getitem_52 = None
    unsqueeze_22 = torch.ops.aten.unsqueeze.default(unsqueeze_21, 3);  unsqueeze_21 = None
    unsqueeze_23 = torch.ops.aten.unsqueeze.default(unsqueeze_22, 4);  unsqueeze_22 = None
    unsqueeze_24 = torch.ops.aten.unsqueeze.default(unsqueeze_23, 5);  unsqueeze_23 = None
    permute_12 = torch.ops.aten.permute.default(unsqueeze_24, [2, 0, 3, 4, 5, 1]);  unsqueeze_24 = None
    permute_13 = torch.ops.aten.permute.default(permute_11, [0, 2, 3, 4, 5, 1]);  permute_11 = None
    view_87 = torch.ops.aten.view.default(permute_13, [1, 5652480, 16]);  permute_13 = None
    permute_14 = torch.ops.aten.permute.default(permute_12, [5, 1, 0, 2, 3, 4]);  permute_12 = None
    view_88 = torch.ops.aten.view.default(permute_14, [1, 16, 4]);  permute_14 = None
    bmm_2 = torch.ops.aten.bmm.default(view_87, view_88);  view_87 = view_88 = None
    view_89 = torch.ops.aten.view.default(bmm_2, [5, 276, 32, 128, 1, 4]);  bmm_2 = None
    permute_15 = torch.ops.aten.permute.default(view_89, [0, 5, 1, 2, 3, 4]);  view_89 = None
    view_90 = torch.ops.aten.view.default(permute_15, [5, 4, 276, 32, 128]);  permute_15 = None
    view_91 = torch.ops.aten.view.default(view_62, [5, 1, 276, 32, 128])
    bitwise_not_3 = torch.ops.aten.bitwise_not.default(view_91);  view_91 = None
    masked_fill_3 = torch.ops.aten.masked_fill.Scalar(view_90, bitwise_not_3, -10000);  view_90 = bitwise_not_3 = None
    native_layer_norm_default_15 = torch.ops.aten.native_layer_norm.default(masked_fill_2, [128], None, None, 0.1)
    getitem_73 = native_layer_norm_default_15[0];  native_layer_norm_default_15 = None
    t_25 = torch.ops.aten.t.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_1_single_layer_norm_lin_s_merged_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_1_single_layer_norm_lin_s_merged_weight = None
    clone_8 = torch.ops.aten.clone.default(view_60, memory_format = torch.contiguous_format)
    _unsafe_view_7 = torch.ops.aten._unsafe_view.default(clone_8, [44160, 128]);  clone_8 = None
    mm_25 = torch.ops.aten.mm.default(_unsafe_view_7, t_25);  _unsafe_view_7 = t_25 = None
    view_92 = torch.ops.aten.view.default(mm_25, [5, 8832, 256]);  mm_25 = None
    split_tensor_11 = torch.ops.aten.split.Tensor(view_92, 128, dim = -1);  view_92 = None
    getitem_76 = split_tensor_11[0]
    getitem_77 = split_tensor_11[1];  split_tensor_11 = None
    add_24 = torch.ops.aten.add.Tensor(getitem_76, 1);  getitem_76 = None
    mul_13 = torch.ops.aten.mul.Tensor(getitem_73, add_24);  getitem_73 = add_24 = None
    add_25 = torch.ops.aten.add.Tensor(mul_13, getitem_77);  mul_13 = getitem_77 = None
    unsqueeze_25 = torch.ops.aten.unsqueeze.default(add_25, 3);  add_25 = None
    unsqueeze_26 = torch.ops.aten.unsqueeze.default(unsqueeze_25, 4);  unsqueeze_25 = None
    unsqueeze_27 = torch.ops.aten.unsqueeze.default(unsqueeze_26, 5);  unsqueeze_26 = None
    permute_16 = torch.ops.aten.permute.default(unsqueeze_27, [3, 0, 4, 1, 5, 2]);  unsqueeze_27 = None
    unsqueeze_28 = torch.ops.aten.unsqueeze.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_1_to_qkv_weight, 4);  atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_1_to_qkv_weight = None
    unsqueeze_29 = torch.ops.aten.unsqueeze.default(unsqueeze_28, 5);  unsqueeze_28 = None
    permute_17 = torch.ops.aten.permute.default(unsqueeze_29, [0, 4, 1, 5, 2, 3]);  unsqueeze_29 = None
    permute_18 = torch.ops.aten.permute.default(permute_16, [1, 3, 5, 0, 2, 4]);  permute_16 = None
    view_93 = torch.ops.aten.view.default(permute_18, [1, 44160, 128]);  permute_18 = None
    permute_19 = torch.ops.aten.permute.default(permute_17, [5, 0, 2, 4, 1, 3]);  permute_17 = None
    view_94 = torch.ops.aten.view.default(permute_19, [1, 128, 384]);  permute_19 = None
    bmm_3 = torch.ops.aten.bmm.default(view_93, view_94);  view_93 = view_94 = None
    view_95 = torch.ops.aten.view.default(bmm_3, [5, 8832, 1, 3, 4, 32]);  bmm_3 = None
    permute_20 = torch.ops.aten.permute.default(view_95, [3, 0, 4, 1, 5, 2]);  view_95 = None
    view_96 = torch.ops.aten.view.default(permute_20, [3, 5, 4, 8832, 32]);  permute_20 = None
    clone_9 = torch.ops.aten.clone.default(view_96, memory_format = torch.contiguous_format);  view_96 = None
    _unsafe_view_8 = torch.ops.aten._unsafe_view.default(clone_9, [3, 20, 8832, 32]);  clone_9 = None
    unbind_int_2 = torch.ops.aten.unbind.int(_unsafe_view_8);  _unsafe_view_8 = None
    getitem_78 = unbind_int_2[0]
    getitem_79 = unbind_int_2[1]
    getitem_80 = unbind_int_2[2];  unbind_int_2 = None
    unsqueeze_30 = torch.ops.aten.unsqueeze.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_1_q_bias, 0);  atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_1_q_bias = None
    expand_7 = torch.ops.aten.expand.default(unsqueeze_30, [5, -1, -1]);  unsqueeze_30 = None
    clone_10 = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    _unsafe_view_9 = torch.ops.aten._unsafe_view.default(clone_10, [20, 1, 32]);  clone_10 = None
    add_26 = torch.ops.aten.add.Tensor(getitem_78, _unsafe_view_9);  getitem_78 = _unsafe_view_9 = None
    view_97 = torch.ops.aten.view.default(add_26, [20, 276, 32, 32]);  add_26 = None
    slice_13 = torch.ops.aten.slice.Tensor(getitem_79, dim = 0, start = 0, end = 9223372036854775807);  getitem_79 = None
    slice_14 = torch.ops.aten.slice.Tensor(slice_13, dim = 2, start = 0, end = 9223372036854775807);  slice_13 = None
    index_8 = torch.ops.aten.index.Tensor(slice_14, [None, remainder]);  slice_14 = None
    slice_15 = torch.ops.aten.slice.Tensor(getitem_80, dim = 0, start = 0, end = 9223372036854775807);  getitem_80 = None
    slice_16 = torch.ops.aten.slice.Tensor(slice_15, dim = 2, start = 0, end = 9223372036854775807);  slice_15 = None
    index_9 = torch.ops.aten.index.Tensor(slice_16, [None, remainder]);  slice_16 = None
    view_98 = torch.ops.aten.view.default(masked_fill_3, [20, 276, 32, 128]);  masked_fill_3 = None
    expand_8 = torch.ops.aten.expand.default(view_98, [20, 276, 32, 128]);  view_98 = None
    _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_97, index_8, index_9, expand_8, False);  view_97 = index_8 = index_9 = expand_8 = None
    getitem_81 = _scaled_dot_product_efficient_attention_default_1[0];  _scaled_dot_product_efficient_attention_default_1 = None
    view_99 = torch.ops.aten.view.default(getitem_81, [5, 4, 276, 32, 32]);  getitem_81 = None
    permute_21 = torch.ops.aten.permute.default(view_99, [0, 2, 3, 1, 4]);  view_99 = None
    clone_11 = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
    _unsafe_view_10 = torch.ops.aten._unsafe_view.default(clone_11, [5, 8832, 128]);  clone_11 = None
    t_26 = torch.ops.aten.t.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_1_out_proj_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_1_out_proj_weight = None
    clone_12 = torch.ops.aten.clone.default(view_60, memory_format = torch.contiguous_format)
    _unsafe_view_11 = torch.ops.aten._unsafe_view.default(clone_12, [44160, 128]);  clone_12 = None
    mm_26 = torch.ops.aten.mm.default(_unsafe_view_11, t_26);  _unsafe_view_11 = t_26 = None
    view_100 = torch.ops.aten.view.default(mm_26, [5, 8832, 128]);  mm_26 = None
    add_27 = torch.ops.aten.add.Tensor(view_100, atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_1_out_proj_bias);  view_100 = atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_1_out_proj_bias = None
    sigmoid_2 = torch.ops.aten.sigmoid.default(add_27);  add_27 = None
    mul_14 = torch.ops.aten.mul.Tensor(_unsafe_view_10, sigmoid_2);  _unsafe_view_10 = sigmoid_2 = None
    native_layer_norm_default_16 = torch.ops.aten.native_layer_norm.default(masked_fill_2, [128], None, None, 0.1)
    getitem_85 = native_layer_norm_default_16[0];  native_layer_norm_default_16 = None
    t_27 = torch.ops.aten.t.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_1_ada_ln_lin_s_merged_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_1_ada_ln_lin_s_merged_weight = None
    clone_13 = torch.ops.aten.clone.default(view_60, memory_format = torch.contiguous_format)
    _unsafe_view_12 = torch.ops.aten._unsafe_view.default(clone_13, [44160, 128]);  clone_13 = None
    mm_27 = torch.ops.aten.mm.default(_unsafe_view_12, t_27);  _unsafe_view_12 = t_27 = None
    view_101 = torch.ops.aten.view.default(mm_27, [5, 8832, 256]);  mm_27 = None
    split_tensor_12 = torch.ops.aten.split.Tensor(view_101, 128, dim = -1);  view_101 = None
    getitem_88 = split_tensor_12[0]
    getitem_89 = split_tensor_12[1];  split_tensor_12 = None
    add_28 = torch.ops.aten.add.Tensor(getitem_88, 1);  getitem_88 = None
    mul_15 = torch.ops.aten.mul.Tensor(getitem_85, add_28);  getitem_85 = add_28 = None
    add_29 = torch.ops.aten.add.Tensor(mul_15, getitem_89);  mul_15 = getitem_89 = None
    t_28 = torch.ops.aten.t.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_1_linear_a_nobias_double_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_1_linear_a_nobias_double_weight = None
    view_102 = torch.ops.aten.view.default(add_29, [44160, 128]);  add_29 = None
    mm_28 = torch.ops.aten.mm.default(view_102, t_28);  view_102 = t_28 = None
    view_103 = torch.ops.aten.view.default(mm_28, [5, 8832, 512]);  mm_28 = None
    split_tensor_13 = torch.ops.aten.split.Tensor(view_103, 256, dim = -1);  view_103 = None
    getitem_90 = split_tensor_13[0]
    getitem_91 = split_tensor_13[1];  split_tensor_13 = None
    silu_5 = torch.ops.aten.silu.default(getitem_90);  getitem_90 = None
    mul_16 = torch.ops.aten.mul.Tensor(silu_5, getitem_91);  silu_5 = getitem_91 = None
    t_29 = torch.ops.aten.t.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_1_linear_s_biasinit_m2_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_1_linear_s_biasinit_m2_weight = None
    clone_14 = torch.ops.aten.clone.default(view_60, memory_format = torch.contiguous_format)
    _unsafe_view_13 = torch.ops.aten._unsafe_view.default(clone_14, [44160, 128]);  clone_14 = None
    mm_29 = torch.ops.aten.mm.default(_unsafe_view_13, t_29);  _unsafe_view_13 = t_29 = None
    view_104 = torch.ops.aten.view.default(mm_29, [5, 8832, 128]);  mm_29 = None
    add_30 = torch.ops.aten.add.Tensor(view_104, atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_1_linear_s_biasinit_m2_bias);  view_104 = atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_1_linear_s_biasinit_m2_bias = None
    sigmoid_3 = torch.ops.aten.sigmoid.default(add_30);  add_30 = None
    t_30 = torch.ops.aten.t.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_1_linear_b_nobias_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_1_linear_b_nobias_weight = None
    view_105 = torch.ops.aten.view.default(mul_16, [44160, 256]);  mul_16 = None
    mm_30 = torch.ops.aten.mm.default(view_105, t_30);  view_105 = t_30 = None
    view_106 = torch.ops.aten.view.default(mm_30, [5, 8832, 128]);  mm_30 = None
    mul_17 = torch.ops.aten.mul.Tensor(sigmoid_3, view_106);  sigmoid_3 = view_106 = None
    add_31 = torch.ops.aten.add.Tensor(masked_fill_2, mul_17);  masked_fill_2 = mul_17 = None
    add_32 = torch.ops.aten.add.Tensor(add_31, mul_14);  add_31 = mul_14 = None
    unsqueeze_31 = torch.ops.aten.unsqueeze.default(view_63, -1);  view_63 = None
    bitwise_not_4 = torch.ops.aten.bitwise_not.default(unsqueeze_31);  unsqueeze_31 = None
    masked_fill_4 = torch.ops.aten.masked_fill.Scalar(add_32, bitwise_not_4, 0.0);  add_32 = bitwise_not_4 = None
    unsqueeze_32 = torch.ops.aten.unsqueeze.default(getitem_48, 5);  getitem_48 = None
    permute_22 = torch.ops.aten.permute.default(unsqueeze_32, [0, 5, 1, 2, 3, 4]);  unsqueeze_32 = None
    unsqueeze_33 = torch.ops.aten.unsqueeze.default(getitem_53, 2);  getitem_53 = None
    unsqueeze_34 = torch.ops.aten.unsqueeze.default(unsqueeze_33, 3);  unsqueeze_33 = None
    unsqueeze_35 = torch.ops.aten.unsqueeze.default(unsqueeze_34, 4);  unsqueeze_34 = None
    unsqueeze_36 = torch.ops.aten.unsqueeze.default(unsqueeze_35, 5);  unsqueeze_35 = None
    permute_23 = torch.ops.aten.permute.default(unsqueeze_36, [2, 0, 3, 4, 5, 1]);  unsqueeze_36 = None
    permute_24 = torch.ops.aten.permute.default(permute_22, [0, 2, 3, 4, 5, 1]);  permute_22 = None
    view_107 = torch.ops.aten.view.default(permute_24, [1, 5652480, 16]);  permute_24 = None
    permute_25 = torch.ops.aten.permute.default(permute_23, [5, 1, 0, 2, 3, 4]);  permute_23 = None
    view_108 = torch.ops.aten.view.default(permute_25, [1, 16, 4]);  permute_25 = None
    bmm_4 = torch.ops.aten.bmm.default(view_107, view_108);  view_107 = view_108 = None
    view_109 = torch.ops.aten.view.default(bmm_4, [5, 276, 32, 128, 1, 4]);  bmm_4 = None
    permute_26 = torch.ops.aten.permute.default(view_109, [0, 5, 1, 2, 3, 4]);  view_109 = None
    view_110 = torch.ops.aten.view.default(permute_26, [5, 4, 276, 32, 128]);  permute_26 = None
    view_111 = torch.ops.aten.view.default(view_62, [5, 1, 276, 32, 128]);  view_62 = None
    bitwise_not_5 = torch.ops.aten.bitwise_not.default(view_111);  view_111 = None
    masked_fill_5 = torch.ops.aten.masked_fill.Scalar(view_110, bitwise_not_5, -10000);  view_110 = bitwise_not_5 = None
    native_layer_norm_default_17 = torch.ops.aten.native_layer_norm.default(masked_fill_4, [128], None, None, 0.1)
    getitem_92 = native_layer_norm_default_17[0];  native_layer_norm_default_17 = None
    t_31 = torch.ops.aten.t.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_2_single_layer_norm_lin_s_merged_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_2_single_layer_norm_lin_s_merged_weight = None
    clone_15 = torch.ops.aten.clone.default(view_60, memory_format = torch.contiguous_format)
    _unsafe_view_14 = torch.ops.aten._unsafe_view.default(clone_15, [44160, 128]);  clone_15 = None
    mm_31 = torch.ops.aten.mm.default(_unsafe_view_14, t_31);  _unsafe_view_14 = t_31 = None
    view_112 = torch.ops.aten.view.default(mm_31, [5, 8832, 256]);  mm_31 = None
    split_tensor_14 = torch.ops.aten.split.Tensor(view_112, 128, dim = -1);  view_112 = None
    getitem_95 = split_tensor_14[0]
    getitem_96 = split_tensor_14[1];  split_tensor_14 = None
    add_33 = torch.ops.aten.add.Tensor(getitem_95, 1);  getitem_95 = None
    mul_18 = torch.ops.aten.mul.Tensor(getitem_92, add_33);  getitem_92 = add_33 = None
    add_34 = torch.ops.aten.add.Tensor(mul_18, getitem_96);  mul_18 = getitem_96 = None
    unsqueeze_37 = torch.ops.aten.unsqueeze.default(add_34, 3);  add_34 = None
    unsqueeze_38 = torch.ops.aten.unsqueeze.default(unsqueeze_37, 4);  unsqueeze_37 = None
    unsqueeze_39 = torch.ops.aten.unsqueeze.default(unsqueeze_38, 5);  unsqueeze_38 = None
    permute_27 = torch.ops.aten.permute.default(unsqueeze_39, [3, 0, 4, 1, 5, 2]);  unsqueeze_39 = None
    unsqueeze_40 = torch.ops.aten.unsqueeze.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_2_to_qkv_weight, 4);  atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_2_to_qkv_weight = None
    unsqueeze_41 = torch.ops.aten.unsqueeze.default(unsqueeze_40, 5);  unsqueeze_40 = None
    permute_28 = torch.ops.aten.permute.default(unsqueeze_41, [0, 4, 1, 5, 2, 3]);  unsqueeze_41 = None
    permute_29 = torch.ops.aten.permute.default(permute_27, [1, 3, 5, 0, 2, 4]);  permute_27 = None
    view_113 = torch.ops.aten.view.default(permute_29, [1, 44160, 128]);  permute_29 = None
    permute_30 = torch.ops.aten.permute.default(permute_28, [5, 0, 2, 4, 1, 3]);  permute_28 = None
    view_114 = torch.ops.aten.view.default(permute_30, [1, 128, 384]);  permute_30 = None
    bmm_5 = torch.ops.aten.bmm.default(view_113, view_114);  view_113 = view_114 = None
    view_115 = torch.ops.aten.view.default(bmm_5, [5, 8832, 1, 3, 4, 32]);  bmm_5 = None
    permute_31 = torch.ops.aten.permute.default(view_115, [3, 0, 4, 1, 5, 2]);  view_115 = None
    view_116 = torch.ops.aten.view.default(permute_31, [3, 5, 4, 8832, 32]);  permute_31 = None
    clone_16 = torch.ops.aten.clone.default(view_116, memory_format = torch.contiguous_format);  view_116 = None
    _unsafe_view_15 = torch.ops.aten._unsafe_view.default(clone_16, [3, 20, 8832, 32]);  clone_16 = None
    unbind_int_3 = torch.ops.aten.unbind.int(_unsafe_view_15);  _unsafe_view_15 = None
    getitem_97 = unbind_int_3[0]
    getitem_98 = unbind_int_3[1]
    getitem_99 = unbind_int_3[2];  unbind_int_3 = None
    unsqueeze_42 = torch.ops.aten.unsqueeze.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_2_q_bias, 0);  atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_2_q_bias = None
    expand_9 = torch.ops.aten.expand.default(unsqueeze_42, [5, -1, -1]);  unsqueeze_42 = None
    clone_17 = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    _unsafe_view_16 = torch.ops.aten._unsafe_view.default(clone_17, [20, 1, 32]);  clone_17 = None
    add_35 = torch.ops.aten.add.Tensor(getitem_97, _unsafe_view_16);  getitem_97 = _unsafe_view_16 = None
    view_117 = torch.ops.aten.view.default(add_35, [20, 276, 32, 32]);  add_35 = None
    slice_17 = torch.ops.aten.slice.Tensor(getitem_98, dim = 0, start = 0, end = 9223372036854775807);  getitem_98 = None
    slice_18 = torch.ops.aten.slice.Tensor(slice_17, dim = 2, start = 0, end = 9223372036854775807);  slice_17 = None
    index_10 = torch.ops.aten.index.Tensor(slice_18, [None, remainder]);  slice_18 = None
    slice_19 = torch.ops.aten.slice.Tensor(getitem_99, dim = 0, start = 0, end = 9223372036854775807);  getitem_99 = None
    slice_20 = torch.ops.aten.slice.Tensor(slice_19, dim = 2, start = 0, end = 9223372036854775807);  slice_19 = None
    index_11 = torch.ops.aten.index.Tensor(slice_20, [None, remainder]);  slice_20 = remainder = None
    view_118 = torch.ops.aten.view.default(masked_fill_5, [20, 276, 32, 128]);  masked_fill_5 = None
    expand_10 = torch.ops.aten.expand.default(view_118, [20, 276, 32, 128]);  view_118 = None
    _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_117, index_10, index_11, expand_10, False);  view_117 = index_10 = index_11 = expand_10 = None
    getitem_100 = _scaled_dot_product_efficient_attention_default_2[0];  _scaled_dot_product_efficient_attention_default_2 = None
    view_119 = torch.ops.aten.view.default(getitem_100, [5, 4, 276, 32, 32]);  getitem_100 = None
    permute_32 = torch.ops.aten.permute.default(view_119, [0, 2, 3, 1, 4]);  view_119 = None
    clone_18 = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
    _unsafe_view_17 = torch.ops.aten._unsafe_view.default(clone_18, [5, 8832, 128]);  clone_18 = None
    t_32 = torch.ops.aten.t.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_2_out_proj_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_2_out_proj_weight = None
    clone_19 = torch.ops.aten.clone.default(view_60, memory_format = torch.contiguous_format)
    _unsafe_view_18 = torch.ops.aten._unsafe_view.default(clone_19, [44160, 128]);  clone_19 = None
    mm_32 = torch.ops.aten.mm.default(_unsafe_view_18, t_32);  _unsafe_view_18 = t_32 = None
    view_120 = torch.ops.aten.view.default(mm_32, [5, 8832, 128]);  mm_32 = None
    add_36 = torch.ops.aten.add.Tensor(view_120, atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_2_out_proj_bias);  view_120 = atom_attention_encoder_atom_transformer_local_diffn_transformer_local_attentions_2_out_proj_bias = None
    sigmoid_4 = torch.ops.aten.sigmoid.default(add_36);  add_36 = None
    mul_19 = torch.ops.aten.mul.Tensor(_unsafe_view_17, sigmoid_4);  _unsafe_view_17 = sigmoid_4 = None
    native_layer_norm_default_18 = torch.ops.aten.native_layer_norm.default(masked_fill_4, [128], None, None, 0.1)
    getitem_104 = native_layer_norm_default_18[0];  native_layer_norm_default_18 = None
    t_33 = torch.ops.aten.t.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_2_ada_ln_lin_s_merged_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_2_ada_ln_lin_s_merged_weight = None
    clone_20 = torch.ops.aten.clone.default(view_60, memory_format = torch.contiguous_format)
    _unsafe_view_19 = torch.ops.aten._unsafe_view.default(clone_20, [44160, 128]);  clone_20 = None
    mm_33 = torch.ops.aten.mm.default(_unsafe_view_19, t_33);  _unsafe_view_19 = t_33 = None
    view_121 = torch.ops.aten.view.default(mm_33, [5, 8832, 256]);  mm_33 = None
    split_tensor_15 = torch.ops.aten.split.Tensor(view_121, 128, dim = -1);  view_121 = None
    getitem_107 = split_tensor_15[0]
    getitem_108 = split_tensor_15[1];  split_tensor_15 = None
    add_37 = torch.ops.aten.add.Tensor(getitem_107, 1);  getitem_107 = None
    mul_20 = torch.ops.aten.mul.Tensor(getitem_104, add_37);  getitem_104 = add_37 = None
    add_38 = torch.ops.aten.add.Tensor(mul_20, getitem_108);  mul_20 = getitem_108 = None
    t_34 = torch.ops.aten.t.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_2_linear_a_nobias_double_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_2_linear_a_nobias_double_weight = None
    view_122 = torch.ops.aten.view.default(add_38, [44160, 128]);  add_38 = None
    mm_34 = torch.ops.aten.mm.default(view_122, t_34);  view_122 = t_34 = None
    view_123 = torch.ops.aten.view.default(mm_34, [5, 8832, 512]);  mm_34 = None
    split_tensor_16 = torch.ops.aten.split.Tensor(view_123, 256, dim = -1);  view_123 = None
    getitem_109 = split_tensor_16[0]
    getitem_110 = split_tensor_16[1];  split_tensor_16 = None
    silu_6 = torch.ops.aten.silu.default(getitem_109);  getitem_109 = None
    mul_21 = torch.ops.aten.mul.Tensor(silu_6, getitem_110);  silu_6 = getitem_110 = None
    t_35 = torch.ops.aten.t.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_2_linear_s_biasinit_m2_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_2_linear_s_biasinit_m2_weight = None
    clone_21 = torch.ops.aten.clone.default(view_60, memory_format = torch.contiguous_format);  view_60 = None
    _unsafe_view_20 = torch.ops.aten._unsafe_view.default(clone_21, [44160, 128]);  clone_21 = None
    mm_35 = torch.ops.aten.mm.default(_unsafe_view_20, t_35);  _unsafe_view_20 = t_35 = None
    view_124 = torch.ops.aten.view.default(mm_35, [5, 8832, 128]);  mm_35 = None
    add_39 = torch.ops.aten.add.Tensor(view_124, atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_2_linear_s_biasinit_m2_bias);  view_124 = atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_2_linear_s_biasinit_m2_bias = None
    sigmoid_5 = torch.ops.aten.sigmoid.default(add_39);  add_39 = None
    t_36 = torch.ops.aten.t.default(atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_2_linear_b_nobias_weight);  atom_attention_encoder_atom_transformer_local_diffn_transformer_transitions_2_linear_b_nobias_weight = None
    view_125 = torch.ops.aten.view.default(mul_21, [44160, 256]);  mul_21 = None
    mm_36 = torch.ops.aten.mm.default(view_125, t_36);  view_125 = t_36 = None
    view_126 = torch.ops.aten.view.default(mm_36, [5, 8832, 128]);  mm_36 = None
    mul_22 = torch.ops.aten.mul.Tensor(sigmoid_5, view_126);  sigmoid_5 = view_126 = None
    add_40 = torch.ops.aten.add.Tensor(masked_fill_4, mul_22);  masked_fill_4 = mul_22 = None
    add_41 = torch.ops.aten.add.Tensor(add_40, mul_19);  add_40 = mul_19 = None
    view_127 = torch.ops.aten.view.default(add_41, [1, 5, 8832, 128]);  add_41 = None
    t_37 = torch.ops.aten.t.default(atom_attention_encoder_to_token_single_0_weight);  atom_attention_encoder_to_token_single_0_weight = None
    view_128 = torch.ops.aten.view.default(view_127, [44160, 128])
    mm_37 = torch.ops.aten.mm.default(view_128, t_37);  view_128 = t_37 = None
    view_129 = torch.ops.aten.view.default(mm_37, [1, 5, 8832, 768]);  mm_37 = None
    relu_3 = torch.ops.aten.relu.default(view_129);  view_129 = None
    view_130 = torch.ops.aten.view.default(relu_3, [44160, 768]);  relu_3 = None
    unsqueeze_43 = torch.ops.aten.unsqueeze.default(arg349_1, 1)
    expand_11 = torch.ops.aten.expand.default(unsqueeze_43, [-1, 5, -1]);  unsqueeze_43 = None
    view_133 = torch.ops.aten.view.default(expand_11, [5, 8832]);  expand_11 = None
    unsqueeze_44 = torch.ops.aten.unsqueeze.default(arg356_1, 1)
    expand_12 = torch.ops.aten.expand.default(unsqueeze_44, [-1, 5, -1]);  unsqueeze_44 = None
    view_134 = torch.ops.aten.view.default(expand_12, [5, 8832]);  expand_12 = None
    view_135 = torch.ops.aten.view.default(view_130, [1, 5, 8832, 768]);  view_130 = None
    view_136 = torch.ops.aten.view.default(view_135, [5, 8832, 768]);  view_135 = None
    new_zeros = torch.ops.aten.new_zeros.default(view_136, [5, 384, 768], pin_memory = False)
    new_zeros_1 = torch.ops.aten.new_zeros.default(view_136, [5, 384], pin_memory = False)
    unsqueeze_45 = torch.ops.aten.unsqueeze.default(view_134, 2)
    expand_13 = torch.ops.aten.expand.default(unsqueeze_45, [-1, -1, 768]);  unsqueeze_45 = None
    unsqueeze_46 = torch.ops.aten.unsqueeze.default(view_133, -1)
    mul_23 = torch.ops.aten.mul.Tensor(view_136, unsqueeze_46);  view_136 = unsqueeze_46 = None
    scatter_reduce = torch.ops.aten.scatter_reduce.two(new_zeros, 1, expand_13, mul_23, 'sum');  new_zeros = expand_13 = mul_23 = None
    _to_copy = torch.ops.aten._to_copy.default(view_133, dtype = torch.float32);  view_133 = None
    scatter_reduce_1 = torch.ops.aten.scatter_reduce.two(new_zeros_1, 1, view_134, _to_copy, 'sum');  new_zeros_1 = view_134 = _to_copy = None
    unsqueeze_47 = torch.ops.aten.unsqueeze.default(scatter_reduce_1, -1);  scatter_reduce_1 = None
    clamp = torch.ops.aten.clamp.default(unsqueeze_47, min = 1);  unsqueeze_47 = None
    div = torch.ops.aten.div.Tensor(scatter_reduce, clamp);  scatter_reduce = clamp = None
    view_137 = torch.ops.aten.view.default(div, [1, 5, 384, 768]);  div = None
    t_38 = torch.ops.aten.t.default(structure_cond_to_token_structure_proj_weight);  structure_cond_to_token_structure_proj_weight = None
    view_138 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_38 = torch.ops.aten.mm.default(view_138, t_38);  view_138 = t_38 = None
    view_139 = torch.ops.aten.view.default(mm_38, [1, 5, 384, 768]);  mm_38 = None
    add_42 = torch.ops.aten.add.Tensor(view_137, view_139);  view_137 = view_139 = None
    view_140 = torch.ops.aten.view.default(getitem_36, [1, 1, 384, 384, 256]);  getitem_36 = None
    view_141 = torch.ops.aten.view.default(arg351_1, [1, 1, 384, 1])
    view_142 = torch.ops.aten.view.default(arg351_1, [1, 1, 1, 384]);  arg351_1 = None
    bitwise_and_1 = torch.ops.aten.bitwise_and.Tensor(view_141, view_142);  view_141 = view_142 = None
    native_layer_norm_default_19 = torch.ops.aten.native_layer_norm.default(view_140, [256], diffusion_transformer_blocks_0_pair_layer_norm_weight, diffusion_transformer_blocks_0_pair_layer_norm_bias, 1e-05);  diffusion_transformer_blocks_0_pair_layer_norm_weight = diffusion_transformer_blocks_0_pair_layer_norm_bias = None
    getitem_111 = native_layer_norm_default_19[0];  native_layer_norm_default_19 = None
    t_39 = torch.ops.aten.t.default(diffusion_transformer_blocks_0_pair_linear_weight);  diffusion_transformer_blocks_0_pair_linear_weight = None
    view_144 = torch.ops.aten.view.default(getitem_111, [147456, 256]);  getitem_111 = None
    mm_39 = torch.ops.aten.mm.default(view_144, t_39);  view_144 = t_39 = None
    view_145 = torch.ops.aten.view.default(mm_39, [1, 1, 384, 384, 16]);  mm_39 = None
    native_layer_norm_default_20 = torch.ops.aten.native_layer_norm.default(add_42, [768], None, None, 0.1)
    getitem_114 = native_layer_norm_default_20[0];  native_layer_norm_default_20 = None
    t_40 = torch.ops.aten.t.default(diffusion_transformer_blocks_0_norm_in_lin_s_merged_weight);  diffusion_transformer_blocks_0_norm_in_lin_s_merged_weight = None
    view_146 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_40 = torch.ops.aten.mm.default(view_146, t_40);  view_146 = t_40 = None
    view_147 = torch.ops.aten.view.default(mm_40, [1, 5, 384, 1536]);  mm_40 = None
    split_tensor_17 = torch.ops.aten.split.Tensor(view_147, 768, dim = -1);  view_147 = None
    getitem_117 = split_tensor_17[0]
    getitem_118 = split_tensor_17[1];  split_tensor_17 = None
    add_43 = torch.ops.aten.add.Tensor(getitem_117, 1);  getitem_117 = None
    mul_24 = torch.ops.aten.mul.Tensor(getitem_114, add_43);  getitem_114 = add_43 = None
    add_44 = torch.ops.aten.add.Tensor(mul_24, getitem_118);  mul_24 = getitem_118 = None
    t_41 = torch.ops.aten.t.default(diffusion_transformer_blocks_0_to_qkv_weight);  diffusion_transformer_blocks_0_to_qkv_weight = None
    view_148 = torch.ops.aten.view.default(add_44, [1920, 768]);  add_44 = None
    mm_41 = torch.ops.aten.mm.default(view_148, t_41);  view_148 = t_41 = None
    view_149 = torch.ops.aten.view.default(mm_41, [1, 5, 384, 2304]);  mm_41 = None
    view_150 = torch.ops.aten.view.default(view_149, [1, 5, 384, 16, 144]);  view_149 = None
    permute_33 = torch.ops.aten.permute.default(view_150, [0, 3, 1, 2, 4]);  view_150 = None
    split_tensor_18 = torch.ops.aten.split.Tensor(permute_33, 48, dim = -1);  permute_33 = None
    getitem_119 = split_tensor_18[0]
    getitem_120 = split_tensor_18[1]
    getitem_121 = split_tensor_18[2];  split_tensor_18 = None
    view_151 = torch.ops.aten.view.default(diffusion_transformer_blocks_0_q_bias, [1, 16, 1, 1, 48]);  diffusion_transformer_blocks_0_q_bias = None
    add_45 = torch.ops.aten.add.Tensor(getitem_119, view_151);  getitem_119 = view_151 = None
    view_152 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 384, 384, 1])
    bitwise_not_6 = torch.ops.aten.bitwise_not.default(view_152);  view_152 = None
    masked_fill_6 = torch.ops.aten.masked_fill.Scalar(view_145, bitwise_not_6, -10000);  view_145 = bitwise_not_6 = None
    permute_34 = torch.ops.aten.permute.default(masked_fill_6, [0, 4, 1, 2, 3]);  masked_fill_6 = None
    view_153 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 1, 384, 384])
    bitwise_not_7 = torch.ops.aten.bitwise_not.default(view_153);  view_153 = None
    masked_fill_7 = torch.ops.aten.masked_fill.Scalar(permute_34, bitwise_not_7, -10000);  permute_34 = bitwise_not_7 = None
    mul_25 = torch.ops.aten.mul.Scalar(add_45, 0.3799178428257963);  add_45 = None
    transpose = torch.ops.aten.transpose.int(getitem_120, -2, -1);  getitem_120 = None
    mul_26 = torch.ops.aten.mul.Scalar(transpose, 0.3799178428257963);  transpose = None
    expand_14 = torch.ops.aten.expand.default(mul_25, [1, 16, 5, 384, 48]);  mul_25 = None
    clone_22 = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    _unsafe_view_21 = torch.ops.aten._unsafe_view.default(clone_22, [80, 384, 48]);  clone_22 = None
    expand_15 = torch.ops.aten.expand.default(mul_26, [1, 16, 5, 48, 384]);  mul_26 = None
    clone_23 = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    _unsafe_view_22 = torch.ops.aten._unsafe_view.default(clone_23, [80, 48, 384]);  clone_23 = None
    bmm_6 = torch.ops.aten.bmm.default(_unsafe_view_21, _unsafe_view_22);  _unsafe_view_21 = _unsafe_view_22 = None
    view_154 = torch.ops.aten.view.default(bmm_6, [1, 16, 5, 384, 384]);  bmm_6 = None
    add_46 = torch.ops.aten.add.Tensor(view_154, masked_fill_7);  view_154 = masked_fill_7 = None
    _softmax = torch.ops.aten._softmax.default(add_46, -1, False);  add_46 = None
    expand_16 = torch.ops.aten.expand.default(_softmax, [1, 16, 5, 384, 384]);  _softmax = None
    view_155 = torch.ops.aten.view.default(expand_16, [80, 384, 384]);  expand_16 = None
    expand_17 = torch.ops.aten.expand.default(getitem_121, [1, 16, 5, 384, 48]);  getitem_121 = None
    clone_24 = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    _unsafe_view_23 = torch.ops.aten._unsafe_view.default(clone_24, [80, 384, 48]);  clone_24 = None
    bmm_7 = torch.ops.aten.bmm.default(view_155, _unsafe_view_23);  view_155 = _unsafe_view_23 = None
    view_156 = torch.ops.aten.view.default(bmm_7, [1, 16, 5, 384, 48]);  bmm_7 = None
    permute_35 = torch.ops.aten.permute.default(view_156, [0, 2, 3, 1, 4]);  view_156 = None
    clone_25 = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    _unsafe_view_24 = torch.ops.aten._unsafe_view.default(clone_25, [1, 5, 384, 768]);  clone_25 = None
    t_42 = torch.ops.aten.t.default(diffusion_transformer_blocks_0_to_out_weight);  diffusion_transformer_blocks_0_to_out_weight = None
    view_157 = torch.ops.aten.view.default(_unsafe_view_24, [1920, 768]);  _unsafe_view_24 = None
    mm_42 = torch.ops.aten.mm.default(view_157, t_42);  view_157 = t_42 = None
    view_158 = torch.ops.aten.view.default(mm_42, [1, 5, 384, 768]);  mm_42 = None
    view_159 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_43 = torch.ops.aten.t.default(diffusion_transformer_blocks_0_gate_proj_0_weight);  diffusion_transformer_blocks_0_gate_proj_0_weight = None
    addmm = torch.ops.aten.addmm.default(diffusion_transformer_blocks_0_gate_proj_0_bias, view_159, t_43);  diffusion_transformer_blocks_0_gate_proj_0_bias = view_159 = t_43 = None
    view_160 = torch.ops.aten.view.default(addmm, [1, 5, 384, 768]);  addmm = None
    sigmoid_6 = torch.ops.aten.sigmoid.default(view_160);  view_160 = None
    mul_27 = torch.ops.aten.mul.Tensor(sigmoid_6, view_158);  sigmoid_6 = view_158 = None
    native_layer_norm_default_21 = torch.ops.aten.native_layer_norm.default(add_42, [768], None, None, 0.1)
    getitem_122 = native_layer_norm_default_21[0];  native_layer_norm_default_21 = None
    t_44 = torch.ops.aten.t.default(diffusion_transformer_blocks_0_transition_ada_ln_lin_s_merged_weight);  diffusion_transformer_blocks_0_transition_ada_ln_lin_s_merged_weight = None
    view_161 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_43 = torch.ops.aten.mm.default(view_161, t_44);  view_161 = t_44 = None
    view_162 = torch.ops.aten.view.default(mm_43, [1, 5, 384, 1536]);  mm_43 = None
    split_tensor_19 = torch.ops.aten.split.Tensor(view_162, 768, dim = -1);  view_162 = None
    getitem_125 = split_tensor_19[0]
    getitem_126 = split_tensor_19[1];  split_tensor_19 = None
    add_47 = torch.ops.aten.add.Tensor(getitem_125, 1);  getitem_125 = None
    mul_28 = torch.ops.aten.mul.Tensor(getitem_122, add_47);  getitem_122 = add_47 = None
    add_48 = torch.ops.aten.add.Tensor(mul_28, getitem_126);  mul_28 = getitem_126 = None
    t_45 = torch.ops.aten.t.default(diffusion_transformer_blocks_0_transition_linear_a_nobias_double_weight);  diffusion_transformer_blocks_0_transition_linear_a_nobias_double_weight = None
    view_163 = torch.ops.aten.view.default(add_48, [1920, 768]);  add_48 = None
    mm_44 = torch.ops.aten.mm.default(view_163, t_45);  view_163 = t_45 = None
    view_164 = torch.ops.aten.view.default(mm_44, [1, 5, 384, 3072]);  mm_44 = None
    split_tensor_20 = torch.ops.aten.split.Tensor(view_164, 1536, dim = -1);  view_164 = None
    getitem_127 = split_tensor_20[0]
    getitem_128 = split_tensor_20[1];  split_tensor_20 = None
    silu_7 = torch.ops.aten.silu.default(getitem_127);  getitem_127 = None
    mul_29 = torch.ops.aten.mul.Tensor(silu_7, getitem_128);  silu_7 = getitem_128 = None
    view_165 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_46 = torch.ops.aten.t.default(diffusion_transformer_blocks_0_transition_linear_s_biasinit_m2_weight);  diffusion_transformer_blocks_0_transition_linear_s_biasinit_m2_weight = None
    addmm_1 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_0_transition_linear_s_biasinit_m2_bias, view_165, t_46);  diffusion_transformer_blocks_0_transition_linear_s_biasinit_m2_bias = view_165 = t_46 = None
    view_166 = torch.ops.aten.view.default(addmm_1, [1, 5, 384, 768]);  addmm_1 = None
    sigmoid_7 = torch.ops.aten.sigmoid.default(view_166);  view_166 = None
    t_47 = torch.ops.aten.t.default(diffusion_transformer_blocks_0_transition_linear_b_nobias_weight);  diffusion_transformer_blocks_0_transition_linear_b_nobias_weight = None
    view_167 = torch.ops.aten.view.default(mul_29, [1920, 1536]);  mul_29 = None
    mm_45 = torch.ops.aten.mm.default(view_167, t_47);  view_167 = t_47 = None
    view_168 = torch.ops.aten.view.default(mm_45, [1, 5, 384, 768]);  mm_45 = None
    mul_30 = torch.ops.aten.mul.Tensor(sigmoid_7, view_168);  sigmoid_7 = view_168 = None
    add_49 = torch.ops.aten.add.Tensor(mul_27, mul_30);  mul_27 = mul_30 = None
    add_50 = torch.ops.aten.add.Tensor(add_42, add_49);  add_42 = add_49 = None
    native_layer_norm_default_22 = torch.ops.aten.native_layer_norm.default(view_140, [256], diffusion_transformer_blocks_1_pair_layer_norm_weight, diffusion_transformer_blocks_1_pair_layer_norm_bias, 1e-05);  diffusion_transformer_blocks_1_pair_layer_norm_weight = diffusion_transformer_blocks_1_pair_layer_norm_bias = None
    getitem_129 = native_layer_norm_default_22[0];  native_layer_norm_default_22 = None
    t_48 = torch.ops.aten.t.default(diffusion_transformer_blocks_1_pair_linear_weight);  diffusion_transformer_blocks_1_pair_linear_weight = None
    view_169 = torch.ops.aten.view.default(getitem_129, [147456, 256]);  getitem_129 = None
    mm_46 = torch.ops.aten.mm.default(view_169, t_48);  view_169 = t_48 = None
    view_170 = torch.ops.aten.view.default(mm_46, [1, 1, 384, 384, 16]);  mm_46 = None
    native_layer_norm_default_23 = torch.ops.aten.native_layer_norm.default(add_50, [768], None, None, 0.1)
    getitem_132 = native_layer_norm_default_23[0];  native_layer_norm_default_23 = None
    t_49 = torch.ops.aten.t.default(diffusion_transformer_blocks_1_norm_in_lin_s_merged_weight);  diffusion_transformer_blocks_1_norm_in_lin_s_merged_weight = None
    view_171 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_47 = torch.ops.aten.mm.default(view_171, t_49);  view_171 = t_49 = None
    view_172 = torch.ops.aten.view.default(mm_47, [1, 5, 384, 1536]);  mm_47 = None
    split_tensor_21 = torch.ops.aten.split.Tensor(view_172, 768, dim = -1);  view_172 = None
    getitem_135 = split_tensor_21[0]
    getitem_136 = split_tensor_21[1];  split_tensor_21 = None
    add_51 = torch.ops.aten.add.Tensor(getitem_135, 1);  getitem_135 = None
    mul_31 = torch.ops.aten.mul.Tensor(getitem_132, add_51);  getitem_132 = add_51 = None
    add_52 = torch.ops.aten.add.Tensor(mul_31, getitem_136);  mul_31 = getitem_136 = None
    t_50 = torch.ops.aten.t.default(diffusion_transformer_blocks_1_to_qkv_weight);  diffusion_transformer_blocks_1_to_qkv_weight = None
    view_173 = torch.ops.aten.view.default(add_52, [1920, 768]);  add_52 = None
    mm_48 = torch.ops.aten.mm.default(view_173, t_50);  view_173 = t_50 = None
    view_174 = torch.ops.aten.view.default(mm_48, [1, 5, 384, 2304]);  mm_48 = None
    view_175 = torch.ops.aten.view.default(view_174, [1, 5, 384, 16, 144]);  view_174 = None
    permute_36 = torch.ops.aten.permute.default(view_175, [0, 3, 1, 2, 4]);  view_175 = None
    split_tensor_22 = torch.ops.aten.split.Tensor(permute_36, 48, dim = -1);  permute_36 = None
    getitem_137 = split_tensor_22[0]
    getitem_138 = split_tensor_22[1]
    getitem_139 = split_tensor_22[2];  split_tensor_22 = None
    view_176 = torch.ops.aten.view.default(diffusion_transformer_blocks_1_q_bias, [1, 16, 1, 1, 48]);  diffusion_transformer_blocks_1_q_bias = None
    add_53 = torch.ops.aten.add.Tensor(getitem_137, view_176);  getitem_137 = view_176 = None
    view_177 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 384, 384, 1])
    bitwise_not_8 = torch.ops.aten.bitwise_not.default(view_177);  view_177 = None
    masked_fill_8 = torch.ops.aten.masked_fill.Scalar(view_170, bitwise_not_8, -10000);  view_170 = bitwise_not_8 = None
    permute_37 = torch.ops.aten.permute.default(masked_fill_8, [0, 4, 1, 2, 3]);  masked_fill_8 = None
    view_178 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 1, 384, 384])
    bitwise_not_9 = torch.ops.aten.bitwise_not.default(view_178);  view_178 = None
    masked_fill_9 = torch.ops.aten.masked_fill.Scalar(permute_37, bitwise_not_9, -10000);  permute_37 = bitwise_not_9 = None
    mul_32 = torch.ops.aten.mul.Scalar(add_53, 0.3799178428257963);  add_53 = None
    transpose_1 = torch.ops.aten.transpose.int(getitem_138, -2, -1);  getitem_138 = None
    mul_33 = torch.ops.aten.mul.Scalar(transpose_1, 0.3799178428257963);  transpose_1 = None
    expand_18 = torch.ops.aten.expand.default(mul_32, [1, 16, 5, 384, 48]);  mul_32 = None
    clone_26 = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
    _unsafe_view_25 = torch.ops.aten._unsafe_view.default(clone_26, [80, 384, 48]);  clone_26 = None
    expand_19 = torch.ops.aten.expand.default(mul_33, [1, 16, 5, 48, 384]);  mul_33 = None
    clone_27 = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    _unsafe_view_26 = torch.ops.aten._unsafe_view.default(clone_27, [80, 48, 384]);  clone_27 = None
    bmm_8 = torch.ops.aten.bmm.default(_unsafe_view_25, _unsafe_view_26);  _unsafe_view_25 = _unsafe_view_26 = None
    view_179 = torch.ops.aten.view.default(bmm_8, [1, 16, 5, 384, 384]);  bmm_8 = None
    add_54 = torch.ops.aten.add.Tensor(view_179, masked_fill_9);  view_179 = masked_fill_9 = None
    _softmax_1 = torch.ops.aten._softmax.default(add_54, -1, False);  add_54 = None
    expand_20 = torch.ops.aten.expand.default(_softmax_1, [1, 16, 5, 384, 384]);  _softmax_1 = None
    view_180 = torch.ops.aten.view.default(expand_20, [80, 384, 384]);  expand_20 = None
    expand_21 = torch.ops.aten.expand.default(getitem_139, [1, 16, 5, 384, 48]);  getitem_139 = None
    clone_28 = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    _unsafe_view_27 = torch.ops.aten._unsafe_view.default(clone_28, [80, 384, 48]);  clone_28 = None
    bmm_9 = torch.ops.aten.bmm.default(view_180, _unsafe_view_27);  view_180 = _unsafe_view_27 = None
    view_181 = torch.ops.aten.view.default(bmm_9, [1, 16, 5, 384, 48]);  bmm_9 = None
    permute_38 = torch.ops.aten.permute.default(view_181, [0, 2, 3, 1, 4]);  view_181 = None
    clone_29 = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    _unsafe_view_28 = torch.ops.aten._unsafe_view.default(clone_29, [1, 5, 384, 768]);  clone_29 = None
    t_51 = torch.ops.aten.t.default(diffusion_transformer_blocks_1_to_out_weight);  diffusion_transformer_blocks_1_to_out_weight = None
    view_182 = torch.ops.aten.view.default(_unsafe_view_28, [1920, 768]);  _unsafe_view_28 = None
    mm_49 = torch.ops.aten.mm.default(view_182, t_51);  view_182 = t_51 = None
    view_183 = torch.ops.aten.view.default(mm_49, [1, 5, 384, 768]);  mm_49 = None
    view_184 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_52 = torch.ops.aten.t.default(diffusion_transformer_blocks_1_gate_proj_0_weight);  diffusion_transformer_blocks_1_gate_proj_0_weight = None
    addmm_2 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_1_gate_proj_0_bias, view_184, t_52);  diffusion_transformer_blocks_1_gate_proj_0_bias = view_184 = t_52 = None
    view_185 = torch.ops.aten.view.default(addmm_2, [1, 5, 384, 768]);  addmm_2 = None
    sigmoid_8 = torch.ops.aten.sigmoid.default(view_185);  view_185 = None
    mul_34 = torch.ops.aten.mul.Tensor(sigmoid_8, view_183);  sigmoid_8 = view_183 = None
    native_layer_norm_default_24 = torch.ops.aten.native_layer_norm.default(add_50, [768], None, None, 0.1)
    getitem_140 = native_layer_norm_default_24[0];  native_layer_norm_default_24 = None
    t_53 = torch.ops.aten.t.default(diffusion_transformer_blocks_1_transition_ada_ln_lin_s_merged_weight);  diffusion_transformer_blocks_1_transition_ada_ln_lin_s_merged_weight = None
    view_186 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_50 = torch.ops.aten.mm.default(view_186, t_53);  view_186 = t_53 = None
    view_187 = torch.ops.aten.view.default(mm_50, [1, 5, 384, 1536]);  mm_50 = None
    split_tensor_23 = torch.ops.aten.split.Tensor(view_187, 768, dim = -1);  view_187 = None
    getitem_143 = split_tensor_23[0]
    getitem_144 = split_tensor_23[1];  split_tensor_23 = None
    add_55 = torch.ops.aten.add.Tensor(getitem_143, 1);  getitem_143 = None
    mul_35 = torch.ops.aten.mul.Tensor(getitem_140, add_55);  getitem_140 = add_55 = None
    add_56 = torch.ops.aten.add.Tensor(mul_35, getitem_144);  mul_35 = getitem_144 = None
    t_54 = torch.ops.aten.t.default(diffusion_transformer_blocks_1_transition_linear_a_nobias_double_weight);  diffusion_transformer_blocks_1_transition_linear_a_nobias_double_weight = None
    view_188 = torch.ops.aten.view.default(add_56, [1920, 768]);  add_56 = None
    mm_51 = torch.ops.aten.mm.default(view_188, t_54);  view_188 = t_54 = None
    view_189 = torch.ops.aten.view.default(mm_51, [1, 5, 384, 3072]);  mm_51 = None
    split_tensor_24 = torch.ops.aten.split.Tensor(view_189, 1536, dim = -1);  view_189 = None
    getitem_145 = split_tensor_24[0]
    getitem_146 = split_tensor_24[1];  split_tensor_24 = None
    silu_8 = torch.ops.aten.silu.default(getitem_145);  getitem_145 = None
    mul_36 = torch.ops.aten.mul.Tensor(silu_8, getitem_146);  silu_8 = getitem_146 = None
    view_190 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_55 = torch.ops.aten.t.default(diffusion_transformer_blocks_1_transition_linear_s_biasinit_m2_weight);  diffusion_transformer_blocks_1_transition_linear_s_biasinit_m2_weight = None
    addmm_3 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_1_transition_linear_s_biasinit_m2_bias, view_190, t_55);  diffusion_transformer_blocks_1_transition_linear_s_biasinit_m2_bias = view_190 = t_55 = None
    view_191 = torch.ops.aten.view.default(addmm_3, [1, 5, 384, 768]);  addmm_3 = None
    sigmoid_9 = torch.ops.aten.sigmoid.default(view_191);  view_191 = None
    t_56 = torch.ops.aten.t.default(diffusion_transformer_blocks_1_transition_linear_b_nobias_weight);  diffusion_transformer_blocks_1_transition_linear_b_nobias_weight = None
    view_192 = torch.ops.aten.view.default(mul_36, [1920, 1536]);  mul_36 = None
    mm_52 = torch.ops.aten.mm.default(view_192, t_56);  view_192 = t_56 = None
    view_193 = torch.ops.aten.view.default(mm_52, [1, 5, 384, 768]);  mm_52 = None
    mul_37 = torch.ops.aten.mul.Tensor(sigmoid_9, view_193);  sigmoid_9 = view_193 = None
    add_57 = torch.ops.aten.add.Tensor(mul_34, mul_37);  mul_34 = mul_37 = None
    add_58 = torch.ops.aten.add.Tensor(add_50, add_57);  add_50 = add_57 = None
    native_layer_norm_default_25 = torch.ops.aten.native_layer_norm.default(view_140, [256], diffusion_transformer_blocks_2_pair_layer_norm_weight, diffusion_transformer_blocks_2_pair_layer_norm_bias, 1e-05);  diffusion_transformer_blocks_2_pair_layer_norm_weight = diffusion_transformer_blocks_2_pair_layer_norm_bias = None
    getitem_147 = native_layer_norm_default_25[0];  native_layer_norm_default_25 = None
    t_57 = torch.ops.aten.t.default(diffusion_transformer_blocks_2_pair_linear_weight);  diffusion_transformer_blocks_2_pair_linear_weight = None
    view_194 = torch.ops.aten.view.default(getitem_147, [147456, 256]);  getitem_147 = None
    mm_53 = torch.ops.aten.mm.default(view_194, t_57);  view_194 = t_57 = None
    view_195 = torch.ops.aten.view.default(mm_53, [1, 1, 384, 384, 16]);  mm_53 = None
    native_layer_norm_default_26 = torch.ops.aten.native_layer_norm.default(add_58, [768], None, None, 0.1)
    getitem_150 = native_layer_norm_default_26[0];  native_layer_norm_default_26 = None
    t_58 = torch.ops.aten.t.default(diffusion_transformer_blocks_2_norm_in_lin_s_merged_weight);  diffusion_transformer_blocks_2_norm_in_lin_s_merged_weight = None
    view_196 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_54 = torch.ops.aten.mm.default(view_196, t_58);  view_196 = t_58 = None
    view_197 = torch.ops.aten.view.default(mm_54, [1, 5, 384, 1536]);  mm_54 = None
    split_tensor_25 = torch.ops.aten.split.Tensor(view_197, 768, dim = -1);  view_197 = None
    getitem_153 = split_tensor_25[0]
    getitem_154 = split_tensor_25[1];  split_tensor_25 = None
    add_59 = torch.ops.aten.add.Tensor(getitem_153, 1);  getitem_153 = None
    mul_38 = torch.ops.aten.mul.Tensor(getitem_150, add_59);  getitem_150 = add_59 = None
    add_60 = torch.ops.aten.add.Tensor(mul_38, getitem_154);  mul_38 = getitem_154 = None
    t_59 = torch.ops.aten.t.default(diffusion_transformer_blocks_2_to_qkv_weight);  diffusion_transformer_blocks_2_to_qkv_weight = None
    view_198 = torch.ops.aten.view.default(add_60, [1920, 768]);  add_60 = None
    mm_55 = torch.ops.aten.mm.default(view_198, t_59);  view_198 = t_59 = None
    view_199 = torch.ops.aten.view.default(mm_55, [1, 5, 384, 2304]);  mm_55 = None
    view_200 = torch.ops.aten.view.default(view_199, [1, 5, 384, 16, 144]);  view_199 = None
    permute_39 = torch.ops.aten.permute.default(view_200, [0, 3, 1, 2, 4]);  view_200 = None
    split_tensor_26 = torch.ops.aten.split.Tensor(permute_39, 48, dim = -1);  permute_39 = None
    getitem_155 = split_tensor_26[0]
    getitem_156 = split_tensor_26[1]
    getitem_157 = split_tensor_26[2];  split_tensor_26 = None
    view_201 = torch.ops.aten.view.default(diffusion_transformer_blocks_2_q_bias, [1, 16, 1, 1, 48]);  diffusion_transformer_blocks_2_q_bias = None
    add_61 = torch.ops.aten.add.Tensor(getitem_155, view_201);  getitem_155 = view_201 = None
    view_202 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 384, 384, 1])
    bitwise_not_10 = torch.ops.aten.bitwise_not.default(view_202);  view_202 = None
    masked_fill_10 = torch.ops.aten.masked_fill.Scalar(view_195, bitwise_not_10, -10000);  view_195 = bitwise_not_10 = None
    permute_40 = torch.ops.aten.permute.default(masked_fill_10, [0, 4, 1, 2, 3]);  masked_fill_10 = None
    view_203 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 1, 384, 384])
    bitwise_not_11 = torch.ops.aten.bitwise_not.default(view_203);  view_203 = None
    masked_fill_11 = torch.ops.aten.masked_fill.Scalar(permute_40, bitwise_not_11, -10000);  permute_40 = bitwise_not_11 = None
    mul_39 = torch.ops.aten.mul.Scalar(add_61, 0.3799178428257963);  add_61 = None
    transpose_2 = torch.ops.aten.transpose.int(getitem_156, -2, -1);  getitem_156 = None
    mul_40 = torch.ops.aten.mul.Scalar(transpose_2, 0.3799178428257963);  transpose_2 = None
    expand_22 = torch.ops.aten.expand.default(mul_39, [1, 16, 5, 384, 48]);  mul_39 = None
    clone_30 = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
    _unsafe_view_29 = torch.ops.aten._unsafe_view.default(clone_30, [80, 384, 48]);  clone_30 = None
    expand_23 = torch.ops.aten.expand.default(mul_40, [1, 16, 5, 48, 384]);  mul_40 = None
    clone_31 = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    _unsafe_view_30 = torch.ops.aten._unsafe_view.default(clone_31, [80, 48, 384]);  clone_31 = None
    bmm_10 = torch.ops.aten.bmm.default(_unsafe_view_29, _unsafe_view_30);  _unsafe_view_29 = _unsafe_view_30 = None
    view_204 = torch.ops.aten.view.default(bmm_10, [1, 16, 5, 384, 384]);  bmm_10 = None
    add_62 = torch.ops.aten.add.Tensor(view_204, masked_fill_11);  view_204 = masked_fill_11 = None
    _softmax_2 = torch.ops.aten._softmax.default(add_62, -1, False);  add_62 = None
    expand_24 = torch.ops.aten.expand.default(_softmax_2, [1, 16, 5, 384, 384]);  _softmax_2 = None
    view_205 = torch.ops.aten.view.default(expand_24, [80, 384, 384]);  expand_24 = None
    expand_25 = torch.ops.aten.expand.default(getitem_157, [1, 16, 5, 384, 48]);  getitem_157 = None
    clone_32 = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    _unsafe_view_31 = torch.ops.aten._unsafe_view.default(clone_32, [80, 384, 48]);  clone_32 = None
    bmm_11 = torch.ops.aten.bmm.default(view_205, _unsafe_view_31);  view_205 = _unsafe_view_31 = None
    view_206 = torch.ops.aten.view.default(bmm_11, [1, 16, 5, 384, 48]);  bmm_11 = None
    permute_41 = torch.ops.aten.permute.default(view_206, [0, 2, 3, 1, 4]);  view_206 = None
    clone_33 = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
    _unsafe_view_32 = torch.ops.aten._unsafe_view.default(clone_33, [1, 5, 384, 768]);  clone_33 = None
    t_60 = torch.ops.aten.t.default(diffusion_transformer_blocks_2_to_out_weight);  diffusion_transformer_blocks_2_to_out_weight = None
    view_207 = torch.ops.aten.view.default(_unsafe_view_32, [1920, 768]);  _unsafe_view_32 = None
    mm_56 = torch.ops.aten.mm.default(view_207, t_60);  view_207 = t_60 = None
    view_208 = torch.ops.aten.view.default(mm_56, [1, 5, 384, 768]);  mm_56 = None
    view_209 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_61 = torch.ops.aten.t.default(diffusion_transformer_blocks_2_gate_proj_0_weight);  diffusion_transformer_blocks_2_gate_proj_0_weight = None
    addmm_4 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_2_gate_proj_0_bias, view_209, t_61);  diffusion_transformer_blocks_2_gate_proj_0_bias = view_209 = t_61 = None
    view_210 = torch.ops.aten.view.default(addmm_4, [1, 5, 384, 768]);  addmm_4 = None
    sigmoid_10 = torch.ops.aten.sigmoid.default(view_210);  view_210 = None
    mul_41 = torch.ops.aten.mul.Tensor(sigmoid_10, view_208);  sigmoid_10 = view_208 = None
    native_layer_norm_default_27 = torch.ops.aten.native_layer_norm.default(add_58, [768], None, None, 0.1)
    getitem_158 = native_layer_norm_default_27[0];  native_layer_norm_default_27 = None
    t_62 = torch.ops.aten.t.default(diffusion_transformer_blocks_2_transition_ada_ln_lin_s_merged_weight);  diffusion_transformer_blocks_2_transition_ada_ln_lin_s_merged_weight = None
    view_211 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_57 = torch.ops.aten.mm.default(view_211, t_62);  view_211 = t_62 = None
    view_212 = torch.ops.aten.view.default(mm_57, [1, 5, 384, 1536]);  mm_57 = None
    split_tensor_27 = torch.ops.aten.split.Tensor(view_212, 768, dim = -1);  view_212 = None
    getitem_161 = split_tensor_27[0]
    getitem_162 = split_tensor_27[1];  split_tensor_27 = None
    add_63 = torch.ops.aten.add.Tensor(getitem_161, 1);  getitem_161 = None
    mul_42 = torch.ops.aten.mul.Tensor(getitem_158, add_63);  getitem_158 = add_63 = None
    add_64 = torch.ops.aten.add.Tensor(mul_42, getitem_162);  mul_42 = getitem_162 = None
    t_63 = torch.ops.aten.t.default(diffusion_transformer_blocks_2_transition_linear_a_nobias_double_weight);  diffusion_transformer_blocks_2_transition_linear_a_nobias_double_weight = None
    view_213 = torch.ops.aten.view.default(add_64, [1920, 768]);  add_64 = None
    mm_58 = torch.ops.aten.mm.default(view_213, t_63);  view_213 = t_63 = None
    view_214 = torch.ops.aten.view.default(mm_58, [1, 5, 384, 3072]);  mm_58 = None
    split_tensor_28 = torch.ops.aten.split.Tensor(view_214, 1536, dim = -1);  view_214 = None
    getitem_163 = split_tensor_28[0]
    getitem_164 = split_tensor_28[1];  split_tensor_28 = None
    silu_9 = torch.ops.aten.silu.default(getitem_163);  getitem_163 = None
    mul_43 = torch.ops.aten.mul.Tensor(silu_9, getitem_164);  silu_9 = getitem_164 = None
    view_215 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_64 = torch.ops.aten.t.default(diffusion_transformer_blocks_2_transition_linear_s_biasinit_m2_weight);  diffusion_transformer_blocks_2_transition_linear_s_biasinit_m2_weight = None
    addmm_5 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_2_transition_linear_s_biasinit_m2_bias, view_215, t_64);  diffusion_transformer_blocks_2_transition_linear_s_biasinit_m2_bias = view_215 = t_64 = None
    view_216 = torch.ops.aten.view.default(addmm_5, [1, 5, 384, 768]);  addmm_5 = None
    sigmoid_11 = torch.ops.aten.sigmoid.default(view_216);  view_216 = None
    t_65 = torch.ops.aten.t.default(diffusion_transformer_blocks_2_transition_linear_b_nobias_weight);  diffusion_transformer_blocks_2_transition_linear_b_nobias_weight = None
    view_217 = torch.ops.aten.view.default(mul_43, [1920, 1536]);  mul_43 = None
    mm_59 = torch.ops.aten.mm.default(view_217, t_65);  view_217 = t_65 = None
    view_218 = torch.ops.aten.view.default(mm_59, [1, 5, 384, 768]);  mm_59 = None
    mul_44 = torch.ops.aten.mul.Tensor(sigmoid_11, view_218);  sigmoid_11 = view_218 = None
    add_65 = torch.ops.aten.add.Tensor(mul_41, mul_44);  mul_41 = mul_44 = None
    add_66 = torch.ops.aten.add.Tensor(add_58, add_65);  add_58 = add_65 = None
    native_layer_norm_default_28 = torch.ops.aten.native_layer_norm.default(view_140, [256], diffusion_transformer_blocks_3_pair_layer_norm_weight, diffusion_transformer_blocks_3_pair_layer_norm_bias, 1e-05);  diffusion_transformer_blocks_3_pair_layer_norm_weight = diffusion_transformer_blocks_3_pair_layer_norm_bias = None
    getitem_165 = native_layer_norm_default_28[0];  native_layer_norm_default_28 = None
    t_66 = torch.ops.aten.t.default(diffusion_transformer_blocks_3_pair_linear_weight);  diffusion_transformer_blocks_3_pair_linear_weight = None
    view_219 = torch.ops.aten.view.default(getitem_165, [147456, 256]);  getitem_165 = None
    mm_60 = torch.ops.aten.mm.default(view_219, t_66);  view_219 = t_66 = None
    view_220 = torch.ops.aten.view.default(mm_60, [1, 1, 384, 384, 16]);  mm_60 = None
    native_layer_norm_default_29 = torch.ops.aten.native_layer_norm.default(add_66, [768], None, None, 0.1)
    getitem_168 = native_layer_norm_default_29[0];  native_layer_norm_default_29 = None
    t_67 = torch.ops.aten.t.default(diffusion_transformer_blocks_3_norm_in_lin_s_merged_weight);  diffusion_transformer_blocks_3_norm_in_lin_s_merged_weight = None
    view_221 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_61 = torch.ops.aten.mm.default(view_221, t_67);  view_221 = t_67 = None
    view_222 = torch.ops.aten.view.default(mm_61, [1, 5, 384, 1536]);  mm_61 = None
    split_tensor_29 = torch.ops.aten.split.Tensor(view_222, 768, dim = -1);  view_222 = None
    getitem_171 = split_tensor_29[0]
    getitem_172 = split_tensor_29[1];  split_tensor_29 = None
    add_67 = torch.ops.aten.add.Tensor(getitem_171, 1);  getitem_171 = None
    mul_45 = torch.ops.aten.mul.Tensor(getitem_168, add_67);  getitem_168 = add_67 = None
    add_68 = torch.ops.aten.add.Tensor(mul_45, getitem_172);  mul_45 = getitem_172 = None
    t_68 = torch.ops.aten.t.default(diffusion_transformer_blocks_3_to_qkv_weight);  diffusion_transformer_blocks_3_to_qkv_weight = None
    view_223 = torch.ops.aten.view.default(add_68, [1920, 768]);  add_68 = None
    mm_62 = torch.ops.aten.mm.default(view_223, t_68);  view_223 = t_68 = None
    view_224 = torch.ops.aten.view.default(mm_62, [1, 5, 384, 2304]);  mm_62 = None
    view_225 = torch.ops.aten.view.default(view_224, [1, 5, 384, 16, 144]);  view_224 = None
    permute_42 = torch.ops.aten.permute.default(view_225, [0, 3, 1, 2, 4]);  view_225 = None
    split_tensor_30 = torch.ops.aten.split.Tensor(permute_42, 48, dim = -1);  permute_42 = None
    getitem_173 = split_tensor_30[0]
    getitem_174 = split_tensor_30[1]
    getitem_175 = split_tensor_30[2];  split_tensor_30 = None
    view_226 = torch.ops.aten.view.default(diffusion_transformer_blocks_3_q_bias, [1, 16, 1, 1, 48]);  diffusion_transformer_blocks_3_q_bias = None
    add_69 = torch.ops.aten.add.Tensor(getitem_173, view_226);  getitem_173 = view_226 = None
    view_227 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 384, 384, 1])
    bitwise_not_12 = torch.ops.aten.bitwise_not.default(view_227);  view_227 = None
    masked_fill_12 = torch.ops.aten.masked_fill.Scalar(view_220, bitwise_not_12, -10000);  view_220 = bitwise_not_12 = None
    permute_43 = torch.ops.aten.permute.default(masked_fill_12, [0, 4, 1, 2, 3]);  masked_fill_12 = None
    view_228 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 1, 384, 384])
    bitwise_not_13 = torch.ops.aten.bitwise_not.default(view_228);  view_228 = None
    masked_fill_13 = torch.ops.aten.masked_fill.Scalar(permute_43, bitwise_not_13, -10000);  permute_43 = bitwise_not_13 = None
    mul_46 = torch.ops.aten.mul.Scalar(add_69, 0.3799178428257963);  add_69 = None
    transpose_3 = torch.ops.aten.transpose.int(getitem_174, -2, -1);  getitem_174 = None
    mul_47 = torch.ops.aten.mul.Scalar(transpose_3, 0.3799178428257963);  transpose_3 = None
    expand_26 = torch.ops.aten.expand.default(mul_46, [1, 16, 5, 384, 48]);  mul_46 = None
    clone_34 = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    _unsafe_view_33 = torch.ops.aten._unsafe_view.default(clone_34, [80, 384, 48]);  clone_34 = None
    expand_27 = torch.ops.aten.expand.default(mul_47, [1, 16, 5, 48, 384]);  mul_47 = None
    clone_35 = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    _unsafe_view_34 = torch.ops.aten._unsafe_view.default(clone_35, [80, 48, 384]);  clone_35 = None
    bmm_12 = torch.ops.aten.bmm.default(_unsafe_view_33, _unsafe_view_34);  _unsafe_view_33 = _unsafe_view_34 = None
    view_229 = torch.ops.aten.view.default(bmm_12, [1, 16, 5, 384, 384]);  bmm_12 = None
    add_70 = torch.ops.aten.add.Tensor(view_229, masked_fill_13);  view_229 = masked_fill_13 = None
    _softmax_3 = torch.ops.aten._softmax.default(add_70, -1, False);  add_70 = None
    expand_28 = torch.ops.aten.expand.default(_softmax_3, [1, 16, 5, 384, 384]);  _softmax_3 = None
    view_230 = torch.ops.aten.view.default(expand_28, [80, 384, 384]);  expand_28 = None
    expand_29 = torch.ops.aten.expand.default(getitem_175, [1, 16, 5, 384, 48]);  getitem_175 = None
    clone_36 = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    _unsafe_view_35 = torch.ops.aten._unsafe_view.default(clone_36, [80, 384, 48]);  clone_36 = None
    bmm_13 = torch.ops.aten.bmm.default(view_230, _unsafe_view_35);  view_230 = _unsafe_view_35 = None
    view_231 = torch.ops.aten.view.default(bmm_13, [1, 16, 5, 384, 48]);  bmm_13 = None
    permute_44 = torch.ops.aten.permute.default(view_231, [0, 2, 3, 1, 4]);  view_231 = None
    clone_37 = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
    _unsafe_view_36 = torch.ops.aten._unsafe_view.default(clone_37, [1, 5, 384, 768]);  clone_37 = None
    t_69 = torch.ops.aten.t.default(diffusion_transformer_blocks_3_to_out_weight);  diffusion_transformer_blocks_3_to_out_weight = None
    view_232 = torch.ops.aten.view.default(_unsafe_view_36, [1920, 768]);  _unsafe_view_36 = None
    mm_63 = torch.ops.aten.mm.default(view_232, t_69);  view_232 = t_69 = None
    view_233 = torch.ops.aten.view.default(mm_63, [1, 5, 384, 768]);  mm_63 = None
    view_234 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_70 = torch.ops.aten.t.default(diffusion_transformer_blocks_3_gate_proj_0_weight);  diffusion_transformer_blocks_3_gate_proj_0_weight = None
    addmm_6 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_3_gate_proj_0_bias, view_234, t_70);  diffusion_transformer_blocks_3_gate_proj_0_bias = view_234 = t_70 = None
    view_235 = torch.ops.aten.view.default(addmm_6, [1, 5, 384, 768]);  addmm_6 = None
    sigmoid_12 = torch.ops.aten.sigmoid.default(view_235);  view_235 = None
    mul_48 = torch.ops.aten.mul.Tensor(sigmoid_12, view_233);  sigmoid_12 = view_233 = None
    native_layer_norm_default_30 = torch.ops.aten.native_layer_norm.default(add_66, [768], None, None, 0.1)
    getitem_176 = native_layer_norm_default_30[0];  native_layer_norm_default_30 = None
    t_71 = torch.ops.aten.t.default(diffusion_transformer_blocks_3_transition_ada_ln_lin_s_merged_weight);  diffusion_transformer_blocks_3_transition_ada_ln_lin_s_merged_weight = None
    view_236 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_64 = torch.ops.aten.mm.default(view_236, t_71);  view_236 = t_71 = None
    view_237 = torch.ops.aten.view.default(mm_64, [1, 5, 384, 1536]);  mm_64 = None
    split_tensor_31 = torch.ops.aten.split.Tensor(view_237, 768, dim = -1);  view_237 = None
    getitem_179 = split_tensor_31[0]
    getitem_180 = split_tensor_31[1];  split_tensor_31 = None
    add_71 = torch.ops.aten.add.Tensor(getitem_179, 1);  getitem_179 = None
    mul_49 = torch.ops.aten.mul.Tensor(getitem_176, add_71);  getitem_176 = add_71 = None
    add_72 = torch.ops.aten.add.Tensor(mul_49, getitem_180);  mul_49 = getitem_180 = None
    t_72 = torch.ops.aten.t.default(diffusion_transformer_blocks_3_transition_linear_a_nobias_double_weight);  diffusion_transformer_blocks_3_transition_linear_a_nobias_double_weight = None
    view_238 = torch.ops.aten.view.default(add_72, [1920, 768]);  add_72 = None
    mm_65 = torch.ops.aten.mm.default(view_238, t_72);  view_238 = t_72 = None
    view_239 = torch.ops.aten.view.default(mm_65, [1, 5, 384, 3072]);  mm_65 = None
    split_tensor_32 = torch.ops.aten.split.Tensor(view_239, 1536, dim = -1);  view_239 = None
    getitem_181 = split_tensor_32[0]
    getitem_182 = split_tensor_32[1];  split_tensor_32 = None
    silu_10 = torch.ops.aten.silu.default(getitem_181);  getitem_181 = None
    mul_50 = torch.ops.aten.mul.Tensor(silu_10, getitem_182);  silu_10 = getitem_182 = None
    view_240 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_73 = torch.ops.aten.t.default(diffusion_transformer_blocks_3_transition_linear_s_biasinit_m2_weight);  diffusion_transformer_blocks_3_transition_linear_s_biasinit_m2_weight = None
    addmm_7 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_3_transition_linear_s_biasinit_m2_bias, view_240, t_73);  diffusion_transformer_blocks_3_transition_linear_s_biasinit_m2_bias = view_240 = t_73 = None
    view_241 = torch.ops.aten.view.default(addmm_7, [1, 5, 384, 768]);  addmm_7 = None
    sigmoid_13 = torch.ops.aten.sigmoid.default(view_241);  view_241 = None
    t_74 = torch.ops.aten.t.default(diffusion_transformer_blocks_3_transition_linear_b_nobias_weight);  diffusion_transformer_blocks_3_transition_linear_b_nobias_weight = None
    view_242 = torch.ops.aten.view.default(mul_50, [1920, 1536]);  mul_50 = None
    mm_66 = torch.ops.aten.mm.default(view_242, t_74);  view_242 = t_74 = None
    view_243 = torch.ops.aten.view.default(mm_66, [1, 5, 384, 768]);  mm_66 = None
    mul_51 = torch.ops.aten.mul.Tensor(sigmoid_13, view_243);  sigmoid_13 = view_243 = None
    add_73 = torch.ops.aten.add.Tensor(mul_48, mul_51);  mul_48 = mul_51 = None
    add_74 = torch.ops.aten.add.Tensor(add_66, add_73);  add_66 = add_73 = None
    native_layer_norm_default_31 = torch.ops.aten.native_layer_norm.default(view_140, [256], diffusion_transformer_blocks_4_pair_layer_norm_weight, diffusion_transformer_blocks_4_pair_layer_norm_bias, 1e-05);  diffusion_transformer_blocks_4_pair_layer_norm_weight = diffusion_transformer_blocks_4_pair_layer_norm_bias = None
    getitem_183 = native_layer_norm_default_31[0];  native_layer_norm_default_31 = None
    t_75 = torch.ops.aten.t.default(diffusion_transformer_blocks_4_pair_linear_weight);  diffusion_transformer_blocks_4_pair_linear_weight = None
    view_244 = torch.ops.aten.view.default(getitem_183, [147456, 256]);  getitem_183 = None
    mm_67 = torch.ops.aten.mm.default(view_244, t_75);  view_244 = t_75 = None
    view_245 = torch.ops.aten.view.default(mm_67, [1, 1, 384, 384, 16]);  mm_67 = None
    native_layer_norm_default_32 = torch.ops.aten.native_layer_norm.default(add_74, [768], None, None, 0.1)
    getitem_186 = native_layer_norm_default_32[0];  native_layer_norm_default_32 = None
    t_76 = torch.ops.aten.t.default(diffusion_transformer_blocks_4_norm_in_lin_s_merged_weight);  diffusion_transformer_blocks_4_norm_in_lin_s_merged_weight = None
    view_246 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_68 = torch.ops.aten.mm.default(view_246, t_76);  view_246 = t_76 = None
    view_247 = torch.ops.aten.view.default(mm_68, [1, 5, 384, 1536]);  mm_68 = None
    split_tensor_33 = torch.ops.aten.split.Tensor(view_247, 768, dim = -1);  view_247 = None
    getitem_189 = split_tensor_33[0]
    getitem_190 = split_tensor_33[1];  split_tensor_33 = None
    add_75 = torch.ops.aten.add.Tensor(getitem_189, 1);  getitem_189 = None
    mul_52 = torch.ops.aten.mul.Tensor(getitem_186, add_75);  getitem_186 = add_75 = None
    add_76 = torch.ops.aten.add.Tensor(mul_52, getitem_190);  mul_52 = getitem_190 = None
    t_77 = torch.ops.aten.t.default(diffusion_transformer_blocks_4_to_qkv_weight);  diffusion_transformer_blocks_4_to_qkv_weight = None
    view_248 = torch.ops.aten.view.default(add_76, [1920, 768]);  add_76 = None
    mm_69 = torch.ops.aten.mm.default(view_248, t_77);  view_248 = t_77 = None
    view_249 = torch.ops.aten.view.default(mm_69, [1, 5, 384, 2304]);  mm_69 = None
    view_250 = torch.ops.aten.view.default(view_249, [1, 5, 384, 16, 144]);  view_249 = None
    permute_45 = torch.ops.aten.permute.default(view_250, [0, 3, 1, 2, 4]);  view_250 = None
    split_tensor_34 = torch.ops.aten.split.Tensor(permute_45, 48, dim = -1);  permute_45 = None
    getitem_191 = split_tensor_34[0]
    getitem_192 = split_tensor_34[1]
    getitem_193 = split_tensor_34[2];  split_tensor_34 = None
    view_251 = torch.ops.aten.view.default(diffusion_transformer_blocks_4_q_bias, [1, 16, 1, 1, 48]);  diffusion_transformer_blocks_4_q_bias = None
    add_77 = torch.ops.aten.add.Tensor(getitem_191, view_251);  getitem_191 = view_251 = None
    view_252 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 384, 384, 1])
    bitwise_not_14 = torch.ops.aten.bitwise_not.default(view_252);  view_252 = None
    masked_fill_14 = torch.ops.aten.masked_fill.Scalar(view_245, bitwise_not_14, -10000);  view_245 = bitwise_not_14 = None
    permute_46 = torch.ops.aten.permute.default(masked_fill_14, [0, 4, 1, 2, 3]);  masked_fill_14 = None
    view_253 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 1, 384, 384])
    bitwise_not_15 = torch.ops.aten.bitwise_not.default(view_253);  view_253 = None
    masked_fill_15 = torch.ops.aten.masked_fill.Scalar(permute_46, bitwise_not_15, -10000);  permute_46 = bitwise_not_15 = None
    mul_53 = torch.ops.aten.mul.Scalar(add_77, 0.3799178428257963);  add_77 = None
    transpose_4 = torch.ops.aten.transpose.int(getitem_192, -2, -1);  getitem_192 = None
    mul_54 = torch.ops.aten.mul.Scalar(transpose_4, 0.3799178428257963);  transpose_4 = None
    expand_30 = torch.ops.aten.expand.default(mul_53, [1, 16, 5, 384, 48]);  mul_53 = None
    clone_38 = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
    _unsafe_view_37 = torch.ops.aten._unsafe_view.default(clone_38, [80, 384, 48]);  clone_38 = None
    expand_31 = torch.ops.aten.expand.default(mul_54, [1, 16, 5, 48, 384]);  mul_54 = None
    clone_39 = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    _unsafe_view_38 = torch.ops.aten._unsafe_view.default(clone_39, [80, 48, 384]);  clone_39 = None
    bmm_14 = torch.ops.aten.bmm.default(_unsafe_view_37, _unsafe_view_38);  _unsafe_view_37 = _unsafe_view_38 = None
    view_254 = torch.ops.aten.view.default(bmm_14, [1, 16, 5, 384, 384]);  bmm_14 = None
    add_78 = torch.ops.aten.add.Tensor(view_254, masked_fill_15);  view_254 = masked_fill_15 = None
    _softmax_4 = torch.ops.aten._softmax.default(add_78, -1, False);  add_78 = None
    expand_32 = torch.ops.aten.expand.default(_softmax_4, [1, 16, 5, 384, 384]);  _softmax_4 = None
    view_255 = torch.ops.aten.view.default(expand_32, [80, 384, 384]);  expand_32 = None
    expand_33 = torch.ops.aten.expand.default(getitem_193, [1, 16, 5, 384, 48]);  getitem_193 = None
    clone_40 = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    _unsafe_view_39 = torch.ops.aten._unsafe_view.default(clone_40, [80, 384, 48]);  clone_40 = None
    bmm_15 = torch.ops.aten.bmm.default(view_255, _unsafe_view_39);  view_255 = _unsafe_view_39 = None
    view_256 = torch.ops.aten.view.default(bmm_15, [1, 16, 5, 384, 48]);  bmm_15 = None
    permute_47 = torch.ops.aten.permute.default(view_256, [0, 2, 3, 1, 4]);  view_256 = None
    clone_41 = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
    _unsafe_view_40 = torch.ops.aten._unsafe_view.default(clone_41, [1, 5, 384, 768]);  clone_41 = None
    t_78 = torch.ops.aten.t.default(diffusion_transformer_blocks_4_to_out_weight);  diffusion_transformer_blocks_4_to_out_weight = None
    view_257 = torch.ops.aten.view.default(_unsafe_view_40, [1920, 768]);  _unsafe_view_40 = None
    mm_70 = torch.ops.aten.mm.default(view_257, t_78);  view_257 = t_78 = None
    view_258 = torch.ops.aten.view.default(mm_70, [1, 5, 384, 768]);  mm_70 = None
    view_259 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_79 = torch.ops.aten.t.default(diffusion_transformer_blocks_4_gate_proj_0_weight);  diffusion_transformer_blocks_4_gate_proj_0_weight = None
    addmm_8 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_4_gate_proj_0_bias, view_259, t_79);  diffusion_transformer_blocks_4_gate_proj_0_bias = view_259 = t_79 = None
    view_260 = torch.ops.aten.view.default(addmm_8, [1, 5, 384, 768]);  addmm_8 = None
    sigmoid_14 = torch.ops.aten.sigmoid.default(view_260);  view_260 = None
    mul_55 = torch.ops.aten.mul.Tensor(sigmoid_14, view_258);  sigmoid_14 = view_258 = None
    native_layer_norm_default_33 = torch.ops.aten.native_layer_norm.default(add_74, [768], None, None, 0.1)
    getitem_194 = native_layer_norm_default_33[0];  native_layer_norm_default_33 = None
    t_80 = torch.ops.aten.t.default(diffusion_transformer_blocks_4_transition_ada_ln_lin_s_merged_weight);  diffusion_transformer_blocks_4_transition_ada_ln_lin_s_merged_weight = None
    view_261 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_71 = torch.ops.aten.mm.default(view_261, t_80);  view_261 = t_80 = None
    view_262 = torch.ops.aten.view.default(mm_71, [1, 5, 384, 1536]);  mm_71 = None
    split_tensor_35 = torch.ops.aten.split.Tensor(view_262, 768, dim = -1);  view_262 = None
    getitem_197 = split_tensor_35[0]
    getitem_198 = split_tensor_35[1];  split_tensor_35 = None
    add_79 = torch.ops.aten.add.Tensor(getitem_197, 1);  getitem_197 = None
    mul_56 = torch.ops.aten.mul.Tensor(getitem_194, add_79);  getitem_194 = add_79 = None
    add_80 = torch.ops.aten.add.Tensor(mul_56, getitem_198);  mul_56 = getitem_198 = None
    t_81 = torch.ops.aten.t.default(diffusion_transformer_blocks_4_transition_linear_a_nobias_double_weight);  diffusion_transformer_blocks_4_transition_linear_a_nobias_double_weight = None
    view_263 = torch.ops.aten.view.default(add_80, [1920, 768]);  add_80 = None
    mm_72 = torch.ops.aten.mm.default(view_263, t_81);  view_263 = t_81 = None
    view_264 = torch.ops.aten.view.default(mm_72, [1, 5, 384, 3072]);  mm_72 = None
    split_tensor_36 = torch.ops.aten.split.Tensor(view_264, 1536, dim = -1);  view_264 = None
    getitem_199 = split_tensor_36[0]
    getitem_200 = split_tensor_36[1];  split_tensor_36 = None
    silu_11 = torch.ops.aten.silu.default(getitem_199);  getitem_199 = None
    mul_57 = torch.ops.aten.mul.Tensor(silu_11, getitem_200);  silu_11 = getitem_200 = None
    view_265 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_82 = torch.ops.aten.t.default(diffusion_transformer_blocks_4_transition_linear_s_biasinit_m2_weight);  diffusion_transformer_blocks_4_transition_linear_s_biasinit_m2_weight = None
    addmm_9 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_4_transition_linear_s_biasinit_m2_bias, view_265, t_82);  diffusion_transformer_blocks_4_transition_linear_s_biasinit_m2_bias = view_265 = t_82 = None
    view_266 = torch.ops.aten.view.default(addmm_9, [1, 5, 384, 768]);  addmm_9 = None
    sigmoid_15 = torch.ops.aten.sigmoid.default(view_266);  view_266 = None
    t_83 = torch.ops.aten.t.default(diffusion_transformer_blocks_4_transition_linear_b_nobias_weight);  diffusion_transformer_blocks_4_transition_linear_b_nobias_weight = None
    view_267 = torch.ops.aten.view.default(mul_57, [1920, 1536]);  mul_57 = None
    mm_73 = torch.ops.aten.mm.default(view_267, t_83);  view_267 = t_83 = None
    view_268 = torch.ops.aten.view.default(mm_73, [1, 5, 384, 768]);  mm_73 = None
    mul_58 = torch.ops.aten.mul.Tensor(sigmoid_15, view_268);  sigmoid_15 = view_268 = None
    add_81 = torch.ops.aten.add.Tensor(mul_55, mul_58);  mul_55 = mul_58 = None
    add_82 = torch.ops.aten.add.Tensor(add_74, add_81);  add_74 = add_81 = None
    native_layer_norm_default_34 = torch.ops.aten.native_layer_norm.default(view_140, [256], diffusion_transformer_blocks_5_pair_layer_norm_weight, diffusion_transformer_blocks_5_pair_layer_norm_bias, 1e-05);  diffusion_transformer_blocks_5_pair_layer_norm_weight = diffusion_transformer_blocks_5_pair_layer_norm_bias = None
    getitem_201 = native_layer_norm_default_34[0];  native_layer_norm_default_34 = None
    t_84 = torch.ops.aten.t.default(diffusion_transformer_blocks_5_pair_linear_weight);  diffusion_transformer_blocks_5_pair_linear_weight = None
    view_269 = torch.ops.aten.view.default(getitem_201, [147456, 256]);  getitem_201 = None
    mm_74 = torch.ops.aten.mm.default(view_269, t_84);  view_269 = t_84 = None
    view_270 = torch.ops.aten.view.default(mm_74, [1, 1, 384, 384, 16]);  mm_74 = None
    native_layer_norm_default_35 = torch.ops.aten.native_layer_norm.default(add_82, [768], None, None, 0.1)
    getitem_204 = native_layer_norm_default_35[0];  native_layer_norm_default_35 = None
    t_85 = torch.ops.aten.t.default(diffusion_transformer_blocks_5_norm_in_lin_s_merged_weight);  diffusion_transformer_blocks_5_norm_in_lin_s_merged_weight = None
    view_271 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_75 = torch.ops.aten.mm.default(view_271, t_85);  view_271 = t_85 = None
    view_272 = torch.ops.aten.view.default(mm_75, [1, 5, 384, 1536]);  mm_75 = None
    split_tensor_37 = torch.ops.aten.split.Tensor(view_272, 768, dim = -1);  view_272 = None
    getitem_207 = split_tensor_37[0]
    getitem_208 = split_tensor_37[1];  split_tensor_37 = None
    add_83 = torch.ops.aten.add.Tensor(getitem_207, 1);  getitem_207 = None
    mul_59 = torch.ops.aten.mul.Tensor(getitem_204, add_83);  getitem_204 = add_83 = None
    add_84 = torch.ops.aten.add.Tensor(mul_59, getitem_208);  mul_59 = getitem_208 = None
    t_86 = torch.ops.aten.t.default(diffusion_transformer_blocks_5_to_qkv_weight);  diffusion_transformer_blocks_5_to_qkv_weight = None
    view_273 = torch.ops.aten.view.default(add_84, [1920, 768]);  add_84 = None
    mm_76 = torch.ops.aten.mm.default(view_273, t_86);  view_273 = t_86 = None
    view_274 = torch.ops.aten.view.default(mm_76, [1, 5, 384, 2304]);  mm_76 = None
    view_275 = torch.ops.aten.view.default(view_274, [1, 5, 384, 16, 144]);  view_274 = None
    permute_48 = torch.ops.aten.permute.default(view_275, [0, 3, 1, 2, 4]);  view_275 = None
    split_tensor_38 = torch.ops.aten.split.Tensor(permute_48, 48, dim = -1);  permute_48 = None
    getitem_209 = split_tensor_38[0]
    getitem_210 = split_tensor_38[1]
    getitem_211 = split_tensor_38[2];  split_tensor_38 = None
    view_276 = torch.ops.aten.view.default(diffusion_transformer_blocks_5_q_bias, [1, 16, 1, 1, 48]);  diffusion_transformer_blocks_5_q_bias = None
    add_85 = torch.ops.aten.add.Tensor(getitem_209, view_276);  getitem_209 = view_276 = None
    view_277 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 384, 384, 1])
    bitwise_not_16 = torch.ops.aten.bitwise_not.default(view_277);  view_277 = None
    masked_fill_16 = torch.ops.aten.masked_fill.Scalar(view_270, bitwise_not_16, -10000);  view_270 = bitwise_not_16 = None
    permute_49 = torch.ops.aten.permute.default(masked_fill_16, [0, 4, 1, 2, 3]);  masked_fill_16 = None
    view_278 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 1, 384, 384])
    bitwise_not_17 = torch.ops.aten.bitwise_not.default(view_278);  view_278 = None
    masked_fill_17 = torch.ops.aten.masked_fill.Scalar(permute_49, bitwise_not_17, -10000);  permute_49 = bitwise_not_17 = None
    mul_60 = torch.ops.aten.mul.Scalar(add_85, 0.3799178428257963);  add_85 = None
    transpose_5 = torch.ops.aten.transpose.int(getitem_210, -2, -1);  getitem_210 = None
    mul_61 = torch.ops.aten.mul.Scalar(transpose_5, 0.3799178428257963);  transpose_5 = None
    expand_34 = torch.ops.aten.expand.default(mul_60, [1, 16, 5, 384, 48]);  mul_60 = None
    clone_42 = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
    _unsafe_view_41 = torch.ops.aten._unsafe_view.default(clone_42, [80, 384, 48]);  clone_42 = None
    expand_35 = torch.ops.aten.expand.default(mul_61, [1, 16, 5, 48, 384]);  mul_61 = None
    clone_43 = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    _unsafe_view_42 = torch.ops.aten._unsafe_view.default(clone_43, [80, 48, 384]);  clone_43 = None
    bmm_16 = torch.ops.aten.bmm.default(_unsafe_view_41, _unsafe_view_42);  _unsafe_view_41 = _unsafe_view_42 = None
    view_279 = torch.ops.aten.view.default(bmm_16, [1, 16, 5, 384, 384]);  bmm_16 = None
    add_86 = torch.ops.aten.add.Tensor(view_279, masked_fill_17);  view_279 = masked_fill_17 = None
    _softmax_5 = torch.ops.aten._softmax.default(add_86, -1, False);  add_86 = None
    expand_36 = torch.ops.aten.expand.default(_softmax_5, [1, 16, 5, 384, 384]);  _softmax_5 = None
    view_280 = torch.ops.aten.view.default(expand_36, [80, 384, 384]);  expand_36 = None
    expand_37 = torch.ops.aten.expand.default(getitem_211, [1, 16, 5, 384, 48]);  getitem_211 = None
    clone_44 = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    _unsafe_view_43 = torch.ops.aten._unsafe_view.default(clone_44, [80, 384, 48]);  clone_44 = None
    bmm_17 = torch.ops.aten.bmm.default(view_280, _unsafe_view_43);  view_280 = _unsafe_view_43 = None
    view_281 = torch.ops.aten.view.default(bmm_17, [1, 16, 5, 384, 48]);  bmm_17 = None
    permute_50 = torch.ops.aten.permute.default(view_281, [0, 2, 3, 1, 4]);  view_281 = None
    clone_45 = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
    _unsafe_view_44 = torch.ops.aten._unsafe_view.default(clone_45, [1, 5, 384, 768]);  clone_45 = None
    t_87 = torch.ops.aten.t.default(diffusion_transformer_blocks_5_to_out_weight);  diffusion_transformer_blocks_5_to_out_weight = None
    view_282 = torch.ops.aten.view.default(_unsafe_view_44, [1920, 768]);  _unsafe_view_44 = None
    mm_77 = torch.ops.aten.mm.default(view_282, t_87);  view_282 = t_87 = None
    view_283 = torch.ops.aten.view.default(mm_77, [1, 5, 384, 768]);  mm_77 = None
    view_284 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_88 = torch.ops.aten.t.default(diffusion_transformer_blocks_5_gate_proj_0_weight);  diffusion_transformer_blocks_5_gate_proj_0_weight = None
    addmm_10 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_5_gate_proj_0_bias, view_284, t_88);  diffusion_transformer_blocks_5_gate_proj_0_bias = view_284 = t_88 = None
    view_285 = torch.ops.aten.view.default(addmm_10, [1, 5, 384, 768]);  addmm_10 = None
    sigmoid_16 = torch.ops.aten.sigmoid.default(view_285);  view_285 = None
    mul_62 = torch.ops.aten.mul.Tensor(sigmoid_16, view_283);  sigmoid_16 = view_283 = None
    native_layer_norm_default_36 = torch.ops.aten.native_layer_norm.default(add_82, [768], None, None, 0.1)
    getitem_212 = native_layer_norm_default_36[0];  native_layer_norm_default_36 = None
    t_89 = torch.ops.aten.t.default(diffusion_transformer_blocks_5_transition_ada_ln_lin_s_merged_weight);  diffusion_transformer_blocks_5_transition_ada_ln_lin_s_merged_weight = None
    view_286 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_78 = torch.ops.aten.mm.default(view_286, t_89);  view_286 = t_89 = None
    view_287 = torch.ops.aten.view.default(mm_78, [1, 5, 384, 1536]);  mm_78 = None
    split_tensor_39 = torch.ops.aten.split.Tensor(view_287, 768, dim = -1);  view_287 = None
    getitem_215 = split_tensor_39[0]
    getitem_216 = split_tensor_39[1];  split_tensor_39 = None
    add_87 = torch.ops.aten.add.Tensor(getitem_215, 1);  getitem_215 = None
    mul_63 = torch.ops.aten.mul.Tensor(getitem_212, add_87);  getitem_212 = add_87 = None
    add_88 = torch.ops.aten.add.Tensor(mul_63, getitem_216);  mul_63 = getitem_216 = None
    t_90 = torch.ops.aten.t.default(diffusion_transformer_blocks_5_transition_linear_a_nobias_double_weight);  diffusion_transformer_blocks_5_transition_linear_a_nobias_double_weight = None
    view_288 = torch.ops.aten.view.default(add_88, [1920, 768]);  add_88 = None
    mm_79 = torch.ops.aten.mm.default(view_288, t_90);  view_288 = t_90 = None
    view_289 = torch.ops.aten.view.default(mm_79, [1, 5, 384, 3072]);  mm_79 = None
    split_tensor_40 = torch.ops.aten.split.Tensor(view_289, 1536, dim = -1);  view_289 = None
    getitem_217 = split_tensor_40[0]
    getitem_218 = split_tensor_40[1];  split_tensor_40 = None
    silu_12 = torch.ops.aten.silu.default(getitem_217);  getitem_217 = None
    mul_64 = torch.ops.aten.mul.Tensor(silu_12, getitem_218);  silu_12 = getitem_218 = None
    view_290 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_91 = torch.ops.aten.t.default(diffusion_transformer_blocks_5_transition_linear_s_biasinit_m2_weight);  diffusion_transformer_blocks_5_transition_linear_s_biasinit_m2_weight = None
    addmm_11 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_5_transition_linear_s_biasinit_m2_bias, view_290, t_91);  diffusion_transformer_blocks_5_transition_linear_s_biasinit_m2_bias = view_290 = t_91 = None
    view_291 = torch.ops.aten.view.default(addmm_11, [1, 5, 384, 768]);  addmm_11 = None
    sigmoid_17 = torch.ops.aten.sigmoid.default(view_291);  view_291 = None
    t_92 = torch.ops.aten.t.default(diffusion_transformer_blocks_5_transition_linear_b_nobias_weight);  diffusion_transformer_blocks_5_transition_linear_b_nobias_weight = None
    view_292 = torch.ops.aten.view.default(mul_64, [1920, 1536]);  mul_64 = None
    mm_80 = torch.ops.aten.mm.default(view_292, t_92);  view_292 = t_92 = None
    view_293 = torch.ops.aten.view.default(mm_80, [1, 5, 384, 768]);  mm_80 = None
    mul_65 = torch.ops.aten.mul.Tensor(sigmoid_17, view_293);  sigmoid_17 = view_293 = None
    add_89 = torch.ops.aten.add.Tensor(mul_62, mul_65);  mul_62 = mul_65 = None
    add_90 = torch.ops.aten.add.Tensor(add_82, add_89);  add_82 = add_89 = None
    native_layer_norm_default_37 = torch.ops.aten.native_layer_norm.default(view_140, [256], diffusion_transformer_blocks_6_pair_layer_norm_weight, diffusion_transformer_blocks_6_pair_layer_norm_bias, 1e-05);  diffusion_transformer_blocks_6_pair_layer_norm_weight = diffusion_transformer_blocks_6_pair_layer_norm_bias = None
    getitem_219 = native_layer_norm_default_37[0];  native_layer_norm_default_37 = None
    t_93 = torch.ops.aten.t.default(diffusion_transformer_blocks_6_pair_linear_weight);  diffusion_transformer_blocks_6_pair_linear_weight = None
    view_294 = torch.ops.aten.view.default(getitem_219, [147456, 256]);  getitem_219 = None
    mm_81 = torch.ops.aten.mm.default(view_294, t_93);  view_294 = t_93 = None
    view_295 = torch.ops.aten.view.default(mm_81, [1, 1, 384, 384, 16]);  mm_81 = None
    native_layer_norm_default_38 = torch.ops.aten.native_layer_norm.default(add_90, [768], None, None, 0.1)
    getitem_222 = native_layer_norm_default_38[0];  native_layer_norm_default_38 = None
    t_94 = torch.ops.aten.t.default(diffusion_transformer_blocks_6_norm_in_lin_s_merged_weight);  diffusion_transformer_blocks_6_norm_in_lin_s_merged_weight = None
    view_296 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_82 = torch.ops.aten.mm.default(view_296, t_94);  view_296 = t_94 = None
    view_297 = torch.ops.aten.view.default(mm_82, [1, 5, 384, 1536]);  mm_82 = None
    split_tensor_41 = torch.ops.aten.split.Tensor(view_297, 768, dim = -1);  view_297 = None
    getitem_225 = split_tensor_41[0]
    getitem_226 = split_tensor_41[1];  split_tensor_41 = None
    add_91 = torch.ops.aten.add.Tensor(getitem_225, 1);  getitem_225 = None
    mul_66 = torch.ops.aten.mul.Tensor(getitem_222, add_91);  getitem_222 = add_91 = None
    add_92 = torch.ops.aten.add.Tensor(mul_66, getitem_226);  mul_66 = getitem_226 = None
    t_95 = torch.ops.aten.t.default(diffusion_transformer_blocks_6_to_qkv_weight);  diffusion_transformer_blocks_6_to_qkv_weight = None
    view_298 = torch.ops.aten.view.default(add_92, [1920, 768]);  add_92 = None
    mm_83 = torch.ops.aten.mm.default(view_298, t_95);  view_298 = t_95 = None
    view_299 = torch.ops.aten.view.default(mm_83, [1, 5, 384, 2304]);  mm_83 = None
    view_300 = torch.ops.aten.view.default(view_299, [1, 5, 384, 16, 144]);  view_299 = None
    permute_51 = torch.ops.aten.permute.default(view_300, [0, 3, 1, 2, 4]);  view_300 = None
    split_tensor_42 = torch.ops.aten.split.Tensor(permute_51, 48, dim = -1);  permute_51 = None
    getitem_227 = split_tensor_42[0]
    getitem_228 = split_tensor_42[1]
    getitem_229 = split_tensor_42[2];  split_tensor_42 = None
    view_301 = torch.ops.aten.view.default(diffusion_transformer_blocks_6_q_bias, [1, 16, 1, 1, 48]);  diffusion_transformer_blocks_6_q_bias = None
    add_93 = torch.ops.aten.add.Tensor(getitem_227, view_301);  getitem_227 = view_301 = None
    view_302 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 384, 384, 1])
    bitwise_not_18 = torch.ops.aten.bitwise_not.default(view_302);  view_302 = None
    masked_fill_18 = torch.ops.aten.masked_fill.Scalar(view_295, bitwise_not_18, -10000);  view_295 = bitwise_not_18 = None
    permute_52 = torch.ops.aten.permute.default(masked_fill_18, [0, 4, 1, 2, 3]);  masked_fill_18 = None
    view_303 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 1, 384, 384])
    bitwise_not_19 = torch.ops.aten.bitwise_not.default(view_303);  view_303 = None
    masked_fill_19 = torch.ops.aten.masked_fill.Scalar(permute_52, bitwise_not_19, -10000);  permute_52 = bitwise_not_19 = None
    mul_67 = torch.ops.aten.mul.Scalar(add_93, 0.3799178428257963);  add_93 = None
    transpose_6 = torch.ops.aten.transpose.int(getitem_228, -2, -1);  getitem_228 = None
    mul_68 = torch.ops.aten.mul.Scalar(transpose_6, 0.3799178428257963);  transpose_6 = None
    expand_38 = torch.ops.aten.expand.default(mul_67, [1, 16, 5, 384, 48]);  mul_67 = None
    clone_46 = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
    _unsafe_view_45 = torch.ops.aten._unsafe_view.default(clone_46, [80, 384, 48]);  clone_46 = None
    expand_39 = torch.ops.aten.expand.default(mul_68, [1, 16, 5, 48, 384]);  mul_68 = None
    clone_47 = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    _unsafe_view_46 = torch.ops.aten._unsafe_view.default(clone_47, [80, 48, 384]);  clone_47 = None
    bmm_18 = torch.ops.aten.bmm.default(_unsafe_view_45, _unsafe_view_46);  _unsafe_view_45 = _unsafe_view_46 = None
    view_304 = torch.ops.aten.view.default(bmm_18, [1, 16, 5, 384, 384]);  bmm_18 = None
    add_94 = torch.ops.aten.add.Tensor(view_304, masked_fill_19);  view_304 = masked_fill_19 = None
    _softmax_6 = torch.ops.aten._softmax.default(add_94, -1, False);  add_94 = None
    expand_40 = torch.ops.aten.expand.default(_softmax_6, [1, 16, 5, 384, 384]);  _softmax_6 = None
    view_305 = torch.ops.aten.view.default(expand_40, [80, 384, 384]);  expand_40 = None
    expand_41 = torch.ops.aten.expand.default(getitem_229, [1, 16, 5, 384, 48]);  getitem_229 = None
    clone_48 = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    _unsafe_view_47 = torch.ops.aten._unsafe_view.default(clone_48, [80, 384, 48]);  clone_48 = None
    bmm_19 = torch.ops.aten.bmm.default(view_305, _unsafe_view_47);  view_305 = _unsafe_view_47 = None
    view_306 = torch.ops.aten.view.default(bmm_19, [1, 16, 5, 384, 48]);  bmm_19 = None
    permute_53 = torch.ops.aten.permute.default(view_306, [0, 2, 3, 1, 4]);  view_306 = None
    clone_49 = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
    _unsafe_view_48 = torch.ops.aten._unsafe_view.default(clone_49, [1, 5, 384, 768]);  clone_49 = None
    t_96 = torch.ops.aten.t.default(diffusion_transformer_blocks_6_to_out_weight);  diffusion_transformer_blocks_6_to_out_weight = None
    view_307 = torch.ops.aten.view.default(_unsafe_view_48, [1920, 768]);  _unsafe_view_48 = None
    mm_84 = torch.ops.aten.mm.default(view_307, t_96);  view_307 = t_96 = None
    view_308 = torch.ops.aten.view.default(mm_84, [1, 5, 384, 768]);  mm_84 = None
    view_309 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_97 = torch.ops.aten.t.default(diffusion_transformer_blocks_6_gate_proj_0_weight);  diffusion_transformer_blocks_6_gate_proj_0_weight = None
    addmm_12 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_6_gate_proj_0_bias, view_309, t_97);  diffusion_transformer_blocks_6_gate_proj_0_bias = view_309 = t_97 = None
    view_310 = torch.ops.aten.view.default(addmm_12, [1, 5, 384, 768]);  addmm_12 = None
    sigmoid_18 = torch.ops.aten.sigmoid.default(view_310);  view_310 = None
    mul_69 = torch.ops.aten.mul.Tensor(sigmoid_18, view_308);  sigmoid_18 = view_308 = None
    native_layer_norm_default_39 = torch.ops.aten.native_layer_norm.default(add_90, [768], None, None, 0.1)
    getitem_230 = native_layer_norm_default_39[0];  native_layer_norm_default_39 = None
    t_98 = torch.ops.aten.t.default(diffusion_transformer_blocks_6_transition_ada_ln_lin_s_merged_weight);  diffusion_transformer_blocks_6_transition_ada_ln_lin_s_merged_weight = None
    view_311 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_85 = torch.ops.aten.mm.default(view_311, t_98);  view_311 = t_98 = None
    view_312 = torch.ops.aten.view.default(mm_85, [1, 5, 384, 1536]);  mm_85 = None
    split_tensor_43 = torch.ops.aten.split.Tensor(view_312, 768, dim = -1);  view_312 = None
    getitem_233 = split_tensor_43[0]
    getitem_234 = split_tensor_43[1];  split_tensor_43 = None
    add_95 = torch.ops.aten.add.Tensor(getitem_233, 1);  getitem_233 = None
    mul_70 = torch.ops.aten.mul.Tensor(getitem_230, add_95);  getitem_230 = add_95 = None
    add_96 = torch.ops.aten.add.Tensor(mul_70, getitem_234);  mul_70 = getitem_234 = None
    t_99 = torch.ops.aten.t.default(diffusion_transformer_blocks_6_transition_linear_a_nobias_double_weight);  diffusion_transformer_blocks_6_transition_linear_a_nobias_double_weight = None
    view_313 = torch.ops.aten.view.default(add_96, [1920, 768]);  add_96 = None
    mm_86 = torch.ops.aten.mm.default(view_313, t_99);  view_313 = t_99 = None
    view_314 = torch.ops.aten.view.default(mm_86, [1, 5, 384, 3072]);  mm_86 = None
    split_tensor_44 = torch.ops.aten.split.Tensor(view_314, 1536, dim = -1);  view_314 = None
    getitem_235 = split_tensor_44[0]
    getitem_236 = split_tensor_44[1];  split_tensor_44 = None
    silu_13 = torch.ops.aten.silu.default(getitem_235);  getitem_235 = None
    mul_71 = torch.ops.aten.mul.Tensor(silu_13, getitem_236);  silu_13 = getitem_236 = None
    view_315 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_100 = torch.ops.aten.t.default(diffusion_transformer_blocks_6_transition_linear_s_biasinit_m2_weight);  diffusion_transformer_blocks_6_transition_linear_s_biasinit_m2_weight = None
    addmm_13 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_6_transition_linear_s_biasinit_m2_bias, view_315, t_100);  diffusion_transformer_blocks_6_transition_linear_s_biasinit_m2_bias = view_315 = t_100 = None
    view_316 = torch.ops.aten.view.default(addmm_13, [1, 5, 384, 768]);  addmm_13 = None
    sigmoid_19 = torch.ops.aten.sigmoid.default(view_316);  view_316 = None
    t_101 = torch.ops.aten.t.default(diffusion_transformer_blocks_6_transition_linear_b_nobias_weight);  diffusion_transformer_blocks_6_transition_linear_b_nobias_weight = None
    view_317 = torch.ops.aten.view.default(mul_71, [1920, 1536]);  mul_71 = None
    mm_87 = torch.ops.aten.mm.default(view_317, t_101);  view_317 = t_101 = None
    view_318 = torch.ops.aten.view.default(mm_87, [1, 5, 384, 768]);  mm_87 = None
    mul_72 = torch.ops.aten.mul.Tensor(sigmoid_19, view_318);  sigmoid_19 = view_318 = None
    add_97 = torch.ops.aten.add.Tensor(mul_69, mul_72);  mul_69 = mul_72 = None
    add_98 = torch.ops.aten.add.Tensor(add_90, add_97);  add_90 = add_97 = None
    native_layer_norm_default_40 = torch.ops.aten.native_layer_norm.default(view_140, [256], diffusion_transformer_blocks_7_pair_layer_norm_weight, diffusion_transformer_blocks_7_pair_layer_norm_bias, 1e-05);  diffusion_transformer_blocks_7_pair_layer_norm_weight = diffusion_transformer_blocks_7_pair_layer_norm_bias = None
    getitem_237 = native_layer_norm_default_40[0];  native_layer_norm_default_40 = None
    t_102 = torch.ops.aten.t.default(diffusion_transformer_blocks_7_pair_linear_weight);  diffusion_transformer_blocks_7_pair_linear_weight = None
    view_319 = torch.ops.aten.view.default(getitem_237, [147456, 256]);  getitem_237 = None
    mm_88 = torch.ops.aten.mm.default(view_319, t_102);  view_319 = t_102 = None
    view_320 = torch.ops.aten.view.default(mm_88, [1, 1, 384, 384, 16]);  mm_88 = None
    native_layer_norm_default_41 = torch.ops.aten.native_layer_norm.default(add_98, [768], None, None, 0.1)
    getitem_240 = native_layer_norm_default_41[0];  native_layer_norm_default_41 = None
    t_103 = torch.ops.aten.t.default(diffusion_transformer_blocks_7_norm_in_lin_s_merged_weight);  diffusion_transformer_blocks_7_norm_in_lin_s_merged_weight = None
    view_321 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_89 = torch.ops.aten.mm.default(view_321, t_103);  view_321 = t_103 = None
    view_322 = torch.ops.aten.view.default(mm_89, [1, 5, 384, 1536]);  mm_89 = None
    split_tensor_45 = torch.ops.aten.split.Tensor(view_322, 768, dim = -1);  view_322 = None
    getitem_243 = split_tensor_45[0]
    getitem_244 = split_tensor_45[1];  split_tensor_45 = None
    add_99 = torch.ops.aten.add.Tensor(getitem_243, 1);  getitem_243 = None
    mul_73 = torch.ops.aten.mul.Tensor(getitem_240, add_99);  getitem_240 = add_99 = None
    add_100 = torch.ops.aten.add.Tensor(mul_73, getitem_244);  mul_73 = getitem_244 = None
    t_104 = torch.ops.aten.t.default(diffusion_transformer_blocks_7_to_qkv_weight);  diffusion_transformer_blocks_7_to_qkv_weight = None
    view_323 = torch.ops.aten.view.default(add_100, [1920, 768]);  add_100 = None
    mm_90 = torch.ops.aten.mm.default(view_323, t_104);  view_323 = t_104 = None
    view_324 = torch.ops.aten.view.default(mm_90, [1, 5, 384, 2304]);  mm_90 = None
    view_325 = torch.ops.aten.view.default(view_324, [1, 5, 384, 16, 144]);  view_324 = None
    permute_54 = torch.ops.aten.permute.default(view_325, [0, 3, 1, 2, 4]);  view_325 = None
    split_tensor_46 = torch.ops.aten.split.Tensor(permute_54, 48, dim = -1);  permute_54 = None
    getitem_245 = split_tensor_46[0]
    getitem_246 = split_tensor_46[1]
    getitem_247 = split_tensor_46[2];  split_tensor_46 = None
    view_326 = torch.ops.aten.view.default(diffusion_transformer_blocks_7_q_bias, [1, 16, 1, 1, 48]);  diffusion_transformer_blocks_7_q_bias = None
    add_101 = torch.ops.aten.add.Tensor(getitem_245, view_326);  getitem_245 = view_326 = None
    view_327 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 384, 384, 1])
    bitwise_not_20 = torch.ops.aten.bitwise_not.default(view_327);  view_327 = None
    masked_fill_20 = torch.ops.aten.masked_fill.Scalar(view_320, bitwise_not_20, -10000);  view_320 = bitwise_not_20 = None
    permute_55 = torch.ops.aten.permute.default(masked_fill_20, [0, 4, 1, 2, 3]);  masked_fill_20 = None
    view_328 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 1, 384, 384])
    bitwise_not_21 = torch.ops.aten.bitwise_not.default(view_328);  view_328 = None
    masked_fill_21 = torch.ops.aten.masked_fill.Scalar(permute_55, bitwise_not_21, -10000);  permute_55 = bitwise_not_21 = None
    mul_74 = torch.ops.aten.mul.Scalar(add_101, 0.3799178428257963);  add_101 = None
    transpose_7 = torch.ops.aten.transpose.int(getitem_246, -2, -1);  getitem_246 = None
    mul_75 = torch.ops.aten.mul.Scalar(transpose_7, 0.3799178428257963);  transpose_7 = None
    expand_42 = torch.ops.aten.expand.default(mul_74, [1, 16, 5, 384, 48]);  mul_74 = None
    clone_50 = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
    _unsafe_view_49 = torch.ops.aten._unsafe_view.default(clone_50, [80, 384, 48]);  clone_50 = None
    expand_43 = torch.ops.aten.expand.default(mul_75, [1, 16, 5, 48, 384]);  mul_75 = None
    clone_51 = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    _unsafe_view_50 = torch.ops.aten._unsafe_view.default(clone_51, [80, 48, 384]);  clone_51 = None
    bmm_20 = torch.ops.aten.bmm.default(_unsafe_view_49, _unsafe_view_50);  _unsafe_view_49 = _unsafe_view_50 = None
    view_329 = torch.ops.aten.view.default(bmm_20, [1, 16, 5, 384, 384]);  bmm_20 = None
    add_102 = torch.ops.aten.add.Tensor(view_329, masked_fill_21);  view_329 = masked_fill_21 = None
    _softmax_7 = torch.ops.aten._softmax.default(add_102, -1, False);  add_102 = None
    expand_44 = torch.ops.aten.expand.default(_softmax_7, [1, 16, 5, 384, 384]);  _softmax_7 = None
    view_330 = torch.ops.aten.view.default(expand_44, [80, 384, 384]);  expand_44 = None
    expand_45 = torch.ops.aten.expand.default(getitem_247, [1, 16, 5, 384, 48]);  getitem_247 = None
    clone_52 = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    _unsafe_view_51 = torch.ops.aten._unsafe_view.default(clone_52, [80, 384, 48]);  clone_52 = None
    bmm_21 = torch.ops.aten.bmm.default(view_330, _unsafe_view_51);  view_330 = _unsafe_view_51 = None
    view_331 = torch.ops.aten.view.default(bmm_21, [1, 16, 5, 384, 48]);  bmm_21 = None
    permute_56 = torch.ops.aten.permute.default(view_331, [0, 2, 3, 1, 4]);  view_331 = None
    clone_53 = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
    _unsafe_view_52 = torch.ops.aten._unsafe_view.default(clone_53, [1, 5, 384, 768]);  clone_53 = None
    t_105 = torch.ops.aten.t.default(diffusion_transformer_blocks_7_to_out_weight);  diffusion_transformer_blocks_7_to_out_weight = None
    view_332 = torch.ops.aten.view.default(_unsafe_view_52, [1920, 768]);  _unsafe_view_52 = None
    mm_91 = torch.ops.aten.mm.default(view_332, t_105);  view_332 = t_105 = None
    view_333 = torch.ops.aten.view.default(mm_91, [1, 5, 384, 768]);  mm_91 = None
    view_334 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_106 = torch.ops.aten.t.default(diffusion_transformer_blocks_7_gate_proj_0_weight);  diffusion_transformer_blocks_7_gate_proj_0_weight = None
    addmm_14 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_7_gate_proj_0_bias, view_334, t_106);  diffusion_transformer_blocks_7_gate_proj_0_bias = view_334 = t_106 = None
    view_335 = torch.ops.aten.view.default(addmm_14, [1, 5, 384, 768]);  addmm_14 = None
    sigmoid_20 = torch.ops.aten.sigmoid.default(view_335);  view_335 = None
    mul_76 = torch.ops.aten.mul.Tensor(sigmoid_20, view_333);  sigmoid_20 = view_333 = None
    native_layer_norm_default_42 = torch.ops.aten.native_layer_norm.default(add_98, [768], None, None, 0.1)
    getitem_248 = native_layer_norm_default_42[0];  native_layer_norm_default_42 = None
    t_107 = torch.ops.aten.t.default(diffusion_transformer_blocks_7_transition_ada_ln_lin_s_merged_weight);  diffusion_transformer_blocks_7_transition_ada_ln_lin_s_merged_weight = None
    view_336 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_92 = torch.ops.aten.mm.default(view_336, t_107);  view_336 = t_107 = None
    view_337 = torch.ops.aten.view.default(mm_92, [1, 5, 384, 1536]);  mm_92 = None
    split_tensor_47 = torch.ops.aten.split.Tensor(view_337, 768, dim = -1);  view_337 = None
    getitem_251 = split_tensor_47[0]
    getitem_252 = split_tensor_47[1];  split_tensor_47 = None
    add_103 = torch.ops.aten.add.Tensor(getitem_251, 1);  getitem_251 = None
    mul_77 = torch.ops.aten.mul.Tensor(getitem_248, add_103);  getitem_248 = add_103 = None
    add_104 = torch.ops.aten.add.Tensor(mul_77, getitem_252);  mul_77 = getitem_252 = None
    t_108 = torch.ops.aten.t.default(diffusion_transformer_blocks_7_transition_linear_a_nobias_double_weight);  diffusion_transformer_blocks_7_transition_linear_a_nobias_double_weight = None
    view_338 = torch.ops.aten.view.default(add_104, [1920, 768]);  add_104 = None
    mm_93 = torch.ops.aten.mm.default(view_338, t_108);  view_338 = t_108 = None
    view_339 = torch.ops.aten.view.default(mm_93, [1, 5, 384, 3072]);  mm_93 = None
    split_tensor_48 = torch.ops.aten.split.Tensor(view_339, 1536, dim = -1);  view_339 = None
    getitem_253 = split_tensor_48[0]
    getitem_254 = split_tensor_48[1];  split_tensor_48 = None
    silu_14 = torch.ops.aten.silu.default(getitem_253);  getitem_253 = None
    mul_78 = torch.ops.aten.mul.Tensor(silu_14, getitem_254);  silu_14 = getitem_254 = None
    view_340 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_109 = torch.ops.aten.t.default(diffusion_transformer_blocks_7_transition_linear_s_biasinit_m2_weight);  diffusion_transformer_blocks_7_transition_linear_s_biasinit_m2_weight = None
    addmm_15 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_7_transition_linear_s_biasinit_m2_bias, view_340, t_109);  diffusion_transformer_blocks_7_transition_linear_s_biasinit_m2_bias = view_340 = t_109 = None
    view_341 = torch.ops.aten.view.default(addmm_15, [1, 5, 384, 768]);  addmm_15 = None
    sigmoid_21 = torch.ops.aten.sigmoid.default(view_341);  view_341 = None
    t_110 = torch.ops.aten.t.default(diffusion_transformer_blocks_7_transition_linear_b_nobias_weight);  diffusion_transformer_blocks_7_transition_linear_b_nobias_weight = None
    view_342 = torch.ops.aten.view.default(mul_78, [1920, 1536]);  mul_78 = None
    mm_94 = torch.ops.aten.mm.default(view_342, t_110);  view_342 = t_110 = None
    view_343 = torch.ops.aten.view.default(mm_94, [1, 5, 384, 768]);  mm_94 = None
    mul_79 = torch.ops.aten.mul.Tensor(sigmoid_21, view_343);  sigmoid_21 = view_343 = None
    add_105 = torch.ops.aten.add.Tensor(mul_76, mul_79);  mul_76 = mul_79 = None
    add_106 = torch.ops.aten.add.Tensor(add_98, add_105);  add_98 = add_105 = None
    native_layer_norm_default_43 = torch.ops.aten.native_layer_norm.default(view_140, [256], diffusion_transformer_blocks_8_pair_layer_norm_weight, diffusion_transformer_blocks_8_pair_layer_norm_bias, 1e-05);  diffusion_transformer_blocks_8_pair_layer_norm_weight = diffusion_transformer_blocks_8_pair_layer_norm_bias = None
    getitem_255 = native_layer_norm_default_43[0];  native_layer_norm_default_43 = None
    t_111 = torch.ops.aten.t.default(diffusion_transformer_blocks_8_pair_linear_weight);  diffusion_transformer_blocks_8_pair_linear_weight = None
    view_344 = torch.ops.aten.view.default(getitem_255, [147456, 256]);  getitem_255 = None
    mm_95 = torch.ops.aten.mm.default(view_344, t_111);  view_344 = t_111 = None
    view_345 = torch.ops.aten.view.default(mm_95, [1, 1, 384, 384, 16]);  mm_95 = None
    native_layer_norm_default_44 = torch.ops.aten.native_layer_norm.default(add_106, [768], None, None, 0.1)
    getitem_258 = native_layer_norm_default_44[0];  native_layer_norm_default_44 = None
    t_112 = torch.ops.aten.t.default(diffusion_transformer_blocks_8_norm_in_lin_s_merged_weight);  diffusion_transformer_blocks_8_norm_in_lin_s_merged_weight = None
    view_346 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_96 = torch.ops.aten.mm.default(view_346, t_112);  view_346 = t_112 = None
    view_347 = torch.ops.aten.view.default(mm_96, [1, 5, 384, 1536]);  mm_96 = None
    split_tensor_49 = torch.ops.aten.split.Tensor(view_347, 768, dim = -1);  view_347 = None
    getitem_261 = split_tensor_49[0]
    getitem_262 = split_tensor_49[1];  split_tensor_49 = None
    add_107 = torch.ops.aten.add.Tensor(getitem_261, 1);  getitem_261 = None
    mul_80 = torch.ops.aten.mul.Tensor(getitem_258, add_107);  getitem_258 = add_107 = None
    add_108 = torch.ops.aten.add.Tensor(mul_80, getitem_262);  mul_80 = getitem_262 = None
    t_113 = torch.ops.aten.t.default(diffusion_transformer_blocks_8_to_qkv_weight);  diffusion_transformer_blocks_8_to_qkv_weight = None
    view_348 = torch.ops.aten.view.default(add_108, [1920, 768]);  add_108 = None
    mm_97 = torch.ops.aten.mm.default(view_348, t_113);  view_348 = t_113 = None
    view_349 = torch.ops.aten.view.default(mm_97, [1, 5, 384, 2304]);  mm_97 = None
    view_350 = torch.ops.aten.view.default(view_349, [1, 5, 384, 16, 144]);  view_349 = None
    permute_57 = torch.ops.aten.permute.default(view_350, [0, 3, 1, 2, 4]);  view_350 = None
    split_tensor_50 = torch.ops.aten.split.Tensor(permute_57, 48, dim = -1);  permute_57 = None
    getitem_263 = split_tensor_50[0]
    getitem_264 = split_tensor_50[1]
    getitem_265 = split_tensor_50[2];  split_tensor_50 = None
    view_351 = torch.ops.aten.view.default(diffusion_transformer_blocks_8_q_bias, [1, 16, 1, 1, 48]);  diffusion_transformer_blocks_8_q_bias = None
    add_109 = torch.ops.aten.add.Tensor(getitem_263, view_351);  getitem_263 = view_351 = None
    view_352 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 384, 384, 1])
    bitwise_not_22 = torch.ops.aten.bitwise_not.default(view_352);  view_352 = None
    masked_fill_22 = torch.ops.aten.masked_fill.Scalar(view_345, bitwise_not_22, -10000);  view_345 = bitwise_not_22 = None
    permute_58 = torch.ops.aten.permute.default(masked_fill_22, [0, 4, 1, 2, 3]);  masked_fill_22 = None
    view_353 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 1, 384, 384])
    bitwise_not_23 = torch.ops.aten.bitwise_not.default(view_353);  view_353 = None
    masked_fill_23 = torch.ops.aten.masked_fill.Scalar(permute_58, bitwise_not_23, -10000);  permute_58 = bitwise_not_23 = None
    mul_81 = torch.ops.aten.mul.Scalar(add_109, 0.3799178428257963);  add_109 = None
    transpose_8 = torch.ops.aten.transpose.int(getitem_264, -2, -1);  getitem_264 = None
    mul_82 = torch.ops.aten.mul.Scalar(transpose_8, 0.3799178428257963);  transpose_8 = None
    expand_46 = torch.ops.aten.expand.default(mul_81, [1, 16, 5, 384, 48]);  mul_81 = None
    clone_54 = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
    _unsafe_view_53 = torch.ops.aten._unsafe_view.default(clone_54, [80, 384, 48]);  clone_54 = None
    expand_47 = torch.ops.aten.expand.default(mul_82, [1, 16, 5, 48, 384]);  mul_82 = None
    clone_55 = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    _unsafe_view_54 = torch.ops.aten._unsafe_view.default(clone_55, [80, 48, 384]);  clone_55 = None
    bmm_22 = torch.ops.aten.bmm.default(_unsafe_view_53, _unsafe_view_54);  _unsafe_view_53 = _unsafe_view_54 = None
    view_354 = torch.ops.aten.view.default(bmm_22, [1, 16, 5, 384, 384]);  bmm_22 = None
    add_110 = torch.ops.aten.add.Tensor(view_354, masked_fill_23);  view_354 = masked_fill_23 = None
    _softmax_8 = torch.ops.aten._softmax.default(add_110, -1, False);  add_110 = None
    expand_48 = torch.ops.aten.expand.default(_softmax_8, [1, 16, 5, 384, 384]);  _softmax_8 = None
    view_355 = torch.ops.aten.view.default(expand_48, [80, 384, 384]);  expand_48 = None
    expand_49 = torch.ops.aten.expand.default(getitem_265, [1, 16, 5, 384, 48]);  getitem_265 = None
    clone_56 = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    _unsafe_view_55 = torch.ops.aten._unsafe_view.default(clone_56, [80, 384, 48]);  clone_56 = None
    bmm_23 = torch.ops.aten.bmm.default(view_355, _unsafe_view_55);  view_355 = _unsafe_view_55 = None
    view_356 = torch.ops.aten.view.default(bmm_23, [1, 16, 5, 384, 48]);  bmm_23 = None
    permute_59 = torch.ops.aten.permute.default(view_356, [0, 2, 3, 1, 4]);  view_356 = None
    clone_57 = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    _unsafe_view_56 = torch.ops.aten._unsafe_view.default(clone_57, [1, 5, 384, 768]);  clone_57 = None
    t_114 = torch.ops.aten.t.default(diffusion_transformer_blocks_8_to_out_weight);  diffusion_transformer_blocks_8_to_out_weight = None
    view_357 = torch.ops.aten.view.default(_unsafe_view_56, [1920, 768]);  _unsafe_view_56 = None
    mm_98 = torch.ops.aten.mm.default(view_357, t_114);  view_357 = t_114 = None
    view_358 = torch.ops.aten.view.default(mm_98, [1, 5, 384, 768]);  mm_98 = None
    view_359 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_115 = torch.ops.aten.t.default(diffusion_transformer_blocks_8_gate_proj_0_weight);  diffusion_transformer_blocks_8_gate_proj_0_weight = None
    addmm_16 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_8_gate_proj_0_bias, view_359, t_115);  diffusion_transformer_blocks_8_gate_proj_0_bias = view_359 = t_115 = None
    view_360 = torch.ops.aten.view.default(addmm_16, [1, 5, 384, 768]);  addmm_16 = None
    sigmoid_22 = torch.ops.aten.sigmoid.default(view_360);  view_360 = None
    mul_83 = torch.ops.aten.mul.Tensor(sigmoid_22, view_358);  sigmoid_22 = view_358 = None
    native_layer_norm_default_45 = torch.ops.aten.native_layer_norm.default(add_106, [768], None, None, 0.1)
    getitem_266 = native_layer_norm_default_45[0];  native_layer_norm_default_45 = None
    t_116 = torch.ops.aten.t.default(diffusion_transformer_blocks_8_transition_ada_ln_lin_s_merged_weight);  diffusion_transformer_blocks_8_transition_ada_ln_lin_s_merged_weight = None
    view_361 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_99 = torch.ops.aten.mm.default(view_361, t_116);  view_361 = t_116 = None
    view_362 = torch.ops.aten.view.default(mm_99, [1, 5, 384, 1536]);  mm_99 = None
    split_tensor_51 = torch.ops.aten.split.Tensor(view_362, 768, dim = -1);  view_362 = None
    getitem_269 = split_tensor_51[0]
    getitem_270 = split_tensor_51[1];  split_tensor_51 = None
    add_111 = torch.ops.aten.add.Tensor(getitem_269, 1);  getitem_269 = None
    mul_84 = torch.ops.aten.mul.Tensor(getitem_266, add_111);  getitem_266 = add_111 = None
    add_112 = torch.ops.aten.add.Tensor(mul_84, getitem_270);  mul_84 = getitem_270 = None
    t_117 = torch.ops.aten.t.default(diffusion_transformer_blocks_8_transition_linear_a_nobias_double_weight);  diffusion_transformer_blocks_8_transition_linear_a_nobias_double_weight = None
    view_363 = torch.ops.aten.view.default(add_112, [1920, 768]);  add_112 = None
    mm_100 = torch.ops.aten.mm.default(view_363, t_117);  view_363 = t_117 = None
    view_364 = torch.ops.aten.view.default(mm_100, [1, 5, 384, 3072]);  mm_100 = None
    split_tensor_52 = torch.ops.aten.split.Tensor(view_364, 1536, dim = -1);  view_364 = None
    getitem_271 = split_tensor_52[0]
    getitem_272 = split_tensor_52[1];  split_tensor_52 = None
    silu_15 = torch.ops.aten.silu.default(getitem_271);  getitem_271 = None
    mul_85 = torch.ops.aten.mul.Tensor(silu_15, getitem_272);  silu_15 = getitem_272 = None
    view_365 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_118 = torch.ops.aten.t.default(diffusion_transformer_blocks_8_transition_linear_s_biasinit_m2_weight);  diffusion_transformer_blocks_8_transition_linear_s_biasinit_m2_weight = None
    addmm_17 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_8_transition_linear_s_biasinit_m2_bias, view_365, t_118);  diffusion_transformer_blocks_8_transition_linear_s_biasinit_m2_bias = view_365 = t_118 = None
    view_366 = torch.ops.aten.view.default(addmm_17, [1, 5, 384, 768]);  addmm_17 = None
    sigmoid_23 = torch.ops.aten.sigmoid.default(view_366);  view_366 = None
    t_119 = torch.ops.aten.t.default(diffusion_transformer_blocks_8_transition_linear_b_nobias_weight);  diffusion_transformer_blocks_8_transition_linear_b_nobias_weight = None
    view_367 = torch.ops.aten.view.default(mul_85, [1920, 1536]);  mul_85 = None
    mm_101 = torch.ops.aten.mm.default(view_367, t_119);  view_367 = t_119 = None
    view_368 = torch.ops.aten.view.default(mm_101, [1, 5, 384, 768]);  mm_101 = None
    mul_86 = torch.ops.aten.mul.Tensor(sigmoid_23, view_368);  sigmoid_23 = view_368 = None
    add_113 = torch.ops.aten.add.Tensor(mul_83, mul_86);  mul_83 = mul_86 = None
    add_114 = torch.ops.aten.add.Tensor(add_106, add_113);  add_106 = add_113 = None
    native_layer_norm_default_46 = torch.ops.aten.native_layer_norm.default(view_140, [256], diffusion_transformer_blocks_9_pair_layer_norm_weight, diffusion_transformer_blocks_9_pair_layer_norm_bias, 1e-05);  diffusion_transformer_blocks_9_pair_layer_norm_weight = diffusion_transformer_blocks_9_pair_layer_norm_bias = None
    getitem_273 = native_layer_norm_default_46[0];  native_layer_norm_default_46 = None
    t_120 = torch.ops.aten.t.default(diffusion_transformer_blocks_9_pair_linear_weight);  diffusion_transformer_blocks_9_pair_linear_weight = None
    view_369 = torch.ops.aten.view.default(getitem_273, [147456, 256]);  getitem_273 = None
    mm_102 = torch.ops.aten.mm.default(view_369, t_120);  view_369 = t_120 = None
    view_370 = torch.ops.aten.view.default(mm_102, [1, 1, 384, 384, 16]);  mm_102 = None
    native_layer_norm_default_47 = torch.ops.aten.native_layer_norm.default(add_114, [768], None, None, 0.1)
    getitem_276 = native_layer_norm_default_47[0];  native_layer_norm_default_47 = None
    t_121 = torch.ops.aten.t.default(diffusion_transformer_blocks_9_norm_in_lin_s_merged_weight);  diffusion_transformer_blocks_9_norm_in_lin_s_merged_weight = None
    view_371 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_103 = torch.ops.aten.mm.default(view_371, t_121);  view_371 = t_121 = None
    view_372 = torch.ops.aten.view.default(mm_103, [1, 5, 384, 1536]);  mm_103 = None
    split_tensor_53 = torch.ops.aten.split.Tensor(view_372, 768, dim = -1);  view_372 = None
    getitem_279 = split_tensor_53[0]
    getitem_280 = split_tensor_53[1];  split_tensor_53 = None
    add_115 = torch.ops.aten.add.Tensor(getitem_279, 1);  getitem_279 = None
    mul_87 = torch.ops.aten.mul.Tensor(getitem_276, add_115);  getitem_276 = add_115 = None
    add_116 = torch.ops.aten.add.Tensor(mul_87, getitem_280);  mul_87 = getitem_280 = None
    t_122 = torch.ops.aten.t.default(diffusion_transformer_blocks_9_to_qkv_weight);  diffusion_transformer_blocks_9_to_qkv_weight = None
    view_373 = torch.ops.aten.view.default(add_116, [1920, 768]);  add_116 = None
    mm_104 = torch.ops.aten.mm.default(view_373, t_122);  view_373 = t_122 = None
    view_374 = torch.ops.aten.view.default(mm_104, [1, 5, 384, 2304]);  mm_104 = None
    view_375 = torch.ops.aten.view.default(view_374, [1, 5, 384, 16, 144]);  view_374 = None
    permute_60 = torch.ops.aten.permute.default(view_375, [0, 3, 1, 2, 4]);  view_375 = None
    split_tensor_54 = torch.ops.aten.split.Tensor(permute_60, 48, dim = -1);  permute_60 = None
    getitem_281 = split_tensor_54[0]
    getitem_282 = split_tensor_54[1]
    getitem_283 = split_tensor_54[2];  split_tensor_54 = None
    view_376 = torch.ops.aten.view.default(diffusion_transformer_blocks_9_q_bias, [1, 16, 1, 1, 48]);  diffusion_transformer_blocks_9_q_bias = None
    add_117 = torch.ops.aten.add.Tensor(getitem_281, view_376);  getitem_281 = view_376 = None
    view_377 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 384, 384, 1])
    bitwise_not_24 = torch.ops.aten.bitwise_not.default(view_377);  view_377 = None
    masked_fill_24 = torch.ops.aten.masked_fill.Scalar(view_370, bitwise_not_24, -10000);  view_370 = bitwise_not_24 = None
    permute_61 = torch.ops.aten.permute.default(masked_fill_24, [0, 4, 1, 2, 3]);  masked_fill_24 = None
    view_378 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 1, 384, 384])
    bitwise_not_25 = torch.ops.aten.bitwise_not.default(view_378);  view_378 = None
    masked_fill_25 = torch.ops.aten.masked_fill.Scalar(permute_61, bitwise_not_25, -10000);  permute_61 = bitwise_not_25 = None
    mul_88 = torch.ops.aten.mul.Scalar(add_117, 0.3799178428257963);  add_117 = None
    transpose_9 = torch.ops.aten.transpose.int(getitem_282, -2, -1);  getitem_282 = None
    mul_89 = torch.ops.aten.mul.Scalar(transpose_9, 0.3799178428257963);  transpose_9 = None
    expand_50 = torch.ops.aten.expand.default(mul_88, [1, 16, 5, 384, 48]);  mul_88 = None
    clone_58 = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
    _unsafe_view_57 = torch.ops.aten._unsafe_view.default(clone_58, [80, 384, 48]);  clone_58 = None
    expand_51 = torch.ops.aten.expand.default(mul_89, [1, 16, 5, 48, 384]);  mul_89 = None
    clone_59 = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    _unsafe_view_58 = torch.ops.aten._unsafe_view.default(clone_59, [80, 48, 384]);  clone_59 = None
    bmm_24 = torch.ops.aten.bmm.default(_unsafe_view_57, _unsafe_view_58);  _unsafe_view_57 = _unsafe_view_58 = None
    view_379 = torch.ops.aten.view.default(bmm_24, [1, 16, 5, 384, 384]);  bmm_24 = None
    add_118 = torch.ops.aten.add.Tensor(view_379, masked_fill_25);  view_379 = masked_fill_25 = None
    _softmax_9 = torch.ops.aten._softmax.default(add_118, -1, False);  add_118 = None
    expand_52 = torch.ops.aten.expand.default(_softmax_9, [1, 16, 5, 384, 384]);  _softmax_9 = None
    view_380 = torch.ops.aten.view.default(expand_52, [80, 384, 384]);  expand_52 = None
    expand_53 = torch.ops.aten.expand.default(getitem_283, [1, 16, 5, 384, 48]);  getitem_283 = None
    clone_60 = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    _unsafe_view_59 = torch.ops.aten._unsafe_view.default(clone_60, [80, 384, 48]);  clone_60 = None
    bmm_25 = torch.ops.aten.bmm.default(view_380, _unsafe_view_59);  view_380 = _unsafe_view_59 = None
    view_381 = torch.ops.aten.view.default(bmm_25, [1, 16, 5, 384, 48]);  bmm_25 = None
    permute_62 = torch.ops.aten.permute.default(view_381, [0, 2, 3, 1, 4]);  view_381 = None
    clone_61 = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    _unsafe_view_60 = torch.ops.aten._unsafe_view.default(clone_61, [1, 5, 384, 768]);  clone_61 = None
    t_123 = torch.ops.aten.t.default(diffusion_transformer_blocks_9_to_out_weight);  diffusion_transformer_blocks_9_to_out_weight = None
    view_382 = torch.ops.aten.view.default(_unsafe_view_60, [1920, 768]);  _unsafe_view_60 = None
    mm_105 = torch.ops.aten.mm.default(view_382, t_123);  view_382 = t_123 = None
    view_383 = torch.ops.aten.view.default(mm_105, [1, 5, 384, 768]);  mm_105 = None
    view_384 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_124 = torch.ops.aten.t.default(diffusion_transformer_blocks_9_gate_proj_0_weight);  diffusion_transformer_blocks_9_gate_proj_0_weight = None
    addmm_18 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_9_gate_proj_0_bias, view_384, t_124);  diffusion_transformer_blocks_9_gate_proj_0_bias = view_384 = t_124 = None
    view_385 = torch.ops.aten.view.default(addmm_18, [1, 5, 384, 768]);  addmm_18 = None
    sigmoid_24 = torch.ops.aten.sigmoid.default(view_385);  view_385 = None
    mul_90 = torch.ops.aten.mul.Tensor(sigmoid_24, view_383);  sigmoid_24 = view_383 = None
    native_layer_norm_default_48 = torch.ops.aten.native_layer_norm.default(add_114, [768], None, None, 0.1)
    getitem_284 = native_layer_norm_default_48[0];  native_layer_norm_default_48 = None
    t_125 = torch.ops.aten.t.default(diffusion_transformer_blocks_9_transition_ada_ln_lin_s_merged_weight);  diffusion_transformer_blocks_9_transition_ada_ln_lin_s_merged_weight = None
    view_386 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_106 = torch.ops.aten.mm.default(view_386, t_125);  view_386 = t_125 = None
    view_387 = torch.ops.aten.view.default(mm_106, [1, 5, 384, 1536]);  mm_106 = None
    split_tensor_55 = torch.ops.aten.split.Tensor(view_387, 768, dim = -1);  view_387 = None
    getitem_287 = split_tensor_55[0]
    getitem_288 = split_tensor_55[1];  split_tensor_55 = None
    add_119 = torch.ops.aten.add.Tensor(getitem_287, 1);  getitem_287 = None
    mul_91 = torch.ops.aten.mul.Tensor(getitem_284, add_119);  getitem_284 = add_119 = None
    add_120 = torch.ops.aten.add.Tensor(mul_91, getitem_288);  mul_91 = getitem_288 = None
    t_126 = torch.ops.aten.t.default(diffusion_transformer_blocks_9_transition_linear_a_nobias_double_weight);  diffusion_transformer_blocks_9_transition_linear_a_nobias_double_weight = None
    view_388 = torch.ops.aten.view.default(add_120, [1920, 768]);  add_120 = None
    mm_107 = torch.ops.aten.mm.default(view_388, t_126);  view_388 = t_126 = None
    view_389 = torch.ops.aten.view.default(mm_107, [1, 5, 384, 3072]);  mm_107 = None
    split_tensor_56 = torch.ops.aten.split.Tensor(view_389, 1536, dim = -1);  view_389 = None
    getitem_289 = split_tensor_56[0]
    getitem_290 = split_tensor_56[1];  split_tensor_56 = None
    silu_16 = torch.ops.aten.silu.default(getitem_289);  getitem_289 = None
    mul_92 = torch.ops.aten.mul.Tensor(silu_16, getitem_290);  silu_16 = getitem_290 = None
    view_390 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_127 = torch.ops.aten.t.default(diffusion_transformer_blocks_9_transition_linear_s_biasinit_m2_weight);  diffusion_transformer_blocks_9_transition_linear_s_biasinit_m2_weight = None
    addmm_19 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_9_transition_linear_s_biasinit_m2_bias, view_390, t_127);  diffusion_transformer_blocks_9_transition_linear_s_biasinit_m2_bias = view_390 = t_127 = None
    view_391 = torch.ops.aten.view.default(addmm_19, [1, 5, 384, 768]);  addmm_19 = None
    sigmoid_25 = torch.ops.aten.sigmoid.default(view_391);  view_391 = None
    t_128 = torch.ops.aten.t.default(diffusion_transformer_blocks_9_transition_linear_b_nobias_weight);  diffusion_transformer_blocks_9_transition_linear_b_nobias_weight = None
    view_392 = torch.ops.aten.view.default(mul_92, [1920, 1536]);  mul_92 = None
    mm_108 = torch.ops.aten.mm.default(view_392, t_128);  view_392 = t_128 = None
    view_393 = torch.ops.aten.view.default(mm_108, [1, 5, 384, 768]);  mm_108 = None
    mul_93 = torch.ops.aten.mul.Tensor(sigmoid_25, view_393);  sigmoid_25 = view_393 = None
    add_121 = torch.ops.aten.add.Tensor(mul_90, mul_93);  mul_90 = mul_93 = None
    add_122 = torch.ops.aten.add.Tensor(add_114, add_121);  add_114 = add_121 = None
    native_layer_norm_default_49 = torch.ops.aten.native_layer_norm.default(view_140, [256], diffusion_transformer_blocks_10_pair_layer_norm_weight, diffusion_transformer_blocks_10_pair_layer_norm_bias, 1e-05);  diffusion_transformer_blocks_10_pair_layer_norm_weight = diffusion_transformer_blocks_10_pair_layer_norm_bias = None
    getitem_291 = native_layer_norm_default_49[0];  native_layer_norm_default_49 = None
    t_129 = torch.ops.aten.t.default(diffusion_transformer_blocks_10_pair_linear_weight);  diffusion_transformer_blocks_10_pair_linear_weight = None
    view_394 = torch.ops.aten.view.default(getitem_291, [147456, 256]);  getitem_291 = None
    mm_109 = torch.ops.aten.mm.default(view_394, t_129);  view_394 = t_129 = None
    view_395 = torch.ops.aten.view.default(mm_109, [1, 1, 384, 384, 16]);  mm_109 = None
    native_layer_norm_default_50 = torch.ops.aten.native_layer_norm.default(add_122, [768], None, None, 0.1)
    getitem_294 = native_layer_norm_default_50[0];  native_layer_norm_default_50 = None
    t_130 = torch.ops.aten.t.default(diffusion_transformer_blocks_10_norm_in_lin_s_merged_weight);  diffusion_transformer_blocks_10_norm_in_lin_s_merged_weight = None
    view_396 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_110 = torch.ops.aten.mm.default(view_396, t_130);  view_396 = t_130 = None
    view_397 = torch.ops.aten.view.default(mm_110, [1, 5, 384, 1536]);  mm_110 = None
    split_tensor_57 = torch.ops.aten.split.Tensor(view_397, 768, dim = -1);  view_397 = None
    getitem_297 = split_tensor_57[0]
    getitem_298 = split_tensor_57[1];  split_tensor_57 = None
    add_123 = torch.ops.aten.add.Tensor(getitem_297, 1);  getitem_297 = None
    mul_94 = torch.ops.aten.mul.Tensor(getitem_294, add_123);  getitem_294 = add_123 = None
    add_124 = torch.ops.aten.add.Tensor(mul_94, getitem_298);  mul_94 = getitem_298 = None
    t_131 = torch.ops.aten.t.default(diffusion_transformer_blocks_10_to_qkv_weight);  diffusion_transformer_blocks_10_to_qkv_weight = None
    view_398 = torch.ops.aten.view.default(add_124, [1920, 768]);  add_124 = None
    mm_111 = torch.ops.aten.mm.default(view_398, t_131);  view_398 = t_131 = None
    view_399 = torch.ops.aten.view.default(mm_111, [1, 5, 384, 2304]);  mm_111 = None
    view_400 = torch.ops.aten.view.default(view_399, [1, 5, 384, 16, 144]);  view_399 = None
    permute_63 = torch.ops.aten.permute.default(view_400, [0, 3, 1, 2, 4]);  view_400 = None
    split_tensor_58 = torch.ops.aten.split.Tensor(permute_63, 48, dim = -1);  permute_63 = None
    getitem_299 = split_tensor_58[0]
    getitem_300 = split_tensor_58[1]
    getitem_301 = split_tensor_58[2];  split_tensor_58 = None
    view_401 = torch.ops.aten.view.default(diffusion_transformer_blocks_10_q_bias, [1, 16, 1, 1, 48]);  diffusion_transformer_blocks_10_q_bias = None
    add_125 = torch.ops.aten.add.Tensor(getitem_299, view_401);  getitem_299 = view_401 = None
    view_402 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 384, 384, 1])
    bitwise_not_26 = torch.ops.aten.bitwise_not.default(view_402);  view_402 = None
    masked_fill_26 = torch.ops.aten.masked_fill.Scalar(view_395, bitwise_not_26, -10000);  view_395 = bitwise_not_26 = None
    permute_64 = torch.ops.aten.permute.default(masked_fill_26, [0, 4, 1, 2, 3]);  masked_fill_26 = None
    view_403 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 1, 384, 384])
    bitwise_not_27 = torch.ops.aten.bitwise_not.default(view_403);  view_403 = None
    masked_fill_27 = torch.ops.aten.masked_fill.Scalar(permute_64, bitwise_not_27, -10000);  permute_64 = bitwise_not_27 = None
    mul_95 = torch.ops.aten.mul.Scalar(add_125, 0.3799178428257963);  add_125 = None
    transpose_10 = torch.ops.aten.transpose.int(getitem_300, -2, -1);  getitem_300 = None
    mul_96 = torch.ops.aten.mul.Scalar(transpose_10, 0.3799178428257963);  transpose_10 = None
    expand_54 = torch.ops.aten.expand.default(mul_95, [1, 16, 5, 384, 48]);  mul_95 = None
    clone_62 = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
    _unsafe_view_61 = torch.ops.aten._unsafe_view.default(clone_62, [80, 384, 48]);  clone_62 = None
    expand_55 = torch.ops.aten.expand.default(mul_96, [1, 16, 5, 48, 384]);  mul_96 = None
    clone_63 = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
    _unsafe_view_62 = torch.ops.aten._unsafe_view.default(clone_63, [80, 48, 384]);  clone_63 = None
    bmm_26 = torch.ops.aten.bmm.default(_unsafe_view_61, _unsafe_view_62);  _unsafe_view_61 = _unsafe_view_62 = None
    view_404 = torch.ops.aten.view.default(bmm_26, [1, 16, 5, 384, 384]);  bmm_26 = None
    add_126 = torch.ops.aten.add.Tensor(view_404, masked_fill_27);  view_404 = masked_fill_27 = None
    _softmax_10 = torch.ops.aten._softmax.default(add_126, -1, False);  add_126 = None
    expand_56 = torch.ops.aten.expand.default(_softmax_10, [1, 16, 5, 384, 384]);  _softmax_10 = None
    view_405 = torch.ops.aten.view.default(expand_56, [80, 384, 384]);  expand_56 = None
    expand_57 = torch.ops.aten.expand.default(getitem_301, [1, 16, 5, 384, 48]);  getitem_301 = None
    clone_64 = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    _unsafe_view_63 = torch.ops.aten._unsafe_view.default(clone_64, [80, 384, 48]);  clone_64 = None
    bmm_27 = torch.ops.aten.bmm.default(view_405, _unsafe_view_63);  view_405 = _unsafe_view_63 = None
    view_406 = torch.ops.aten.view.default(bmm_27, [1, 16, 5, 384, 48]);  bmm_27 = None
    permute_65 = torch.ops.aten.permute.default(view_406, [0, 2, 3, 1, 4]);  view_406 = None
    clone_65 = torch.ops.aten.clone.default(permute_65, memory_format = torch.contiguous_format);  permute_65 = None
    _unsafe_view_64 = torch.ops.aten._unsafe_view.default(clone_65, [1, 5, 384, 768]);  clone_65 = None
    t_132 = torch.ops.aten.t.default(diffusion_transformer_blocks_10_to_out_weight);  diffusion_transformer_blocks_10_to_out_weight = None
    view_407 = torch.ops.aten.view.default(_unsafe_view_64, [1920, 768]);  _unsafe_view_64 = None
    mm_112 = torch.ops.aten.mm.default(view_407, t_132);  view_407 = t_132 = None
    view_408 = torch.ops.aten.view.default(mm_112, [1, 5, 384, 768]);  mm_112 = None
    view_409 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_133 = torch.ops.aten.t.default(diffusion_transformer_blocks_10_gate_proj_0_weight);  diffusion_transformer_blocks_10_gate_proj_0_weight = None
    addmm_20 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_10_gate_proj_0_bias, view_409, t_133);  diffusion_transformer_blocks_10_gate_proj_0_bias = view_409 = t_133 = None
    view_410 = torch.ops.aten.view.default(addmm_20, [1, 5, 384, 768]);  addmm_20 = None
    sigmoid_26 = torch.ops.aten.sigmoid.default(view_410);  view_410 = None
    mul_97 = torch.ops.aten.mul.Tensor(sigmoid_26, view_408);  sigmoid_26 = view_408 = None
    native_layer_norm_default_51 = torch.ops.aten.native_layer_norm.default(add_122, [768], None, None, 0.1)
    getitem_302 = native_layer_norm_default_51[0];  native_layer_norm_default_51 = None
    t_134 = torch.ops.aten.t.default(diffusion_transformer_blocks_10_transition_ada_ln_lin_s_merged_weight);  diffusion_transformer_blocks_10_transition_ada_ln_lin_s_merged_weight = None
    view_411 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_113 = torch.ops.aten.mm.default(view_411, t_134);  view_411 = t_134 = None
    view_412 = torch.ops.aten.view.default(mm_113, [1, 5, 384, 1536]);  mm_113 = None
    split_tensor_59 = torch.ops.aten.split.Tensor(view_412, 768, dim = -1);  view_412 = None
    getitem_305 = split_tensor_59[0]
    getitem_306 = split_tensor_59[1];  split_tensor_59 = None
    add_127 = torch.ops.aten.add.Tensor(getitem_305, 1);  getitem_305 = None
    mul_98 = torch.ops.aten.mul.Tensor(getitem_302, add_127);  getitem_302 = add_127 = None
    add_128 = torch.ops.aten.add.Tensor(mul_98, getitem_306);  mul_98 = getitem_306 = None
    t_135 = torch.ops.aten.t.default(diffusion_transformer_blocks_10_transition_linear_a_nobias_double_weight);  diffusion_transformer_blocks_10_transition_linear_a_nobias_double_weight = None
    view_413 = torch.ops.aten.view.default(add_128, [1920, 768]);  add_128 = None
    mm_114 = torch.ops.aten.mm.default(view_413, t_135);  view_413 = t_135 = None
    view_414 = torch.ops.aten.view.default(mm_114, [1, 5, 384, 3072]);  mm_114 = None
    split_tensor_60 = torch.ops.aten.split.Tensor(view_414, 1536, dim = -1);  view_414 = None
    getitem_307 = split_tensor_60[0]
    getitem_308 = split_tensor_60[1];  split_tensor_60 = None
    silu_17 = torch.ops.aten.silu.default(getitem_307);  getitem_307 = None
    mul_99 = torch.ops.aten.mul.Tensor(silu_17, getitem_308);  silu_17 = getitem_308 = None
    view_415 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_136 = torch.ops.aten.t.default(diffusion_transformer_blocks_10_transition_linear_s_biasinit_m2_weight);  diffusion_transformer_blocks_10_transition_linear_s_biasinit_m2_weight = None
    addmm_21 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_10_transition_linear_s_biasinit_m2_bias, view_415, t_136);  diffusion_transformer_blocks_10_transition_linear_s_biasinit_m2_bias = view_415 = t_136 = None
    view_416 = torch.ops.aten.view.default(addmm_21, [1, 5, 384, 768]);  addmm_21 = None
    sigmoid_27 = torch.ops.aten.sigmoid.default(view_416);  view_416 = None
    t_137 = torch.ops.aten.t.default(diffusion_transformer_blocks_10_transition_linear_b_nobias_weight);  diffusion_transformer_blocks_10_transition_linear_b_nobias_weight = None
    view_417 = torch.ops.aten.view.default(mul_99, [1920, 1536]);  mul_99 = None
    mm_115 = torch.ops.aten.mm.default(view_417, t_137);  view_417 = t_137 = None
    view_418 = torch.ops.aten.view.default(mm_115, [1, 5, 384, 768]);  mm_115 = None
    mul_100 = torch.ops.aten.mul.Tensor(sigmoid_27, view_418);  sigmoid_27 = view_418 = None
    add_129 = torch.ops.aten.add.Tensor(mul_97, mul_100);  mul_97 = mul_100 = None
    add_130 = torch.ops.aten.add.Tensor(add_122, add_129);  add_122 = add_129 = None
    native_layer_norm_default_52 = torch.ops.aten.native_layer_norm.default(view_140, [256], diffusion_transformer_blocks_11_pair_layer_norm_weight, diffusion_transformer_blocks_11_pair_layer_norm_bias, 1e-05);  diffusion_transformer_blocks_11_pair_layer_norm_weight = diffusion_transformer_blocks_11_pair_layer_norm_bias = None
    getitem_309 = native_layer_norm_default_52[0];  native_layer_norm_default_52 = None
    t_138 = torch.ops.aten.t.default(diffusion_transformer_blocks_11_pair_linear_weight);  diffusion_transformer_blocks_11_pair_linear_weight = None
    view_419 = torch.ops.aten.view.default(getitem_309, [147456, 256]);  getitem_309 = None
    mm_116 = torch.ops.aten.mm.default(view_419, t_138);  view_419 = t_138 = None
    view_420 = torch.ops.aten.view.default(mm_116, [1, 1, 384, 384, 16]);  mm_116 = None
    native_layer_norm_default_53 = torch.ops.aten.native_layer_norm.default(add_130, [768], None, None, 0.1)
    getitem_312 = native_layer_norm_default_53[0];  native_layer_norm_default_53 = None
    t_139 = torch.ops.aten.t.default(diffusion_transformer_blocks_11_norm_in_lin_s_merged_weight);  diffusion_transformer_blocks_11_norm_in_lin_s_merged_weight = None
    view_421 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_117 = torch.ops.aten.mm.default(view_421, t_139);  view_421 = t_139 = None
    view_422 = torch.ops.aten.view.default(mm_117, [1, 5, 384, 1536]);  mm_117 = None
    split_tensor_61 = torch.ops.aten.split.Tensor(view_422, 768, dim = -1);  view_422 = None
    getitem_315 = split_tensor_61[0]
    getitem_316 = split_tensor_61[1];  split_tensor_61 = None
    add_131 = torch.ops.aten.add.Tensor(getitem_315, 1);  getitem_315 = None
    mul_101 = torch.ops.aten.mul.Tensor(getitem_312, add_131);  getitem_312 = add_131 = None
    add_132 = torch.ops.aten.add.Tensor(mul_101, getitem_316);  mul_101 = getitem_316 = None
    t_140 = torch.ops.aten.t.default(diffusion_transformer_blocks_11_to_qkv_weight);  diffusion_transformer_blocks_11_to_qkv_weight = None
    view_423 = torch.ops.aten.view.default(add_132, [1920, 768]);  add_132 = None
    mm_118 = torch.ops.aten.mm.default(view_423, t_140);  view_423 = t_140 = None
    view_424 = torch.ops.aten.view.default(mm_118, [1, 5, 384, 2304]);  mm_118 = None
    view_425 = torch.ops.aten.view.default(view_424, [1, 5, 384, 16, 144]);  view_424 = None
    permute_66 = torch.ops.aten.permute.default(view_425, [0, 3, 1, 2, 4]);  view_425 = None
    split_tensor_62 = torch.ops.aten.split.Tensor(permute_66, 48, dim = -1);  permute_66 = None
    getitem_317 = split_tensor_62[0]
    getitem_318 = split_tensor_62[1]
    getitem_319 = split_tensor_62[2];  split_tensor_62 = None
    view_426 = torch.ops.aten.view.default(diffusion_transformer_blocks_11_q_bias, [1, 16, 1, 1, 48]);  diffusion_transformer_blocks_11_q_bias = None
    add_133 = torch.ops.aten.add.Tensor(getitem_317, view_426);  getitem_317 = view_426 = None
    view_427 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 384, 384, 1])
    bitwise_not_28 = torch.ops.aten.bitwise_not.default(view_427);  view_427 = None
    masked_fill_28 = torch.ops.aten.masked_fill.Scalar(view_420, bitwise_not_28, -10000);  view_420 = bitwise_not_28 = None
    permute_67 = torch.ops.aten.permute.default(masked_fill_28, [0, 4, 1, 2, 3]);  masked_fill_28 = None
    view_428 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 1, 384, 384])
    bitwise_not_29 = torch.ops.aten.bitwise_not.default(view_428);  view_428 = None
    masked_fill_29 = torch.ops.aten.masked_fill.Scalar(permute_67, bitwise_not_29, -10000);  permute_67 = bitwise_not_29 = None
    mul_102 = torch.ops.aten.mul.Scalar(add_133, 0.3799178428257963);  add_133 = None
    transpose_11 = torch.ops.aten.transpose.int(getitem_318, -2, -1);  getitem_318 = None
    mul_103 = torch.ops.aten.mul.Scalar(transpose_11, 0.3799178428257963);  transpose_11 = None
    expand_58 = torch.ops.aten.expand.default(mul_102, [1, 16, 5, 384, 48]);  mul_102 = None
    clone_66 = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
    _unsafe_view_65 = torch.ops.aten._unsafe_view.default(clone_66, [80, 384, 48]);  clone_66 = None
    expand_59 = torch.ops.aten.expand.default(mul_103, [1, 16, 5, 48, 384]);  mul_103 = None
    clone_67 = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
    _unsafe_view_66 = torch.ops.aten._unsafe_view.default(clone_67, [80, 48, 384]);  clone_67 = None
    bmm_28 = torch.ops.aten.bmm.default(_unsafe_view_65, _unsafe_view_66);  _unsafe_view_65 = _unsafe_view_66 = None
    view_429 = torch.ops.aten.view.default(bmm_28, [1, 16, 5, 384, 384]);  bmm_28 = None
    add_134 = torch.ops.aten.add.Tensor(view_429, masked_fill_29);  view_429 = masked_fill_29 = None
    _softmax_11 = torch.ops.aten._softmax.default(add_134, -1, False);  add_134 = None
    expand_60 = torch.ops.aten.expand.default(_softmax_11, [1, 16, 5, 384, 384]);  _softmax_11 = None
    view_430 = torch.ops.aten.view.default(expand_60, [80, 384, 384]);  expand_60 = None
    expand_61 = torch.ops.aten.expand.default(getitem_319, [1, 16, 5, 384, 48]);  getitem_319 = None
    clone_68 = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
    _unsafe_view_67 = torch.ops.aten._unsafe_view.default(clone_68, [80, 384, 48]);  clone_68 = None
    bmm_29 = torch.ops.aten.bmm.default(view_430, _unsafe_view_67);  view_430 = _unsafe_view_67 = None
    view_431 = torch.ops.aten.view.default(bmm_29, [1, 16, 5, 384, 48]);  bmm_29 = None
    permute_68 = torch.ops.aten.permute.default(view_431, [0, 2, 3, 1, 4]);  view_431 = None
    clone_69 = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    _unsafe_view_68 = torch.ops.aten._unsafe_view.default(clone_69, [1, 5, 384, 768]);  clone_69 = None
    t_141 = torch.ops.aten.t.default(diffusion_transformer_blocks_11_to_out_weight);  diffusion_transformer_blocks_11_to_out_weight = None
    view_432 = torch.ops.aten.view.default(_unsafe_view_68, [1920, 768]);  _unsafe_view_68 = None
    mm_119 = torch.ops.aten.mm.default(view_432, t_141);  view_432 = t_141 = None
    view_433 = torch.ops.aten.view.default(mm_119, [1, 5, 384, 768]);  mm_119 = None
    view_434 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_142 = torch.ops.aten.t.default(diffusion_transformer_blocks_11_gate_proj_0_weight);  diffusion_transformer_blocks_11_gate_proj_0_weight = None
    addmm_22 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_11_gate_proj_0_bias, view_434, t_142);  diffusion_transformer_blocks_11_gate_proj_0_bias = view_434 = t_142 = None
    view_435 = torch.ops.aten.view.default(addmm_22, [1, 5, 384, 768]);  addmm_22 = None
    sigmoid_28 = torch.ops.aten.sigmoid.default(view_435);  view_435 = None
    mul_104 = torch.ops.aten.mul.Tensor(sigmoid_28, view_433);  sigmoid_28 = view_433 = None
    native_layer_norm_default_54 = torch.ops.aten.native_layer_norm.default(add_130, [768], None, None, 0.1)
    getitem_320 = native_layer_norm_default_54[0];  native_layer_norm_default_54 = None
    t_143 = torch.ops.aten.t.default(diffusion_transformer_blocks_11_transition_ada_ln_lin_s_merged_weight);  diffusion_transformer_blocks_11_transition_ada_ln_lin_s_merged_weight = None
    view_436 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_120 = torch.ops.aten.mm.default(view_436, t_143);  view_436 = t_143 = None
    view_437 = torch.ops.aten.view.default(mm_120, [1, 5, 384, 1536]);  mm_120 = None
    split_tensor_63 = torch.ops.aten.split.Tensor(view_437, 768, dim = -1);  view_437 = None
    getitem_323 = split_tensor_63[0]
    getitem_324 = split_tensor_63[1];  split_tensor_63 = None
    add_135 = torch.ops.aten.add.Tensor(getitem_323, 1);  getitem_323 = None
    mul_105 = torch.ops.aten.mul.Tensor(getitem_320, add_135);  getitem_320 = add_135 = None
    add_136 = torch.ops.aten.add.Tensor(mul_105, getitem_324);  mul_105 = getitem_324 = None
    t_144 = torch.ops.aten.t.default(diffusion_transformer_blocks_11_transition_linear_a_nobias_double_weight);  diffusion_transformer_blocks_11_transition_linear_a_nobias_double_weight = None
    view_438 = torch.ops.aten.view.default(add_136, [1920, 768]);  add_136 = None
    mm_121 = torch.ops.aten.mm.default(view_438, t_144);  view_438 = t_144 = None
    view_439 = torch.ops.aten.view.default(mm_121, [1, 5, 384, 3072]);  mm_121 = None
    split_tensor_64 = torch.ops.aten.split.Tensor(view_439, 1536, dim = -1);  view_439 = None
    getitem_325 = split_tensor_64[0]
    getitem_326 = split_tensor_64[1];  split_tensor_64 = None
    silu_18 = torch.ops.aten.silu.default(getitem_325);  getitem_325 = None
    mul_106 = torch.ops.aten.mul.Tensor(silu_18, getitem_326);  silu_18 = getitem_326 = None
    view_440 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_145 = torch.ops.aten.t.default(diffusion_transformer_blocks_11_transition_linear_s_biasinit_m2_weight);  diffusion_transformer_blocks_11_transition_linear_s_biasinit_m2_weight = None
    addmm_23 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_11_transition_linear_s_biasinit_m2_bias, view_440, t_145);  diffusion_transformer_blocks_11_transition_linear_s_biasinit_m2_bias = view_440 = t_145 = None
    view_441 = torch.ops.aten.view.default(addmm_23, [1, 5, 384, 768]);  addmm_23 = None
    sigmoid_29 = torch.ops.aten.sigmoid.default(view_441);  view_441 = None
    t_146 = torch.ops.aten.t.default(diffusion_transformer_blocks_11_transition_linear_b_nobias_weight);  diffusion_transformer_blocks_11_transition_linear_b_nobias_weight = None
    view_442 = torch.ops.aten.view.default(mul_106, [1920, 1536]);  mul_106 = None
    mm_122 = torch.ops.aten.mm.default(view_442, t_146);  view_442 = t_146 = None
    view_443 = torch.ops.aten.view.default(mm_122, [1, 5, 384, 768]);  mm_122 = None
    mul_107 = torch.ops.aten.mul.Tensor(sigmoid_29, view_443);  sigmoid_29 = view_443 = None
    add_137 = torch.ops.aten.add.Tensor(mul_104, mul_107);  mul_104 = mul_107 = None
    add_138 = torch.ops.aten.add.Tensor(add_130, add_137);  add_130 = add_137 = None
    native_layer_norm_default_55 = torch.ops.aten.native_layer_norm.default(view_140, [256], diffusion_transformer_blocks_12_pair_layer_norm_weight, diffusion_transformer_blocks_12_pair_layer_norm_bias, 1e-05);  diffusion_transformer_blocks_12_pair_layer_norm_weight = diffusion_transformer_blocks_12_pair_layer_norm_bias = None
    getitem_327 = native_layer_norm_default_55[0];  native_layer_norm_default_55 = None
    t_147 = torch.ops.aten.t.default(diffusion_transformer_blocks_12_pair_linear_weight);  diffusion_transformer_blocks_12_pair_linear_weight = None
    view_444 = torch.ops.aten.view.default(getitem_327, [147456, 256]);  getitem_327 = None
    mm_123 = torch.ops.aten.mm.default(view_444, t_147);  view_444 = t_147 = None
    view_445 = torch.ops.aten.view.default(mm_123, [1, 1, 384, 384, 16]);  mm_123 = None
    native_layer_norm_default_56 = torch.ops.aten.native_layer_norm.default(add_138, [768], None, None, 0.1)
    getitem_330 = native_layer_norm_default_56[0];  native_layer_norm_default_56 = None
    t_148 = torch.ops.aten.t.default(diffusion_transformer_blocks_12_norm_in_lin_s_merged_weight);  diffusion_transformer_blocks_12_norm_in_lin_s_merged_weight = None
    view_446 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_124 = torch.ops.aten.mm.default(view_446, t_148);  view_446 = t_148 = None
    view_447 = torch.ops.aten.view.default(mm_124, [1, 5, 384, 1536]);  mm_124 = None
    split_tensor_65 = torch.ops.aten.split.Tensor(view_447, 768, dim = -1);  view_447 = None
    getitem_333 = split_tensor_65[0]
    getitem_334 = split_tensor_65[1];  split_tensor_65 = None
    add_139 = torch.ops.aten.add.Tensor(getitem_333, 1);  getitem_333 = None
    mul_108 = torch.ops.aten.mul.Tensor(getitem_330, add_139);  getitem_330 = add_139 = None
    add_140 = torch.ops.aten.add.Tensor(mul_108, getitem_334);  mul_108 = getitem_334 = None
    t_149 = torch.ops.aten.t.default(diffusion_transformer_blocks_12_to_qkv_weight);  diffusion_transformer_blocks_12_to_qkv_weight = None
    view_448 = torch.ops.aten.view.default(add_140, [1920, 768]);  add_140 = None
    mm_125 = torch.ops.aten.mm.default(view_448, t_149);  view_448 = t_149 = None
    view_449 = torch.ops.aten.view.default(mm_125, [1, 5, 384, 2304]);  mm_125 = None
    view_450 = torch.ops.aten.view.default(view_449, [1, 5, 384, 16, 144]);  view_449 = None
    permute_69 = torch.ops.aten.permute.default(view_450, [0, 3, 1, 2, 4]);  view_450 = None
    split_tensor_66 = torch.ops.aten.split.Tensor(permute_69, 48, dim = -1);  permute_69 = None
    getitem_335 = split_tensor_66[0]
    getitem_336 = split_tensor_66[1]
    getitem_337 = split_tensor_66[2];  split_tensor_66 = None
    view_451 = torch.ops.aten.view.default(diffusion_transformer_blocks_12_q_bias, [1, 16, 1, 1, 48]);  diffusion_transformer_blocks_12_q_bias = None
    add_141 = torch.ops.aten.add.Tensor(getitem_335, view_451);  getitem_335 = view_451 = None
    view_452 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 384, 384, 1])
    bitwise_not_30 = torch.ops.aten.bitwise_not.default(view_452);  view_452 = None
    masked_fill_30 = torch.ops.aten.masked_fill.Scalar(view_445, bitwise_not_30, -10000);  view_445 = bitwise_not_30 = None
    permute_70 = torch.ops.aten.permute.default(masked_fill_30, [0, 4, 1, 2, 3]);  masked_fill_30 = None
    view_453 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 1, 384, 384])
    bitwise_not_31 = torch.ops.aten.bitwise_not.default(view_453);  view_453 = None
    masked_fill_31 = torch.ops.aten.masked_fill.Scalar(permute_70, bitwise_not_31, -10000);  permute_70 = bitwise_not_31 = None
    mul_109 = torch.ops.aten.mul.Scalar(add_141, 0.3799178428257963);  add_141 = None
    transpose_12 = torch.ops.aten.transpose.int(getitem_336, -2, -1);  getitem_336 = None
    mul_110 = torch.ops.aten.mul.Scalar(transpose_12, 0.3799178428257963);  transpose_12 = None
    expand_62 = torch.ops.aten.expand.default(mul_109, [1, 16, 5, 384, 48]);  mul_109 = None
    clone_70 = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
    _unsafe_view_69 = torch.ops.aten._unsafe_view.default(clone_70, [80, 384, 48]);  clone_70 = None
    expand_63 = torch.ops.aten.expand.default(mul_110, [1, 16, 5, 48, 384]);  mul_110 = None
    clone_71 = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
    _unsafe_view_70 = torch.ops.aten._unsafe_view.default(clone_71, [80, 48, 384]);  clone_71 = None
    bmm_30 = torch.ops.aten.bmm.default(_unsafe_view_69, _unsafe_view_70);  _unsafe_view_69 = _unsafe_view_70 = None
    view_454 = torch.ops.aten.view.default(bmm_30, [1, 16, 5, 384, 384]);  bmm_30 = None
    add_142 = torch.ops.aten.add.Tensor(view_454, masked_fill_31);  view_454 = masked_fill_31 = None
    _softmax_12 = torch.ops.aten._softmax.default(add_142, -1, False);  add_142 = None
    expand_64 = torch.ops.aten.expand.default(_softmax_12, [1, 16, 5, 384, 384]);  _softmax_12 = None
    view_455 = torch.ops.aten.view.default(expand_64, [80, 384, 384]);  expand_64 = None
    expand_65 = torch.ops.aten.expand.default(getitem_337, [1, 16, 5, 384, 48]);  getitem_337 = None
    clone_72 = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    _unsafe_view_71 = torch.ops.aten._unsafe_view.default(clone_72, [80, 384, 48]);  clone_72 = None
    bmm_31 = torch.ops.aten.bmm.default(view_455, _unsafe_view_71);  view_455 = _unsafe_view_71 = None
    view_456 = torch.ops.aten.view.default(bmm_31, [1, 16, 5, 384, 48]);  bmm_31 = None
    permute_71 = torch.ops.aten.permute.default(view_456, [0, 2, 3, 1, 4]);  view_456 = None
    clone_73 = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    _unsafe_view_72 = torch.ops.aten._unsafe_view.default(clone_73, [1, 5, 384, 768]);  clone_73 = None
    t_150 = torch.ops.aten.t.default(diffusion_transformer_blocks_12_to_out_weight);  diffusion_transformer_blocks_12_to_out_weight = None
    view_457 = torch.ops.aten.view.default(_unsafe_view_72, [1920, 768]);  _unsafe_view_72 = None
    mm_126 = torch.ops.aten.mm.default(view_457, t_150);  view_457 = t_150 = None
    view_458 = torch.ops.aten.view.default(mm_126, [1, 5, 384, 768]);  mm_126 = None
    view_459 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_151 = torch.ops.aten.t.default(diffusion_transformer_blocks_12_gate_proj_0_weight);  diffusion_transformer_blocks_12_gate_proj_0_weight = None
    addmm_24 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_12_gate_proj_0_bias, view_459, t_151);  diffusion_transformer_blocks_12_gate_proj_0_bias = view_459 = t_151 = None
    view_460 = torch.ops.aten.view.default(addmm_24, [1, 5, 384, 768]);  addmm_24 = None
    sigmoid_30 = torch.ops.aten.sigmoid.default(view_460);  view_460 = None
    mul_111 = torch.ops.aten.mul.Tensor(sigmoid_30, view_458);  sigmoid_30 = view_458 = None
    native_layer_norm_default_57 = torch.ops.aten.native_layer_norm.default(add_138, [768], None, None, 0.1)
    getitem_338 = native_layer_norm_default_57[0];  native_layer_norm_default_57 = None
    t_152 = torch.ops.aten.t.default(diffusion_transformer_blocks_12_transition_ada_ln_lin_s_merged_weight);  diffusion_transformer_blocks_12_transition_ada_ln_lin_s_merged_weight = None
    view_461 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_127 = torch.ops.aten.mm.default(view_461, t_152);  view_461 = t_152 = None
    view_462 = torch.ops.aten.view.default(mm_127, [1, 5, 384, 1536]);  mm_127 = None
    split_tensor_67 = torch.ops.aten.split.Tensor(view_462, 768, dim = -1);  view_462 = None
    getitem_341 = split_tensor_67[0]
    getitem_342 = split_tensor_67[1];  split_tensor_67 = None
    add_143 = torch.ops.aten.add.Tensor(getitem_341, 1);  getitem_341 = None
    mul_112 = torch.ops.aten.mul.Tensor(getitem_338, add_143);  getitem_338 = add_143 = None
    add_144 = torch.ops.aten.add.Tensor(mul_112, getitem_342);  mul_112 = getitem_342 = None
    t_153 = torch.ops.aten.t.default(diffusion_transformer_blocks_12_transition_linear_a_nobias_double_weight);  diffusion_transformer_blocks_12_transition_linear_a_nobias_double_weight = None
    view_463 = torch.ops.aten.view.default(add_144, [1920, 768]);  add_144 = None
    mm_128 = torch.ops.aten.mm.default(view_463, t_153);  view_463 = t_153 = None
    view_464 = torch.ops.aten.view.default(mm_128, [1, 5, 384, 3072]);  mm_128 = None
    split_tensor_68 = torch.ops.aten.split.Tensor(view_464, 1536, dim = -1);  view_464 = None
    getitem_343 = split_tensor_68[0]
    getitem_344 = split_tensor_68[1];  split_tensor_68 = None
    silu_19 = torch.ops.aten.silu.default(getitem_343);  getitem_343 = None
    mul_113 = torch.ops.aten.mul.Tensor(silu_19, getitem_344);  silu_19 = getitem_344 = None
    view_465 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_154 = torch.ops.aten.t.default(diffusion_transformer_blocks_12_transition_linear_s_biasinit_m2_weight);  diffusion_transformer_blocks_12_transition_linear_s_biasinit_m2_weight = None
    addmm_25 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_12_transition_linear_s_biasinit_m2_bias, view_465, t_154);  diffusion_transformer_blocks_12_transition_linear_s_biasinit_m2_bias = view_465 = t_154 = None
    view_466 = torch.ops.aten.view.default(addmm_25, [1, 5, 384, 768]);  addmm_25 = None
    sigmoid_31 = torch.ops.aten.sigmoid.default(view_466);  view_466 = None
    t_155 = torch.ops.aten.t.default(diffusion_transformer_blocks_12_transition_linear_b_nobias_weight);  diffusion_transformer_blocks_12_transition_linear_b_nobias_weight = None
    view_467 = torch.ops.aten.view.default(mul_113, [1920, 1536]);  mul_113 = None
    mm_129 = torch.ops.aten.mm.default(view_467, t_155);  view_467 = t_155 = None
    view_468 = torch.ops.aten.view.default(mm_129, [1, 5, 384, 768]);  mm_129 = None
    mul_114 = torch.ops.aten.mul.Tensor(sigmoid_31, view_468);  sigmoid_31 = view_468 = None
    add_145 = torch.ops.aten.add.Tensor(mul_111, mul_114);  mul_111 = mul_114 = None
    add_146 = torch.ops.aten.add.Tensor(add_138, add_145);  add_138 = add_145 = None
    native_layer_norm_default_58 = torch.ops.aten.native_layer_norm.default(view_140, [256], diffusion_transformer_blocks_13_pair_layer_norm_weight, diffusion_transformer_blocks_13_pair_layer_norm_bias, 1e-05);  diffusion_transformer_blocks_13_pair_layer_norm_weight = diffusion_transformer_blocks_13_pair_layer_norm_bias = None
    getitem_345 = native_layer_norm_default_58[0];  native_layer_norm_default_58 = None
    t_156 = torch.ops.aten.t.default(diffusion_transformer_blocks_13_pair_linear_weight);  diffusion_transformer_blocks_13_pair_linear_weight = None
    view_469 = torch.ops.aten.view.default(getitem_345, [147456, 256]);  getitem_345 = None
    mm_130 = torch.ops.aten.mm.default(view_469, t_156);  view_469 = t_156 = None
    view_470 = torch.ops.aten.view.default(mm_130, [1, 1, 384, 384, 16]);  mm_130 = None
    native_layer_norm_default_59 = torch.ops.aten.native_layer_norm.default(add_146, [768], None, None, 0.1)
    getitem_348 = native_layer_norm_default_59[0];  native_layer_norm_default_59 = None
    t_157 = torch.ops.aten.t.default(diffusion_transformer_blocks_13_norm_in_lin_s_merged_weight);  diffusion_transformer_blocks_13_norm_in_lin_s_merged_weight = None
    view_471 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_131 = torch.ops.aten.mm.default(view_471, t_157);  view_471 = t_157 = None
    view_472 = torch.ops.aten.view.default(mm_131, [1, 5, 384, 1536]);  mm_131 = None
    split_tensor_69 = torch.ops.aten.split.Tensor(view_472, 768, dim = -1);  view_472 = None
    getitem_351 = split_tensor_69[0]
    getitem_352 = split_tensor_69[1];  split_tensor_69 = None
    add_147 = torch.ops.aten.add.Tensor(getitem_351, 1);  getitem_351 = None
    mul_115 = torch.ops.aten.mul.Tensor(getitem_348, add_147);  getitem_348 = add_147 = None
    add_148 = torch.ops.aten.add.Tensor(mul_115, getitem_352);  mul_115 = getitem_352 = None
    t_158 = torch.ops.aten.t.default(diffusion_transformer_blocks_13_to_qkv_weight);  diffusion_transformer_blocks_13_to_qkv_weight = None
    view_473 = torch.ops.aten.view.default(add_148, [1920, 768]);  add_148 = None
    mm_132 = torch.ops.aten.mm.default(view_473, t_158);  view_473 = t_158 = None
    view_474 = torch.ops.aten.view.default(mm_132, [1, 5, 384, 2304]);  mm_132 = None
    view_475 = torch.ops.aten.view.default(view_474, [1, 5, 384, 16, 144]);  view_474 = None
    permute_72 = torch.ops.aten.permute.default(view_475, [0, 3, 1, 2, 4]);  view_475 = None
    split_tensor_70 = torch.ops.aten.split.Tensor(permute_72, 48, dim = -1);  permute_72 = None
    getitem_353 = split_tensor_70[0]
    getitem_354 = split_tensor_70[1]
    getitem_355 = split_tensor_70[2];  split_tensor_70 = None
    view_476 = torch.ops.aten.view.default(diffusion_transformer_blocks_13_q_bias, [1, 16, 1, 1, 48]);  diffusion_transformer_blocks_13_q_bias = None
    add_149 = torch.ops.aten.add.Tensor(getitem_353, view_476);  getitem_353 = view_476 = None
    view_477 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 384, 384, 1])
    bitwise_not_32 = torch.ops.aten.bitwise_not.default(view_477);  view_477 = None
    masked_fill_32 = torch.ops.aten.masked_fill.Scalar(view_470, bitwise_not_32, -10000);  view_470 = bitwise_not_32 = None
    permute_73 = torch.ops.aten.permute.default(masked_fill_32, [0, 4, 1, 2, 3]);  masked_fill_32 = None
    view_478 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 1, 384, 384])
    bitwise_not_33 = torch.ops.aten.bitwise_not.default(view_478);  view_478 = None
    masked_fill_33 = torch.ops.aten.masked_fill.Scalar(permute_73, bitwise_not_33, -10000);  permute_73 = bitwise_not_33 = None
    mul_116 = torch.ops.aten.mul.Scalar(add_149, 0.3799178428257963);  add_149 = None
    transpose_13 = torch.ops.aten.transpose.int(getitem_354, -2, -1);  getitem_354 = None
    mul_117 = torch.ops.aten.mul.Scalar(transpose_13, 0.3799178428257963);  transpose_13 = None
    expand_66 = torch.ops.aten.expand.default(mul_116, [1, 16, 5, 384, 48]);  mul_116 = None
    clone_74 = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
    _unsafe_view_73 = torch.ops.aten._unsafe_view.default(clone_74, [80, 384, 48]);  clone_74 = None
    expand_67 = torch.ops.aten.expand.default(mul_117, [1, 16, 5, 48, 384]);  mul_117 = None
    clone_75 = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
    _unsafe_view_74 = torch.ops.aten._unsafe_view.default(clone_75, [80, 48, 384]);  clone_75 = None
    bmm_32 = torch.ops.aten.bmm.default(_unsafe_view_73, _unsafe_view_74);  _unsafe_view_73 = _unsafe_view_74 = None
    view_479 = torch.ops.aten.view.default(bmm_32, [1, 16, 5, 384, 384]);  bmm_32 = None
    add_150 = torch.ops.aten.add.Tensor(view_479, masked_fill_33);  view_479 = masked_fill_33 = None
    _softmax_13 = torch.ops.aten._softmax.default(add_150, -1, False);  add_150 = None
    expand_68 = torch.ops.aten.expand.default(_softmax_13, [1, 16, 5, 384, 384]);  _softmax_13 = None
    view_480 = torch.ops.aten.view.default(expand_68, [80, 384, 384]);  expand_68 = None
    expand_69 = torch.ops.aten.expand.default(getitem_355, [1, 16, 5, 384, 48]);  getitem_355 = None
    clone_76 = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
    _unsafe_view_75 = torch.ops.aten._unsafe_view.default(clone_76, [80, 384, 48]);  clone_76 = None
    bmm_33 = torch.ops.aten.bmm.default(view_480, _unsafe_view_75);  view_480 = _unsafe_view_75 = None
    view_481 = torch.ops.aten.view.default(bmm_33, [1, 16, 5, 384, 48]);  bmm_33 = None
    permute_74 = torch.ops.aten.permute.default(view_481, [0, 2, 3, 1, 4]);  view_481 = None
    clone_77 = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
    _unsafe_view_76 = torch.ops.aten._unsafe_view.default(clone_77, [1, 5, 384, 768]);  clone_77 = None
    t_159 = torch.ops.aten.t.default(diffusion_transformer_blocks_13_to_out_weight);  diffusion_transformer_blocks_13_to_out_weight = None
    view_482 = torch.ops.aten.view.default(_unsafe_view_76, [1920, 768]);  _unsafe_view_76 = None
    mm_133 = torch.ops.aten.mm.default(view_482, t_159);  view_482 = t_159 = None
    view_483 = torch.ops.aten.view.default(mm_133, [1, 5, 384, 768]);  mm_133 = None
    view_484 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_160 = torch.ops.aten.t.default(diffusion_transformer_blocks_13_gate_proj_0_weight);  diffusion_transformer_blocks_13_gate_proj_0_weight = None
    addmm_26 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_13_gate_proj_0_bias, view_484, t_160);  diffusion_transformer_blocks_13_gate_proj_0_bias = view_484 = t_160 = None
    view_485 = torch.ops.aten.view.default(addmm_26, [1, 5, 384, 768]);  addmm_26 = None
    sigmoid_32 = torch.ops.aten.sigmoid.default(view_485);  view_485 = None
    mul_118 = torch.ops.aten.mul.Tensor(sigmoid_32, view_483);  sigmoid_32 = view_483 = None
    native_layer_norm_default_60 = torch.ops.aten.native_layer_norm.default(add_146, [768], None, None, 0.1)
    getitem_356 = native_layer_norm_default_60[0];  native_layer_norm_default_60 = None
    t_161 = torch.ops.aten.t.default(diffusion_transformer_blocks_13_transition_ada_ln_lin_s_merged_weight);  diffusion_transformer_blocks_13_transition_ada_ln_lin_s_merged_weight = None
    view_486 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_134 = torch.ops.aten.mm.default(view_486, t_161);  view_486 = t_161 = None
    view_487 = torch.ops.aten.view.default(mm_134, [1, 5, 384, 1536]);  mm_134 = None
    split_tensor_71 = torch.ops.aten.split.Tensor(view_487, 768, dim = -1);  view_487 = None
    getitem_359 = split_tensor_71[0]
    getitem_360 = split_tensor_71[1];  split_tensor_71 = None
    add_151 = torch.ops.aten.add.Tensor(getitem_359, 1);  getitem_359 = None
    mul_119 = torch.ops.aten.mul.Tensor(getitem_356, add_151);  getitem_356 = add_151 = None
    add_152 = torch.ops.aten.add.Tensor(mul_119, getitem_360);  mul_119 = getitem_360 = None
    t_162 = torch.ops.aten.t.default(diffusion_transformer_blocks_13_transition_linear_a_nobias_double_weight);  diffusion_transformer_blocks_13_transition_linear_a_nobias_double_weight = None
    view_488 = torch.ops.aten.view.default(add_152, [1920, 768]);  add_152 = None
    mm_135 = torch.ops.aten.mm.default(view_488, t_162);  view_488 = t_162 = None
    view_489 = torch.ops.aten.view.default(mm_135, [1, 5, 384, 3072]);  mm_135 = None
    split_tensor_72 = torch.ops.aten.split.Tensor(view_489, 1536, dim = -1);  view_489 = None
    getitem_361 = split_tensor_72[0]
    getitem_362 = split_tensor_72[1];  split_tensor_72 = None
    silu_20 = torch.ops.aten.silu.default(getitem_361);  getitem_361 = None
    mul_120 = torch.ops.aten.mul.Tensor(silu_20, getitem_362);  silu_20 = getitem_362 = None
    view_490 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_163 = torch.ops.aten.t.default(diffusion_transformer_blocks_13_transition_linear_s_biasinit_m2_weight);  diffusion_transformer_blocks_13_transition_linear_s_biasinit_m2_weight = None
    addmm_27 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_13_transition_linear_s_biasinit_m2_bias, view_490, t_163);  diffusion_transformer_blocks_13_transition_linear_s_biasinit_m2_bias = view_490 = t_163 = None
    view_491 = torch.ops.aten.view.default(addmm_27, [1, 5, 384, 768]);  addmm_27 = None
    sigmoid_33 = torch.ops.aten.sigmoid.default(view_491);  view_491 = None
    t_164 = torch.ops.aten.t.default(diffusion_transformer_blocks_13_transition_linear_b_nobias_weight);  diffusion_transformer_blocks_13_transition_linear_b_nobias_weight = None
    view_492 = torch.ops.aten.view.default(mul_120, [1920, 1536]);  mul_120 = None
    mm_136 = torch.ops.aten.mm.default(view_492, t_164);  view_492 = t_164 = None
    view_493 = torch.ops.aten.view.default(mm_136, [1, 5, 384, 768]);  mm_136 = None
    mul_121 = torch.ops.aten.mul.Tensor(sigmoid_33, view_493);  sigmoid_33 = view_493 = None
    add_153 = torch.ops.aten.add.Tensor(mul_118, mul_121);  mul_118 = mul_121 = None
    add_154 = torch.ops.aten.add.Tensor(add_146, add_153);  add_146 = add_153 = None
    native_layer_norm_default_61 = torch.ops.aten.native_layer_norm.default(view_140, [256], diffusion_transformer_blocks_14_pair_layer_norm_weight, diffusion_transformer_blocks_14_pair_layer_norm_bias, 1e-05);  diffusion_transformer_blocks_14_pair_layer_norm_weight = diffusion_transformer_blocks_14_pair_layer_norm_bias = None
    getitem_363 = native_layer_norm_default_61[0];  native_layer_norm_default_61 = None
    t_165 = torch.ops.aten.t.default(diffusion_transformer_blocks_14_pair_linear_weight);  diffusion_transformer_blocks_14_pair_linear_weight = None
    view_494 = torch.ops.aten.view.default(getitem_363, [147456, 256]);  getitem_363 = None
    mm_137 = torch.ops.aten.mm.default(view_494, t_165);  view_494 = t_165 = None
    view_495 = torch.ops.aten.view.default(mm_137, [1, 1, 384, 384, 16]);  mm_137 = None
    native_layer_norm_default_62 = torch.ops.aten.native_layer_norm.default(add_154, [768], None, None, 0.1)
    getitem_366 = native_layer_norm_default_62[0];  native_layer_norm_default_62 = None
    t_166 = torch.ops.aten.t.default(diffusion_transformer_blocks_14_norm_in_lin_s_merged_weight);  diffusion_transformer_blocks_14_norm_in_lin_s_merged_weight = None
    view_496 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_138 = torch.ops.aten.mm.default(view_496, t_166);  view_496 = t_166 = None
    view_497 = torch.ops.aten.view.default(mm_138, [1, 5, 384, 1536]);  mm_138 = None
    split_tensor_73 = torch.ops.aten.split.Tensor(view_497, 768, dim = -1);  view_497 = None
    getitem_369 = split_tensor_73[0]
    getitem_370 = split_tensor_73[1];  split_tensor_73 = None
    add_155 = torch.ops.aten.add.Tensor(getitem_369, 1);  getitem_369 = None
    mul_122 = torch.ops.aten.mul.Tensor(getitem_366, add_155);  getitem_366 = add_155 = None
    add_156 = torch.ops.aten.add.Tensor(mul_122, getitem_370);  mul_122 = getitem_370 = None
    t_167 = torch.ops.aten.t.default(diffusion_transformer_blocks_14_to_qkv_weight);  diffusion_transformer_blocks_14_to_qkv_weight = None
    view_498 = torch.ops.aten.view.default(add_156, [1920, 768]);  add_156 = None
    mm_139 = torch.ops.aten.mm.default(view_498, t_167);  view_498 = t_167 = None
    view_499 = torch.ops.aten.view.default(mm_139, [1, 5, 384, 2304]);  mm_139 = None
    view_500 = torch.ops.aten.view.default(view_499, [1, 5, 384, 16, 144]);  view_499 = None
    permute_75 = torch.ops.aten.permute.default(view_500, [0, 3, 1, 2, 4]);  view_500 = None
    split_tensor_74 = torch.ops.aten.split.Tensor(permute_75, 48, dim = -1);  permute_75 = None
    getitem_371 = split_tensor_74[0]
    getitem_372 = split_tensor_74[1]
    getitem_373 = split_tensor_74[2];  split_tensor_74 = None
    view_501 = torch.ops.aten.view.default(diffusion_transformer_blocks_14_q_bias, [1, 16, 1, 1, 48]);  diffusion_transformer_blocks_14_q_bias = None
    add_157 = torch.ops.aten.add.Tensor(getitem_371, view_501);  getitem_371 = view_501 = None
    view_502 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 384, 384, 1])
    bitwise_not_34 = torch.ops.aten.bitwise_not.default(view_502);  view_502 = None
    masked_fill_34 = torch.ops.aten.masked_fill.Scalar(view_495, bitwise_not_34, -10000);  view_495 = bitwise_not_34 = None
    permute_76 = torch.ops.aten.permute.default(masked_fill_34, [0, 4, 1, 2, 3]);  masked_fill_34 = None
    view_503 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 1, 384, 384])
    bitwise_not_35 = torch.ops.aten.bitwise_not.default(view_503);  view_503 = None
    masked_fill_35 = torch.ops.aten.masked_fill.Scalar(permute_76, bitwise_not_35, -10000);  permute_76 = bitwise_not_35 = None
    mul_123 = torch.ops.aten.mul.Scalar(add_157, 0.3799178428257963);  add_157 = None
    transpose_14 = torch.ops.aten.transpose.int(getitem_372, -2, -1);  getitem_372 = None
    mul_124 = torch.ops.aten.mul.Scalar(transpose_14, 0.3799178428257963);  transpose_14 = None
    expand_70 = torch.ops.aten.expand.default(mul_123, [1, 16, 5, 384, 48]);  mul_123 = None
    clone_78 = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
    _unsafe_view_77 = torch.ops.aten._unsafe_view.default(clone_78, [80, 384, 48]);  clone_78 = None
    expand_71 = torch.ops.aten.expand.default(mul_124, [1, 16, 5, 48, 384]);  mul_124 = None
    clone_79 = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
    _unsafe_view_78 = torch.ops.aten._unsafe_view.default(clone_79, [80, 48, 384]);  clone_79 = None
    bmm_34 = torch.ops.aten.bmm.default(_unsafe_view_77, _unsafe_view_78);  _unsafe_view_77 = _unsafe_view_78 = None
    view_504 = torch.ops.aten.view.default(bmm_34, [1, 16, 5, 384, 384]);  bmm_34 = None
    add_158 = torch.ops.aten.add.Tensor(view_504, masked_fill_35);  view_504 = masked_fill_35 = None
    _softmax_14 = torch.ops.aten._softmax.default(add_158, -1, False);  add_158 = None
    expand_72 = torch.ops.aten.expand.default(_softmax_14, [1, 16, 5, 384, 384]);  _softmax_14 = None
    view_505 = torch.ops.aten.view.default(expand_72, [80, 384, 384]);  expand_72 = None
    expand_73 = torch.ops.aten.expand.default(getitem_373, [1, 16, 5, 384, 48]);  getitem_373 = None
    clone_80 = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
    _unsafe_view_79 = torch.ops.aten._unsafe_view.default(clone_80, [80, 384, 48]);  clone_80 = None
    bmm_35 = torch.ops.aten.bmm.default(view_505, _unsafe_view_79);  view_505 = _unsafe_view_79 = None
    view_506 = torch.ops.aten.view.default(bmm_35, [1, 16, 5, 384, 48]);  bmm_35 = None
    permute_77 = torch.ops.aten.permute.default(view_506, [0, 2, 3, 1, 4]);  view_506 = None
    clone_81 = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    _unsafe_view_80 = torch.ops.aten._unsafe_view.default(clone_81, [1, 5, 384, 768]);  clone_81 = None
    t_168 = torch.ops.aten.t.default(diffusion_transformer_blocks_14_to_out_weight);  diffusion_transformer_blocks_14_to_out_weight = None
    view_507 = torch.ops.aten.view.default(_unsafe_view_80, [1920, 768]);  _unsafe_view_80 = None
    mm_140 = torch.ops.aten.mm.default(view_507, t_168);  view_507 = t_168 = None
    view_508 = torch.ops.aten.view.default(mm_140, [1, 5, 384, 768]);  mm_140 = None
    view_509 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_169 = torch.ops.aten.t.default(diffusion_transformer_blocks_14_gate_proj_0_weight);  diffusion_transformer_blocks_14_gate_proj_0_weight = None
    addmm_28 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_14_gate_proj_0_bias, view_509, t_169);  diffusion_transformer_blocks_14_gate_proj_0_bias = view_509 = t_169 = None
    view_510 = torch.ops.aten.view.default(addmm_28, [1, 5, 384, 768]);  addmm_28 = None
    sigmoid_34 = torch.ops.aten.sigmoid.default(view_510);  view_510 = None
    mul_125 = torch.ops.aten.mul.Tensor(sigmoid_34, view_508);  sigmoid_34 = view_508 = None
    native_layer_norm_default_63 = torch.ops.aten.native_layer_norm.default(add_154, [768], None, None, 0.1)
    getitem_374 = native_layer_norm_default_63[0];  native_layer_norm_default_63 = None
    t_170 = torch.ops.aten.t.default(diffusion_transformer_blocks_14_transition_ada_ln_lin_s_merged_weight);  diffusion_transformer_blocks_14_transition_ada_ln_lin_s_merged_weight = None
    view_511 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_141 = torch.ops.aten.mm.default(view_511, t_170);  view_511 = t_170 = None
    view_512 = torch.ops.aten.view.default(mm_141, [1, 5, 384, 1536]);  mm_141 = None
    split_tensor_75 = torch.ops.aten.split.Tensor(view_512, 768, dim = -1);  view_512 = None
    getitem_377 = split_tensor_75[0]
    getitem_378 = split_tensor_75[1];  split_tensor_75 = None
    add_159 = torch.ops.aten.add.Tensor(getitem_377, 1);  getitem_377 = None
    mul_126 = torch.ops.aten.mul.Tensor(getitem_374, add_159);  getitem_374 = add_159 = None
    add_160 = torch.ops.aten.add.Tensor(mul_126, getitem_378);  mul_126 = getitem_378 = None
    t_171 = torch.ops.aten.t.default(diffusion_transformer_blocks_14_transition_linear_a_nobias_double_weight);  diffusion_transformer_blocks_14_transition_linear_a_nobias_double_weight = None
    view_513 = torch.ops.aten.view.default(add_160, [1920, 768]);  add_160 = None
    mm_142 = torch.ops.aten.mm.default(view_513, t_171);  view_513 = t_171 = None
    view_514 = torch.ops.aten.view.default(mm_142, [1, 5, 384, 3072]);  mm_142 = None
    split_tensor_76 = torch.ops.aten.split.Tensor(view_514, 1536, dim = -1);  view_514 = None
    getitem_379 = split_tensor_76[0]
    getitem_380 = split_tensor_76[1];  split_tensor_76 = None
    silu_21 = torch.ops.aten.silu.default(getitem_379);  getitem_379 = None
    mul_127 = torch.ops.aten.mul.Tensor(silu_21, getitem_380);  silu_21 = getitem_380 = None
    view_515 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_172 = torch.ops.aten.t.default(diffusion_transformer_blocks_14_transition_linear_s_biasinit_m2_weight);  diffusion_transformer_blocks_14_transition_linear_s_biasinit_m2_weight = None
    addmm_29 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_14_transition_linear_s_biasinit_m2_bias, view_515, t_172);  diffusion_transformer_blocks_14_transition_linear_s_biasinit_m2_bias = view_515 = t_172 = None
    view_516 = torch.ops.aten.view.default(addmm_29, [1, 5, 384, 768]);  addmm_29 = None
    sigmoid_35 = torch.ops.aten.sigmoid.default(view_516);  view_516 = None
    t_173 = torch.ops.aten.t.default(diffusion_transformer_blocks_14_transition_linear_b_nobias_weight);  diffusion_transformer_blocks_14_transition_linear_b_nobias_weight = None
    view_517 = torch.ops.aten.view.default(mul_127, [1920, 1536]);  mul_127 = None
    mm_143 = torch.ops.aten.mm.default(view_517, t_173);  view_517 = t_173 = None
    view_518 = torch.ops.aten.view.default(mm_143, [1, 5, 384, 768]);  mm_143 = None
    mul_128 = torch.ops.aten.mul.Tensor(sigmoid_35, view_518);  sigmoid_35 = view_518 = None
    add_161 = torch.ops.aten.add.Tensor(mul_125, mul_128);  mul_125 = mul_128 = None
    add_162 = torch.ops.aten.add.Tensor(add_154, add_161);  add_154 = add_161 = None
    native_layer_norm_default_64 = torch.ops.aten.native_layer_norm.default(view_140, [256], diffusion_transformer_blocks_15_pair_layer_norm_weight, diffusion_transformer_blocks_15_pair_layer_norm_bias, 1e-05);  view_140 = diffusion_transformer_blocks_15_pair_layer_norm_weight = diffusion_transformer_blocks_15_pair_layer_norm_bias = None
    getitem_381 = native_layer_norm_default_64[0];  native_layer_norm_default_64 = None
    t_174 = torch.ops.aten.t.default(diffusion_transformer_blocks_15_pair_linear_weight);  diffusion_transformer_blocks_15_pair_linear_weight = None
    view_519 = torch.ops.aten.view.default(getitem_381, [147456, 256]);  getitem_381 = None
    mm_144 = torch.ops.aten.mm.default(view_519, t_174);  view_519 = t_174 = None
    view_520 = torch.ops.aten.view.default(mm_144, [1, 1, 384, 384, 16]);  mm_144 = None
    native_layer_norm_default_65 = torch.ops.aten.native_layer_norm.default(add_162, [768], None, None, 0.1)
    getitem_384 = native_layer_norm_default_65[0];  native_layer_norm_default_65 = None
    t_175 = torch.ops.aten.t.default(diffusion_transformer_blocks_15_norm_in_lin_s_merged_weight);  diffusion_transformer_blocks_15_norm_in_lin_s_merged_weight = None
    view_521 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_145 = torch.ops.aten.mm.default(view_521, t_175);  view_521 = t_175 = None
    view_522 = torch.ops.aten.view.default(mm_145, [1, 5, 384, 1536]);  mm_145 = None
    split_tensor_77 = torch.ops.aten.split.Tensor(view_522, 768, dim = -1);  view_522 = None
    getitem_387 = split_tensor_77[0]
    getitem_388 = split_tensor_77[1];  split_tensor_77 = None
    add_163 = torch.ops.aten.add.Tensor(getitem_387, 1);  getitem_387 = None
    mul_129 = torch.ops.aten.mul.Tensor(getitem_384, add_163);  getitem_384 = add_163 = None
    add_164 = torch.ops.aten.add.Tensor(mul_129, getitem_388);  mul_129 = getitem_388 = None
    t_176 = torch.ops.aten.t.default(diffusion_transformer_blocks_15_to_qkv_weight);  diffusion_transformer_blocks_15_to_qkv_weight = None
    view_523 = torch.ops.aten.view.default(add_164, [1920, 768]);  add_164 = None
    mm_146 = torch.ops.aten.mm.default(view_523, t_176);  view_523 = t_176 = None
    view_524 = torch.ops.aten.view.default(mm_146, [1, 5, 384, 2304]);  mm_146 = None
    view_525 = torch.ops.aten.view.default(view_524, [1, 5, 384, 16, 144]);  view_524 = None
    permute_78 = torch.ops.aten.permute.default(view_525, [0, 3, 1, 2, 4]);  view_525 = None
    split_tensor_78 = torch.ops.aten.split.Tensor(permute_78, 48, dim = -1);  permute_78 = None
    getitem_389 = split_tensor_78[0]
    getitem_390 = split_tensor_78[1]
    getitem_391 = split_tensor_78[2];  split_tensor_78 = None
    view_526 = torch.ops.aten.view.default(diffusion_transformer_blocks_15_q_bias, [1, 16, 1, 1, 48]);  diffusion_transformer_blocks_15_q_bias = None
    add_165 = torch.ops.aten.add.Tensor(getitem_389, view_526);  getitem_389 = view_526 = None
    view_527 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 384, 384, 1])
    bitwise_not_36 = torch.ops.aten.bitwise_not.default(view_527);  view_527 = None
    masked_fill_36 = torch.ops.aten.masked_fill.Scalar(view_520, bitwise_not_36, -10000);  view_520 = bitwise_not_36 = None
    permute_79 = torch.ops.aten.permute.default(masked_fill_36, [0, 4, 1, 2, 3]);  masked_fill_36 = None
    view_528 = torch.ops.aten.view.default(bitwise_and_1, [1, 1, 1, 384, 384]);  bitwise_and_1 = None
    bitwise_not_37 = torch.ops.aten.bitwise_not.default(view_528);  view_528 = None
    masked_fill_37 = torch.ops.aten.masked_fill.Scalar(permute_79, bitwise_not_37, -10000);  permute_79 = bitwise_not_37 = None
    mul_130 = torch.ops.aten.mul.Scalar(add_165, 0.3799178428257963);  add_165 = None
    transpose_15 = torch.ops.aten.transpose.int(getitem_390, -2, -1);  getitem_390 = None
    mul_131 = torch.ops.aten.mul.Scalar(transpose_15, 0.3799178428257963);  transpose_15 = None
    expand_74 = torch.ops.aten.expand.default(mul_130, [1, 16, 5, 384, 48]);  mul_130 = None
    clone_82 = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
    _unsafe_view_81 = torch.ops.aten._unsafe_view.default(clone_82, [80, 384, 48]);  clone_82 = None
    expand_75 = torch.ops.aten.expand.default(mul_131, [1, 16, 5, 48, 384]);  mul_131 = None
    clone_83 = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
    _unsafe_view_82 = torch.ops.aten._unsafe_view.default(clone_83, [80, 48, 384]);  clone_83 = None
    bmm_36 = torch.ops.aten.bmm.default(_unsafe_view_81, _unsafe_view_82);  _unsafe_view_81 = _unsafe_view_82 = None
    view_529 = torch.ops.aten.view.default(bmm_36, [1, 16, 5, 384, 384]);  bmm_36 = None
    add_166 = torch.ops.aten.add.Tensor(view_529, masked_fill_37);  view_529 = masked_fill_37 = None
    _softmax_15 = torch.ops.aten._softmax.default(add_166, -1, False);  add_166 = None
    expand_76 = torch.ops.aten.expand.default(_softmax_15, [1, 16, 5, 384, 384]);  _softmax_15 = None
    view_530 = torch.ops.aten.view.default(expand_76, [80, 384, 384]);  expand_76 = None
    expand_77 = torch.ops.aten.expand.default(getitem_391, [1, 16, 5, 384, 48]);  getitem_391 = None
    clone_84 = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
    _unsafe_view_83 = torch.ops.aten._unsafe_view.default(clone_84, [80, 384, 48]);  clone_84 = None
    bmm_37 = torch.ops.aten.bmm.default(view_530, _unsafe_view_83);  view_530 = _unsafe_view_83 = None
    view_531 = torch.ops.aten.view.default(bmm_37, [1, 16, 5, 384, 48]);  bmm_37 = None
    permute_80 = torch.ops.aten.permute.default(view_531, [0, 2, 3, 1, 4]);  view_531 = None
    clone_85 = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    _unsafe_view_84 = torch.ops.aten._unsafe_view.default(clone_85, [1, 5, 384, 768]);  clone_85 = None
    t_177 = torch.ops.aten.t.default(diffusion_transformer_blocks_15_to_out_weight);  diffusion_transformer_blocks_15_to_out_weight = None
    view_532 = torch.ops.aten.view.default(_unsafe_view_84, [1920, 768]);  _unsafe_view_84 = None
    mm_147 = torch.ops.aten.mm.default(view_532, t_177);  view_532 = t_177 = None
    view_533 = torch.ops.aten.view.default(mm_147, [1, 5, 384, 768]);  mm_147 = None
    view_534 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    t_178 = torch.ops.aten.t.default(diffusion_transformer_blocks_15_gate_proj_0_weight);  diffusion_transformer_blocks_15_gate_proj_0_weight = None
    addmm_30 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_15_gate_proj_0_bias, view_534, t_178);  diffusion_transformer_blocks_15_gate_proj_0_bias = view_534 = t_178 = None
    view_535 = torch.ops.aten.view.default(addmm_30, [1, 5, 384, 768]);  addmm_30 = None
    sigmoid_36 = torch.ops.aten.sigmoid.default(view_535);  view_535 = None
    mul_132 = torch.ops.aten.mul.Tensor(sigmoid_36, view_533);  sigmoid_36 = view_533 = None
    native_layer_norm_default_66 = torch.ops.aten.native_layer_norm.default(add_162, [768], None, None, 0.1)
    getitem_392 = native_layer_norm_default_66[0];  native_layer_norm_default_66 = None
    t_179 = torch.ops.aten.t.default(diffusion_transformer_blocks_15_transition_ada_ln_lin_s_merged_weight);  diffusion_transformer_blocks_15_transition_ada_ln_lin_s_merged_weight = None
    view_536 = torch.ops.aten.view.default(getitem_33, [1920, 384])
    mm_148 = torch.ops.aten.mm.default(view_536, t_179);  view_536 = t_179 = None
    view_537 = torch.ops.aten.view.default(mm_148, [1, 5, 384, 1536]);  mm_148 = None
    split_tensor_79 = torch.ops.aten.split.Tensor(view_537, 768, dim = -1);  view_537 = None
    getitem_395 = split_tensor_79[0]
    getitem_396 = split_tensor_79[1];  split_tensor_79 = None
    add_167 = torch.ops.aten.add.Tensor(getitem_395, 1);  getitem_395 = None
    mul_133 = torch.ops.aten.mul.Tensor(getitem_392, add_167);  getitem_392 = add_167 = None
    add_168 = torch.ops.aten.add.Tensor(mul_133, getitem_396);  mul_133 = getitem_396 = None
    t_180 = torch.ops.aten.t.default(diffusion_transformer_blocks_15_transition_linear_a_nobias_double_weight);  diffusion_transformer_blocks_15_transition_linear_a_nobias_double_weight = None
    view_538 = torch.ops.aten.view.default(add_168, [1920, 768]);  add_168 = None
    mm_149 = torch.ops.aten.mm.default(view_538, t_180);  view_538 = t_180 = None
    view_539 = torch.ops.aten.view.default(mm_149, [1, 5, 384, 3072]);  mm_149 = None
    split_tensor_80 = torch.ops.aten.split.Tensor(view_539, 1536, dim = -1);  view_539 = None
    getitem_397 = split_tensor_80[0]
    getitem_398 = split_tensor_80[1];  split_tensor_80 = None
    silu_22 = torch.ops.aten.silu.default(getitem_397);  getitem_397 = None
    mul_134 = torch.ops.aten.mul.Tensor(silu_22, getitem_398);  silu_22 = getitem_398 = None
    view_540 = torch.ops.aten.view.default(getitem_33, [1920, 384]);  getitem_33 = None
    t_181 = torch.ops.aten.t.default(diffusion_transformer_blocks_15_transition_linear_s_biasinit_m2_weight);  diffusion_transformer_blocks_15_transition_linear_s_biasinit_m2_weight = None
    addmm_31 = torch.ops.aten.addmm.default(diffusion_transformer_blocks_15_transition_linear_s_biasinit_m2_bias, view_540, t_181);  diffusion_transformer_blocks_15_transition_linear_s_biasinit_m2_bias = view_540 = t_181 = None
    view_541 = torch.ops.aten.view.default(addmm_31, [1, 5, 384, 768]);  addmm_31 = None
    sigmoid_37 = torch.ops.aten.sigmoid.default(view_541);  view_541 = None
    t_182 = torch.ops.aten.t.default(diffusion_transformer_blocks_15_transition_linear_b_nobias_weight);  diffusion_transformer_blocks_15_transition_linear_b_nobias_weight = None
    view_542 = torch.ops.aten.view.default(mul_134, [1920, 1536]);  mul_134 = None
    mm_150 = torch.ops.aten.mm.default(view_542, t_182);  view_542 = t_182 = None
    view_543 = torch.ops.aten.view.default(mm_150, [1, 5, 384, 768]);  mm_150 = None
    mul_135 = torch.ops.aten.mul.Tensor(sigmoid_37, view_543);  sigmoid_37 = view_543 = None
    add_169 = torch.ops.aten.add.Tensor(mul_132, mul_135);  mul_132 = mul_135 = None
    add_170 = torch.ops.aten.add.Tensor(add_162, add_169);  add_162 = add_169 = None
    native_layer_norm_default_67 = torch.ops.aten.native_layer_norm.default(add_170, [768], post_attn_layernorm_weight, post_attn_layernorm_bias, 1e-05);  add_170 = post_attn_layernorm_weight = post_attn_layernorm_bias = None
    getitem_399 = native_layer_norm_default_67[0];  native_layer_norm_default_67 = None
    native_layer_norm_default_68 = torch.ops.aten.native_layer_norm.default(getitem_42, [128], post_atom_cond_layernorm_weight, post_atom_cond_layernorm_bias, 1e-05);  getitem_42 = post_atom_cond_layernorm_weight = post_atom_cond_layernorm_bias = None
    getitem_402 = native_layer_norm_default_68[0];  native_layer_norm_default_68 = None
    view_544 = torch.ops.aten.view.default(getitem_399, [5, 384, 768]);  getitem_399 = None
    view_545 = torch.ops.aten.view.default(view_127, [5, 8832, 128]);  view_127 = None
    unsqueeze_48 = torch.ops.aten.unsqueeze.default(getitem_402, 1);  getitem_402 = None
    expand_78 = torch.ops.aten.expand.default(unsqueeze_48, [-1, 5, -1, -1]);  unsqueeze_48 = None
    view_546 = torch.ops.aten.view.default(expand_78, [5, 8832, 128]);  expand_78 = None
    unsqueeze_49 = torch.ops.aten.unsqueeze.default(arg350_1, 1);  arg350_1 = None
    expand_79 = torch.ops.aten.expand.default(unsqueeze_49, [-1, 5, -1, -1, -1]);  unsqueeze_49 = None
    view_548 = torch.ops.aten.view.default(expand_79, [5, 276, 32, 128]);  expand_79 = None
    unsqueeze_50 = torch.ops.aten.unsqueeze.default(arg356_1, 1);  arg356_1 = None
    expand_80 = torch.ops.aten.expand.default(unsqueeze_50, [-1, 5, -1]);  unsqueeze_50 = None
    view_549 = torch.ops.aten.view.default(expand_80, [5, 8832]);  expand_80 = None
    unsqueeze_51 = torch.ops.aten.unsqueeze.default(arg349_1, 1);  arg349_1 = None
    expand_81 = torch.ops.aten.expand.default(unsqueeze_51, [-1, 5, -1]);  unsqueeze_51 = None
    view_550 = torch.ops.aten.view.default(expand_81, [5, 8832]);  expand_81 = None
    t_183 = torch.ops.aten.t.default(atom_attention_decoder_token_to_atom_weight);  atom_attention_decoder_token_to_atom_weight = None
    view_551 = torch.ops.aten.view.default(view_544, [1920, 768]);  view_544 = None
    mm_151 = torch.ops.aten.mm.default(view_551, t_183);  view_551 = t_183 = None
    view_552 = torch.ops.aten.view.default(mm_151, [5, 384, 128]);  mm_151 = None
    arange_4 = torch.ops.aten.arange.default(5, device = device(type='cuda', index=0), pin_memory = False)
    unsqueeze_52 = torch.ops.aten.unsqueeze.default(arange_4, 1);  arange_4 = None
    index_12 = torch.ops.aten.index.Tensor(view_552, [unsqueeze_52, view_549]);  view_552 = unsqueeze_52 = view_549 = None
    clone_86 = torch.ops.aten.clone.default(index_12);  index_12 = None
    add_171 = torch.ops.aten.add.Tensor(view_545, clone_86);  view_545 = clone_86 = None
    arange_5 = torch.ops.aten.arange.default(8832, device = device(type='cuda', index=0), pin_memory = False)
    view_553 = torch.ops.aten.view.default(arange_5, [276, 32]);  arange_5 = None
    slice_21 = torch.ops.aten.slice.Tensor(view_553, dim = 0, start = 0, end = 9223372036854775807);  view_553 = None
    slice_22 = torch.ops.aten.slice.Tensor(slice_21, dim = 1, start = 0, end = 1);  slice_21 = None
    add_172 = torch.ops.aten.add.Tensor(slice_22, -48);  slice_22 = None
    arange_6 = torch.ops.aten.arange.default(128, device = device(type='cuda', index=0), pin_memory = False)
    add_173 = torch.ops.aten.add.Tensor(add_172, arange_6);  add_172 = arange_6 = None
    remainder_1 = torch.ops.aten.remainder.Scalar(add_173, 8832);  add_173 = None
    view_554 = torch.ops.aten.view.default(view_57, [1, 5, 276, 32, 128, 16]);  view_57 = None
    view_555 = torch.ops.aten.view.default(view_554, [5, 276, 32, 128, 16]);  view_554 = None
    native_layer_norm_default_69 = torch.ops.aten.native_layer_norm.default(view_555, [16], atom_attention_decoder_atom_transformer_local_diffn_transformer_blocked_pairs2blocked_bias_0_weight, atom_attention_decoder_atom_transformer_local_diffn_transformer_blocked_pairs2blocked_bias_0_bias, 1e-05);  view_555 = atom_attention_decoder_atom_transformer_local_diffn_transformer_blocked_pairs2blocked_bias_0_weight = atom_attention_decoder_atom_transformer_local_diffn_transformer_blocked_pairs2blocked_bias_0_bias = None
    getitem_405 = native_layer_norm_default_69[0];  native_layer_norm_default_69 = None
    unbind_int_4 = torch.ops.aten.unbind.int(atom_attention_decoder_atom_transformer_local_diffn_transformer_blocked_pairs2blocked_bias_1_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_blocked_pairs2blocked_bias_1_weight = None
    getitem_408 = unbind_int_4[0]
    getitem_409 = unbind_int_4[1]
    getitem_410 = unbind_int_4[2];  unbind_int_4 = None
    unsqueeze_53 = torch.ops.aten.unsqueeze.default(view_550, -1)
    bitwise_not_38 = torch.ops.aten.bitwise_not.default(unsqueeze_53);  unsqueeze_53 = None
    masked_fill_38 = torch.ops.aten.masked_fill.Scalar(add_171, bitwise_not_38, 0.0);  add_171 = bitwise_not_38 = None
    unsqueeze_54 = torch.ops.aten.unsqueeze.default(getitem_405, 5)
    permute_81 = torch.ops.aten.permute.default(unsqueeze_54, [0, 5, 1, 2, 3, 4]);  unsqueeze_54 = None
    unsqueeze_55 = torch.ops.aten.unsqueeze.default(getitem_408, 2);  getitem_408 = None
    unsqueeze_56 = torch.ops.aten.unsqueeze.default(unsqueeze_55, 3);  unsqueeze_55 = None
    unsqueeze_57 = torch.ops.aten.unsqueeze.default(unsqueeze_56, 4);  unsqueeze_56 = None
    unsqueeze_58 = torch.ops.aten.unsqueeze.default(unsqueeze_57, 5);  unsqueeze_57 = None
    permute_82 = torch.ops.aten.permute.default(unsqueeze_58, [2, 0, 3, 4, 5, 1]);  unsqueeze_58 = None
    permute_83 = torch.ops.aten.permute.default(permute_81, [0, 2, 3, 4, 5, 1]);  permute_81 = None
    view_556 = torch.ops.aten.view.default(permute_83, [1, 5652480, 16]);  permute_83 = None
    permute_84 = torch.ops.aten.permute.default(permute_82, [5, 1, 0, 2, 3, 4]);  permute_82 = None
    view_557 = torch.ops.aten.view.default(permute_84, [1, 16, 4]);  permute_84 = None
    bmm_38 = torch.ops.aten.bmm.default(view_556, view_557);  view_556 = view_557 = None
    view_558 = torch.ops.aten.view.default(bmm_38, [5, 276, 32, 128, 1, 4]);  bmm_38 = None
    permute_85 = torch.ops.aten.permute.default(view_558, [0, 5, 1, 2, 3, 4]);  view_558 = None
    view_559 = torch.ops.aten.view.default(permute_85, [5, 4, 276, 32, 128]);  permute_85 = None
    view_560 = torch.ops.aten.view.default(view_548, [5, 1, 276, 32, 128])
    bitwise_not_39 = torch.ops.aten.bitwise_not.default(view_560);  view_560 = None
    masked_fill_39 = torch.ops.aten.masked_fill.Scalar(view_559, bitwise_not_39, -10000);  view_559 = bitwise_not_39 = None
    native_layer_norm_default_70 = torch.ops.aten.native_layer_norm.default(masked_fill_38, [128], None, None, 0.1)
    getitem_411 = native_layer_norm_default_70[0];  native_layer_norm_default_70 = None
    t_184 = torch.ops.aten.t.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_0_single_layer_norm_lin_s_merged_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_0_single_layer_norm_lin_s_merged_weight = None
    clone_87 = torch.ops.aten.clone.default(view_546, memory_format = torch.contiguous_format)
    _unsafe_view_85 = torch.ops.aten._unsafe_view.default(clone_87, [44160, 128]);  clone_87 = None
    mm_152 = torch.ops.aten.mm.default(_unsafe_view_85, t_184);  _unsafe_view_85 = t_184 = None
    view_561 = torch.ops.aten.view.default(mm_152, [5, 8832, 256]);  mm_152 = None
    split_tensor_81 = torch.ops.aten.split.Tensor(view_561, 128, dim = -1);  view_561 = None
    getitem_414 = split_tensor_81[0]
    getitem_415 = split_tensor_81[1];  split_tensor_81 = None
    add_174 = torch.ops.aten.add.Tensor(getitem_414, 1);  getitem_414 = None
    mul_136 = torch.ops.aten.mul.Tensor(getitem_411, add_174);  getitem_411 = add_174 = None
    add_175 = torch.ops.aten.add.Tensor(mul_136, getitem_415);  mul_136 = getitem_415 = None
    unsqueeze_59 = torch.ops.aten.unsqueeze.default(add_175, 3);  add_175 = None
    unsqueeze_60 = torch.ops.aten.unsqueeze.default(unsqueeze_59, 4);  unsqueeze_59 = None
    unsqueeze_61 = torch.ops.aten.unsqueeze.default(unsqueeze_60, 5);  unsqueeze_60 = None
    permute_86 = torch.ops.aten.permute.default(unsqueeze_61, [3, 0, 4, 1, 5, 2]);  unsqueeze_61 = None
    unsqueeze_62 = torch.ops.aten.unsqueeze.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_0_to_qkv_weight, 4);  atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_0_to_qkv_weight = None
    unsqueeze_63 = torch.ops.aten.unsqueeze.default(unsqueeze_62, 5);  unsqueeze_62 = None
    permute_87 = torch.ops.aten.permute.default(unsqueeze_63, [0, 4, 1, 5, 2, 3]);  unsqueeze_63 = None
    permute_88 = torch.ops.aten.permute.default(permute_86, [1, 3, 5, 0, 2, 4]);  permute_86 = None
    view_562 = torch.ops.aten.view.default(permute_88, [1, 44160, 128]);  permute_88 = None
    permute_89 = torch.ops.aten.permute.default(permute_87, [5, 0, 2, 4, 1, 3]);  permute_87 = None
    view_563 = torch.ops.aten.view.default(permute_89, [1, 128, 384]);  permute_89 = None
    bmm_39 = torch.ops.aten.bmm.default(view_562, view_563);  view_562 = view_563 = None
    view_564 = torch.ops.aten.view.default(bmm_39, [5, 8832, 1, 3, 4, 32]);  bmm_39 = None
    permute_90 = torch.ops.aten.permute.default(view_564, [3, 0, 4, 1, 5, 2]);  view_564 = None
    view_565 = torch.ops.aten.view.default(permute_90, [3, 5, 4, 8832, 32]);  permute_90 = None
    clone_88 = torch.ops.aten.clone.default(view_565, memory_format = torch.contiguous_format);  view_565 = None
    _unsafe_view_86 = torch.ops.aten._unsafe_view.default(clone_88, [3, 20, 8832, 32]);  clone_88 = None
    unbind_int_5 = torch.ops.aten.unbind.int(_unsafe_view_86);  _unsafe_view_86 = None
    getitem_416 = unbind_int_5[0]
    getitem_417 = unbind_int_5[1]
    getitem_418 = unbind_int_5[2];  unbind_int_5 = None
    unsqueeze_64 = torch.ops.aten.unsqueeze.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_0_q_bias, 0);  atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_0_q_bias = None
    expand_82 = torch.ops.aten.expand.default(unsqueeze_64, [5, -1, -1]);  unsqueeze_64 = None
    clone_89 = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
    _unsafe_view_87 = torch.ops.aten._unsafe_view.default(clone_89, [20, 1, 32]);  clone_89 = None
    add_176 = torch.ops.aten.add.Tensor(getitem_416, _unsafe_view_87);  getitem_416 = _unsafe_view_87 = None
    view_566 = torch.ops.aten.view.default(add_176, [20, 276, 32, 32]);  add_176 = None
    slice_23 = torch.ops.aten.slice.Tensor(getitem_417, dim = 0, start = 0, end = 9223372036854775807);  getitem_417 = None
    slice_24 = torch.ops.aten.slice.Tensor(slice_23, dim = 2, start = 0, end = 9223372036854775807);  slice_23 = None
    index_13 = torch.ops.aten.index.Tensor(slice_24, [None, remainder_1]);  slice_24 = None
    slice_25 = torch.ops.aten.slice.Tensor(getitem_418, dim = 0, start = 0, end = 9223372036854775807);  getitem_418 = None
    slice_26 = torch.ops.aten.slice.Tensor(slice_25, dim = 2, start = 0, end = 9223372036854775807);  slice_25 = None
    index_14 = torch.ops.aten.index.Tensor(slice_26, [None, remainder_1]);  slice_26 = None
    view_567 = torch.ops.aten.view.default(masked_fill_39, [20, 276, 32, 128]);  masked_fill_39 = None
    expand_83 = torch.ops.aten.expand.default(view_567, [20, 276, 32, 128]);  view_567 = None
    _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_566, index_13, index_14, expand_83, False);  view_566 = index_13 = index_14 = expand_83 = None
    getitem_419 = _scaled_dot_product_efficient_attention_default_3[0];  _scaled_dot_product_efficient_attention_default_3 = None
    view_568 = torch.ops.aten.view.default(getitem_419, [5, 4, 276, 32, 32]);  getitem_419 = None
    permute_91 = torch.ops.aten.permute.default(view_568, [0, 2, 3, 1, 4]);  view_568 = None
    clone_90 = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    _unsafe_view_88 = torch.ops.aten._unsafe_view.default(clone_90, [5, 8832, 128]);  clone_90 = None
    t_185 = torch.ops.aten.t.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_0_out_proj_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_0_out_proj_weight = None
    clone_91 = torch.ops.aten.clone.default(view_546, memory_format = torch.contiguous_format)
    _unsafe_view_89 = torch.ops.aten._unsafe_view.default(clone_91, [44160, 128]);  clone_91 = None
    mm_153 = torch.ops.aten.mm.default(_unsafe_view_89, t_185);  _unsafe_view_89 = t_185 = None
    view_569 = torch.ops.aten.view.default(mm_153, [5, 8832, 128]);  mm_153 = None
    add_177 = torch.ops.aten.add.Tensor(view_569, atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_0_out_proj_bias);  view_569 = atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_0_out_proj_bias = None
    sigmoid_38 = torch.ops.aten.sigmoid.default(add_177);  add_177 = None
    mul_137 = torch.ops.aten.mul.Tensor(_unsafe_view_88, sigmoid_38);  _unsafe_view_88 = sigmoid_38 = None
    native_layer_norm_default_71 = torch.ops.aten.native_layer_norm.default(masked_fill_38, [128], None, None, 0.1)
    getitem_423 = native_layer_norm_default_71[0];  native_layer_norm_default_71 = None
    t_186 = torch.ops.aten.t.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_0_ada_ln_lin_s_merged_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_0_ada_ln_lin_s_merged_weight = None
    clone_92 = torch.ops.aten.clone.default(view_546, memory_format = torch.contiguous_format)
    _unsafe_view_90 = torch.ops.aten._unsafe_view.default(clone_92, [44160, 128]);  clone_92 = None
    mm_154 = torch.ops.aten.mm.default(_unsafe_view_90, t_186);  _unsafe_view_90 = t_186 = None
    view_570 = torch.ops.aten.view.default(mm_154, [5, 8832, 256]);  mm_154 = None
    split_tensor_82 = torch.ops.aten.split.Tensor(view_570, 128, dim = -1);  view_570 = None
    getitem_426 = split_tensor_82[0]
    getitem_427 = split_tensor_82[1];  split_tensor_82 = None
    add_178 = torch.ops.aten.add.Tensor(getitem_426, 1);  getitem_426 = None
    mul_138 = torch.ops.aten.mul.Tensor(getitem_423, add_178);  getitem_423 = add_178 = None
    add_179 = torch.ops.aten.add.Tensor(mul_138, getitem_427);  mul_138 = getitem_427 = None
    t_187 = torch.ops.aten.t.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_0_linear_a_nobias_double_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_0_linear_a_nobias_double_weight = None
    view_571 = torch.ops.aten.view.default(add_179, [44160, 128]);  add_179 = None
    mm_155 = torch.ops.aten.mm.default(view_571, t_187);  view_571 = t_187 = None
    view_572 = torch.ops.aten.view.default(mm_155, [5, 8832, 512]);  mm_155 = None
    split_tensor_83 = torch.ops.aten.split.Tensor(view_572, 256, dim = -1);  view_572 = None
    getitem_428 = split_tensor_83[0]
    getitem_429 = split_tensor_83[1];  split_tensor_83 = None
    silu_23 = torch.ops.aten.silu.default(getitem_428);  getitem_428 = None
    mul_139 = torch.ops.aten.mul.Tensor(silu_23, getitem_429);  silu_23 = getitem_429 = None
    t_188 = torch.ops.aten.t.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_0_linear_s_biasinit_m2_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_0_linear_s_biasinit_m2_weight = None
    clone_93 = torch.ops.aten.clone.default(view_546, memory_format = torch.contiguous_format)
    _unsafe_view_91 = torch.ops.aten._unsafe_view.default(clone_93, [44160, 128]);  clone_93 = None
    mm_156 = torch.ops.aten.mm.default(_unsafe_view_91, t_188);  _unsafe_view_91 = t_188 = None
    view_573 = torch.ops.aten.view.default(mm_156, [5, 8832, 128]);  mm_156 = None
    add_180 = torch.ops.aten.add.Tensor(view_573, atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_0_linear_s_biasinit_m2_bias);  view_573 = atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_0_linear_s_biasinit_m2_bias = None
    sigmoid_39 = torch.ops.aten.sigmoid.default(add_180);  add_180 = None
    t_189 = torch.ops.aten.t.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_0_linear_b_nobias_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_0_linear_b_nobias_weight = None
    view_574 = torch.ops.aten.view.default(mul_139, [44160, 256]);  mul_139 = None
    mm_157 = torch.ops.aten.mm.default(view_574, t_189);  view_574 = t_189 = None
    view_575 = torch.ops.aten.view.default(mm_157, [5, 8832, 128]);  mm_157 = None
    mul_140 = torch.ops.aten.mul.Tensor(sigmoid_39, view_575);  sigmoid_39 = view_575 = None
    add_181 = torch.ops.aten.add.Tensor(masked_fill_38, mul_140);  masked_fill_38 = mul_140 = None
    add_182 = torch.ops.aten.add.Tensor(add_181, mul_137);  add_181 = mul_137 = None
    unsqueeze_65 = torch.ops.aten.unsqueeze.default(view_550, -1)
    bitwise_not_40 = torch.ops.aten.bitwise_not.default(unsqueeze_65);  unsqueeze_65 = None
    masked_fill_40 = torch.ops.aten.masked_fill.Scalar(add_182, bitwise_not_40, 0.0);  add_182 = bitwise_not_40 = None
    unsqueeze_66 = torch.ops.aten.unsqueeze.default(getitem_405, 5)
    permute_92 = torch.ops.aten.permute.default(unsqueeze_66, [0, 5, 1, 2, 3, 4]);  unsqueeze_66 = None
    unsqueeze_67 = torch.ops.aten.unsqueeze.default(getitem_409, 2);  getitem_409 = None
    unsqueeze_68 = torch.ops.aten.unsqueeze.default(unsqueeze_67, 3);  unsqueeze_67 = None
    unsqueeze_69 = torch.ops.aten.unsqueeze.default(unsqueeze_68, 4);  unsqueeze_68 = None
    unsqueeze_70 = torch.ops.aten.unsqueeze.default(unsqueeze_69, 5);  unsqueeze_69 = None
    permute_93 = torch.ops.aten.permute.default(unsqueeze_70, [2, 0, 3, 4, 5, 1]);  unsqueeze_70 = None
    permute_94 = torch.ops.aten.permute.default(permute_92, [0, 2, 3, 4, 5, 1]);  permute_92 = None
    view_576 = torch.ops.aten.view.default(permute_94, [1, 5652480, 16]);  permute_94 = None
    permute_95 = torch.ops.aten.permute.default(permute_93, [5, 1, 0, 2, 3, 4]);  permute_93 = None
    view_577 = torch.ops.aten.view.default(permute_95, [1, 16, 4]);  permute_95 = None
    bmm_40 = torch.ops.aten.bmm.default(view_576, view_577);  view_576 = view_577 = None
    view_578 = torch.ops.aten.view.default(bmm_40, [5, 276, 32, 128, 1, 4]);  bmm_40 = None
    permute_96 = torch.ops.aten.permute.default(view_578, [0, 5, 1, 2, 3, 4]);  view_578 = None
    view_579 = torch.ops.aten.view.default(permute_96, [5, 4, 276, 32, 128]);  permute_96 = None
    view_580 = torch.ops.aten.view.default(view_548, [5, 1, 276, 32, 128])
    bitwise_not_41 = torch.ops.aten.bitwise_not.default(view_580);  view_580 = None
    masked_fill_41 = torch.ops.aten.masked_fill.Scalar(view_579, bitwise_not_41, -10000);  view_579 = bitwise_not_41 = None
    native_layer_norm_default_72 = torch.ops.aten.native_layer_norm.default(masked_fill_40, [128], None, None, 0.1)
    getitem_430 = native_layer_norm_default_72[0];  native_layer_norm_default_72 = None
    t_190 = torch.ops.aten.t.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_1_single_layer_norm_lin_s_merged_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_1_single_layer_norm_lin_s_merged_weight = None
    clone_94 = torch.ops.aten.clone.default(view_546, memory_format = torch.contiguous_format)
    _unsafe_view_92 = torch.ops.aten._unsafe_view.default(clone_94, [44160, 128]);  clone_94 = None
    mm_158 = torch.ops.aten.mm.default(_unsafe_view_92, t_190);  _unsafe_view_92 = t_190 = None
    view_581 = torch.ops.aten.view.default(mm_158, [5, 8832, 256]);  mm_158 = None
    split_tensor_84 = torch.ops.aten.split.Tensor(view_581, 128, dim = -1);  view_581 = None
    getitem_433 = split_tensor_84[0]
    getitem_434 = split_tensor_84[1];  split_tensor_84 = None
    add_183 = torch.ops.aten.add.Tensor(getitem_433, 1);  getitem_433 = None
    mul_141 = torch.ops.aten.mul.Tensor(getitem_430, add_183);  getitem_430 = add_183 = None
    add_184 = torch.ops.aten.add.Tensor(mul_141, getitem_434);  mul_141 = getitem_434 = None
    unsqueeze_71 = torch.ops.aten.unsqueeze.default(add_184, 3);  add_184 = None
    unsqueeze_72 = torch.ops.aten.unsqueeze.default(unsqueeze_71, 4);  unsqueeze_71 = None
    unsqueeze_73 = torch.ops.aten.unsqueeze.default(unsqueeze_72, 5);  unsqueeze_72 = None
    permute_97 = torch.ops.aten.permute.default(unsqueeze_73, [3, 0, 4, 1, 5, 2]);  unsqueeze_73 = None
    unsqueeze_74 = torch.ops.aten.unsqueeze.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_1_to_qkv_weight, 4);  atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_1_to_qkv_weight = None
    unsqueeze_75 = torch.ops.aten.unsqueeze.default(unsqueeze_74, 5);  unsqueeze_74 = None
    permute_98 = torch.ops.aten.permute.default(unsqueeze_75, [0, 4, 1, 5, 2, 3]);  unsqueeze_75 = None
    permute_99 = torch.ops.aten.permute.default(permute_97, [1, 3, 5, 0, 2, 4]);  permute_97 = None
    view_582 = torch.ops.aten.view.default(permute_99, [1, 44160, 128]);  permute_99 = None
    permute_100 = torch.ops.aten.permute.default(permute_98, [5, 0, 2, 4, 1, 3]);  permute_98 = None
    view_583 = torch.ops.aten.view.default(permute_100, [1, 128, 384]);  permute_100 = None
    bmm_41 = torch.ops.aten.bmm.default(view_582, view_583);  view_582 = view_583 = None
    view_584 = torch.ops.aten.view.default(bmm_41, [5, 8832, 1, 3, 4, 32]);  bmm_41 = None
    permute_101 = torch.ops.aten.permute.default(view_584, [3, 0, 4, 1, 5, 2]);  view_584 = None
    view_585 = torch.ops.aten.view.default(permute_101, [3, 5, 4, 8832, 32]);  permute_101 = None
    clone_95 = torch.ops.aten.clone.default(view_585, memory_format = torch.contiguous_format);  view_585 = None
    _unsafe_view_93 = torch.ops.aten._unsafe_view.default(clone_95, [3, 20, 8832, 32]);  clone_95 = None
    unbind_int_6 = torch.ops.aten.unbind.int(_unsafe_view_93);  _unsafe_view_93 = None
    getitem_435 = unbind_int_6[0]
    getitem_436 = unbind_int_6[1]
    getitem_437 = unbind_int_6[2];  unbind_int_6 = None
    unsqueeze_76 = torch.ops.aten.unsqueeze.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_1_q_bias, 0);  atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_1_q_bias = None
    expand_84 = torch.ops.aten.expand.default(unsqueeze_76, [5, -1, -1]);  unsqueeze_76 = None
    clone_96 = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
    _unsafe_view_94 = torch.ops.aten._unsafe_view.default(clone_96, [20, 1, 32]);  clone_96 = None
    add_185 = torch.ops.aten.add.Tensor(getitem_435, _unsafe_view_94);  getitem_435 = _unsafe_view_94 = None
    view_586 = torch.ops.aten.view.default(add_185, [20, 276, 32, 32]);  add_185 = None
    slice_27 = torch.ops.aten.slice.Tensor(getitem_436, dim = 0, start = 0, end = 9223372036854775807);  getitem_436 = None
    slice_28 = torch.ops.aten.slice.Tensor(slice_27, dim = 2, start = 0, end = 9223372036854775807);  slice_27 = None
    index_15 = torch.ops.aten.index.Tensor(slice_28, [None, remainder_1]);  slice_28 = None
    slice_29 = torch.ops.aten.slice.Tensor(getitem_437, dim = 0, start = 0, end = 9223372036854775807);  getitem_437 = None
    slice_30 = torch.ops.aten.slice.Tensor(slice_29, dim = 2, start = 0, end = 9223372036854775807);  slice_29 = None
    index_16 = torch.ops.aten.index.Tensor(slice_30, [None, remainder_1]);  slice_30 = None
    view_587 = torch.ops.aten.view.default(masked_fill_41, [20, 276, 32, 128]);  masked_fill_41 = None
    expand_85 = torch.ops.aten.expand.default(view_587, [20, 276, 32, 128]);  view_587 = None
    _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_586, index_15, index_16, expand_85, False);  view_586 = index_15 = index_16 = expand_85 = None
    getitem_438 = _scaled_dot_product_efficient_attention_default_4[0];  _scaled_dot_product_efficient_attention_default_4 = None
    view_588 = torch.ops.aten.view.default(getitem_438, [5, 4, 276, 32, 32]);  getitem_438 = None
    permute_102 = torch.ops.aten.permute.default(view_588, [0, 2, 3, 1, 4]);  view_588 = None
    clone_97 = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
    _unsafe_view_95 = torch.ops.aten._unsafe_view.default(clone_97, [5, 8832, 128]);  clone_97 = None
    t_191 = torch.ops.aten.t.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_1_out_proj_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_1_out_proj_weight = None
    clone_98 = torch.ops.aten.clone.default(view_546, memory_format = torch.contiguous_format)
    _unsafe_view_96 = torch.ops.aten._unsafe_view.default(clone_98, [44160, 128]);  clone_98 = None
    mm_159 = torch.ops.aten.mm.default(_unsafe_view_96, t_191);  _unsafe_view_96 = t_191 = None
    view_589 = torch.ops.aten.view.default(mm_159, [5, 8832, 128]);  mm_159 = None
    add_186 = torch.ops.aten.add.Tensor(view_589, atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_1_out_proj_bias);  view_589 = atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_1_out_proj_bias = None
    sigmoid_40 = torch.ops.aten.sigmoid.default(add_186);  add_186 = None
    mul_142 = torch.ops.aten.mul.Tensor(_unsafe_view_95, sigmoid_40);  _unsafe_view_95 = sigmoid_40 = None
    native_layer_norm_default_73 = torch.ops.aten.native_layer_norm.default(masked_fill_40, [128], None, None, 0.1)
    getitem_442 = native_layer_norm_default_73[0];  native_layer_norm_default_73 = None
    t_192 = torch.ops.aten.t.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_1_ada_ln_lin_s_merged_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_1_ada_ln_lin_s_merged_weight = None
    clone_99 = torch.ops.aten.clone.default(view_546, memory_format = torch.contiguous_format)
    _unsafe_view_97 = torch.ops.aten._unsafe_view.default(clone_99, [44160, 128]);  clone_99 = None
    mm_160 = torch.ops.aten.mm.default(_unsafe_view_97, t_192);  _unsafe_view_97 = t_192 = None
    view_590 = torch.ops.aten.view.default(mm_160, [5, 8832, 256]);  mm_160 = None
    split_tensor_85 = torch.ops.aten.split.Tensor(view_590, 128, dim = -1);  view_590 = None
    getitem_445 = split_tensor_85[0]
    getitem_446 = split_tensor_85[1];  split_tensor_85 = None
    add_187 = torch.ops.aten.add.Tensor(getitem_445, 1);  getitem_445 = None
    mul_143 = torch.ops.aten.mul.Tensor(getitem_442, add_187);  getitem_442 = add_187 = None
    add_188 = torch.ops.aten.add.Tensor(mul_143, getitem_446);  mul_143 = getitem_446 = None
    t_193 = torch.ops.aten.t.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_1_linear_a_nobias_double_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_1_linear_a_nobias_double_weight = None
    view_591 = torch.ops.aten.view.default(add_188, [44160, 128]);  add_188 = None
    mm_161 = torch.ops.aten.mm.default(view_591, t_193);  view_591 = t_193 = None
    view_592 = torch.ops.aten.view.default(mm_161, [5, 8832, 512]);  mm_161 = None
    split_tensor_86 = torch.ops.aten.split.Tensor(view_592, 256, dim = -1);  view_592 = None
    getitem_447 = split_tensor_86[0]
    getitem_448 = split_tensor_86[1];  split_tensor_86 = None
    silu_24 = torch.ops.aten.silu.default(getitem_447);  getitem_447 = None
    mul_144 = torch.ops.aten.mul.Tensor(silu_24, getitem_448);  silu_24 = getitem_448 = None
    t_194 = torch.ops.aten.t.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_1_linear_s_biasinit_m2_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_1_linear_s_biasinit_m2_weight = None
    clone_100 = torch.ops.aten.clone.default(view_546, memory_format = torch.contiguous_format)
    _unsafe_view_98 = torch.ops.aten._unsafe_view.default(clone_100, [44160, 128]);  clone_100 = None
    mm_162 = torch.ops.aten.mm.default(_unsafe_view_98, t_194);  _unsafe_view_98 = t_194 = None
    view_593 = torch.ops.aten.view.default(mm_162, [5, 8832, 128]);  mm_162 = None
    add_189 = torch.ops.aten.add.Tensor(view_593, atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_1_linear_s_biasinit_m2_bias);  view_593 = atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_1_linear_s_biasinit_m2_bias = None
    sigmoid_41 = torch.ops.aten.sigmoid.default(add_189);  add_189 = None
    t_195 = torch.ops.aten.t.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_1_linear_b_nobias_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_1_linear_b_nobias_weight = None
    view_594 = torch.ops.aten.view.default(mul_144, [44160, 256]);  mul_144 = None
    mm_163 = torch.ops.aten.mm.default(view_594, t_195);  view_594 = t_195 = None
    view_595 = torch.ops.aten.view.default(mm_163, [5, 8832, 128]);  mm_163 = None
    mul_145 = torch.ops.aten.mul.Tensor(sigmoid_41, view_595);  sigmoid_41 = view_595 = None
    add_190 = torch.ops.aten.add.Tensor(masked_fill_40, mul_145);  masked_fill_40 = mul_145 = None
    add_191 = torch.ops.aten.add.Tensor(add_190, mul_142);  add_190 = mul_142 = None
    unsqueeze_77 = torch.ops.aten.unsqueeze.default(view_550, -1);  view_550 = None
    bitwise_not_42 = torch.ops.aten.bitwise_not.default(unsqueeze_77);  unsqueeze_77 = None
    masked_fill_42 = torch.ops.aten.masked_fill.Scalar(add_191, bitwise_not_42, 0.0);  add_191 = bitwise_not_42 = None
    unsqueeze_78 = torch.ops.aten.unsqueeze.default(getitem_405, 5);  getitem_405 = None
    permute_103 = torch.ops.aten.permute.default(unsqueeze_78, [0, 5, 1, 2, 3, 4]);  unsqueeze_78 = None
    unsqueeze_79 = torch.ops.aten.unsqueeze.default(getitem_410, 2);  getitem_410 = None
    unsqueeze_80 = torch.ops.aten.unsqueeze.default(unsqueeze_79, 3);  unsqueeze_79 = None
    unsqueeze_81 = torch.ops.aten.unsqueeze.default(unsqueeze_80, 4);  unsqueeze_80 = None
    unsqueeze_82 = torch.ops.aten.unsqueeze.default(unsqueeze_81, 5);  unsqueeze_81 = None
    permute_104 = torch.ops.aten.permute.default(unsqueeze_82, [2, 0, 3, 4, 5, 1]);  unsqueeze_82 = None
    permute_105 = torch.ops.aten.permute.default(permute_103, [0, 2, 3, 4, 5, 1]);  permute_103 = None
    view_596 = torch.ops.aten.view.default(permute_105, [1, 5652480, 16]);  permute_105 = None
    permute_106 = torch.ops.aten.permute.default(permute_104, [5, 1, 0, 2, 3, 4]);  permute_104 = None
    view_597 = torch.ops.aten.view.default(permute_106, [1, 16, 4]);  permute_106 = None
    bmm_42 = torch.ops.aten.bmm.default(view_596, view_597);  view_596 = view_597 = None
    view_598 = torch.ops.aten.view.default(bmm_42, [5, 276, 32, 128, 1, 4]);  bmm_42 = None
    permute_107 = torch.ops.aten.permute.default(view_598, [0, 5, 1, 2, 3, 4]);  view_598 = None
    view_599 = torch.ops.aten.view.default(permute_107, [5, 4, 276, 32, 128]);  permute_107 = None
    view_600 = torch.ops.aten.view.default(view_548, [5, 1, 276, 32, 128]);  view_548 = None
    bitwise_not_43 = torch.ops.aten.bitwise_not.default(view_600);  view_600 = None
    masked_fill_43 = torch.ops.aten.masked_fill.Scalar(view_599, bitwise_not_43, -10000);  view_599 = bitwise_not_43 = None
    native_layer_norm_default_74 = torch.ops.aten.native_layer_norm.default(masked_fill_42, [128], None, None, 0.1)
    getitem_449 = native_layer_norm_default_74[0];  native_layer_norm_default_74 = None
    t_196 = torch.ops.aten.t.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_2_single_layer_norm_lin_s_merged_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_2_single_layer_norm_lin_s_merged_weight = None
    clone_101 = torch.ops.aten.clone.default(view_546, memory_format = torch.contiguous_format)
    _unsafe_view_99 = torch.ops.aten._unsafe_view.default(clone_101, [44160, 128]);  clone_101 = None
    mm_164 = torch.ops.aten.mm.default(_unsafe_view_99, t_196);  _unsafe_view_99 = t_196 = None
    view_601 = torch.ops.aten.view.default(mm_164, [5, 8832, 256]);  mm_164 = None
    split_tensor_87 = torch.ops.aten.split.Tensor(view_601, 128, dim = -1);  view_601 = None
    getitem_452 = split_tensor_87[0]
    getitem_453 = split_tensor_87[1];  split_tensor_87 = None
    add_192 = torch.ops.aten.add.Tensor(getitem_452, 1);  getitem_452 = None
    mul_146 = torch.ops.aten.mul.Tensor(getitem_449, add_192);  getitem_449 = add_192 = None
    add_193 = torch.ops.aten.add.Tensor(mul_146, getitem_453);  mul_146 = getitem_453 = None
    unsqueeze_83 = torch.ops.aten.unsqueeze.default(add_193, 3);  add_193 = None
    unsqueeze_84 = torch.ops.aten.unsqueeze.default(unsqueeze_83, 4);  unsqueeze_83 = None
    unsqueeze_85 = torch.ops.aten.unsqueeze.default(unsqueeze_84, 5);  unsqueeze_84 = None
    permute_108 = torch.ops.aten.permute.default(unsqueeze_85, [3, 0, 4, 1, 5, 2]);  unsqueeze_85 = None
    unsqueeze_86 = torch.ops.aten.unsqueeze.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_2_to_qkv_weight, 4);  atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_2_to_qkv_weight = None
    unsqueeze_87 = torch.ops.aten.unsqueeze.default(unsqueeze_86, 5);  unsqueeze_86 = None
    permute_109 = torch.ops.aten.permute.default(unsqueeze_87, [0, 4, 1, 5, 2, 3]);  unsqueeze_87 = None
    permute_110 = torch.ops.aten.permute.default(permute_108, [1, 3, 5, 0, 2, 4]);  permute_108 = None
    view_602 = torch.ops.aten.view.default(permute_110, [1, 44160, 128]);  permute_110 = None
    permute_111 = torch.ops.aten.permute.default(permute_109, [5, 0, 2, 4, 1, 3]);  permute_109 = None
    view_603 = torch.ops.aten.view.default(permute_111, [1, 128, 384]);  permute_111 = None
    bmm_43 = torch.ops.aten.bmm.default(view_602, view_603);  view_602 = view_603 = None
    view_604 = torch.ops.aten.view.default(bmm_43, [5, 8832, 1, 3, 4, 32]);  bmm_43 = None
    permute_112 = torch.ops.aten.permute.default(view_604, [3, 0, 4, 1, 5, 2]);  view_604 = None
    view_605 = torch.ops.aten.view.default(permute_112, [3, 5, 4, 8832, 32]);  permute_112 = None
    clone_102 = torch.ops.aten.clone.default(view_605, memory_format = torch.contiguous_format);  view_605 = None
    _unsafe_view_100 = torch.ops.aten._unsafe_view.default(clone_102, [3, 20, 8832, 32]);  clone_102 = None
    unbind_int_7 = torch.ops.aten.unbind.int(_unsafe_view_100);  _unsafe_view_100 = None
    getitem_454 = unbind_int_7[0]
    getitem_455 = unbind_int_7[1]
    getitem_456 = unbind_int_7[2];  unbind_int_7 = None
    unsqueeze_88 = torch.ops.aten.unsqueeze.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_2_q_bias, 0);  atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_2_q_bias = None
    expand_86 = torch.ops.aten.expand.default(unsqueeze_88, [5, -1, -1]);  unsqueeze_88 = None
    clone_103 = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
    _unsafe_view_101 = torch.ops.aten._unsafe_view.default(clone_103, [20, 1, 32]);  clone_103 = None
    add_194 = torch.ops.aten.add.Tensor(getitem_454, _unsafe_view_101);  getitem_454 = _unsafe_view_101 = None
    view_606 = torch.ops.aten.view.default(add_194, [20, 276, 32, 32]);  add_194 = None
    slice_31 = torch.ops.aten.slice.Tensor(getitem_455, dim = 0, start = 0, end = 9223372036854775807);  getitem_455 = None
    slice_32 = torch.ops.aten.slice.Tensor(slice_31, dim = 2, start = 0, end = 9223372036854775807);  slice_31 = None
    index_17 = torch.ops.aten.index.Tensor(slice_32, [None, remainder_1]);  slice_32 = None
    slice_33 = torch.ops.aten.slice.Tensor(getitem_456, dim = 0, start = 0, end = 9223372036854775807);  getitem_456 = None
    slice_34 = torch.ops.aten.slice.Tensor(slice_33, dim = 2, start = 0, end = 9223372036854775807);  slice_33 = None
    index_18 = torch.ops.aten.index.Tensor(slice_34, [None, remainder_1]);  slice_34 = remainder_1 = None
    view_607 = torch.ops.aten.view.default(masked_fill_43, [20, 276, 32, 128]);  masked_fill_43 = None
    expand_87 = torch.ops.aten.expand.default(view_607, [20, 276, 32, 128]);  view_607 = None
    _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(view_606, index_17, index_18, expand_87, False);  view_606 = index_17 = index_18 = expand_87 = None
    getitem_457 = _scaled_dot_product_efficient_attention_default_5[0];  _scaled_dot_product_efficient_attention_default_5 = None
    view_608 = torch.ops.aten.view.default(getitem_457, [5, 4, 276, 32, 32]);  getitem_457 = None
    permute_113 = torch.ops.aten.permute.default(view_608, [0, 2, 3, 1, 4]);  view_608 = None
    clone_104 = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
    _unsafe_view_102 = torch.ops.aten._unsafe_view.default(clone_104, [5, 8832, 128]);  clone_104 = None
    t_197 = torch.ops.aten.t.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_2_out_proj_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_2_out_proj_weight = None
    clone_105 = torch.ops.aten.clone.default(view_546, memory_format = torch.contiguous_format)
    _unsafe_view_103 = torch.ops.aten._unsafe_view.default(clone_105, [44160, 128]);  clone_105 = None
    mm_165 = torch.ops.aten.mm.default(_unsafe_view_103, t_197);  _unsafe_view_103 = t_197 = None
    view_609 = torch.ops.aten.view.default(mm_165, [5, 8832, 128]);  mm_165 = None
    add_195 = torch.ops.aten.add.Tensor(view_609, atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_2_out_proj_bias);  view_609 = atom_attention_decoder_atom_transformer_local_diffn_transformer_local_attentions_2_out_proj_bias = None
    sigmoid_42 = torch.ops.aten.sigmoid.default(add_195);  add_195 = None
    mul_147 = torch.ops.aten.mul.Tensor(_unsafe_view_102, sigmoid_42);  _unsafe_view_102 = sigmoid_42 = None
    native_layer_norm_default_75 = torch.ops.aten.native_layer_norm.default(masked_fill_42, [128], None, None, 0.1)
    getitem_461 = native_layer_norm_default_75[0];  native_layer_norm_default_75 = None
    t_198 = torch.ops.aten.t.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_2_ada_ln_lin_s_merged_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_2_ada_ln_lin_s_merged_weight = None
    clone_106 = torch.ops.aten.clone.default(view_546, memory_format = torch.contiguous_format)
    _unsafe_view_104 = torch.ops.aten._unsafe_view.default(clone_106, [44160, 128]);  clone_106 = None
    mm_166 = torch.ops.aten.mm.default(_unsafe_view_104, t_198);  _unsafe_view_104 = t_198 = None
    view_610 = torch.ops.aten.view.default(mm_166, [5, 8832, 256]);  mm_166 = None
    split_tensor_88 = torch.ops.aten.split.Tensor(view_610, 128, dim = -1);  view_610 = None
    getitem_464 = split_tensor_88[0]
    getitem_465 = split_tensor_88[1];  split_tensor_88 = None
    add_196 = torch.ops.aten.add.Tensor(getitem_464, 1);  getitem_464 = None
    mul_148 = torch.ops.aten.mul.Tensor(getitem_461, add_196);  getitem_461 = add_196 = None
    add_197 = torch.ops.aten.add.Tensor(mul_148, getitem_465);  mul_148 = getitem_465 = None
    t_199 = torch.ops.aten.t.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_2_linear_a_nobias_double_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_2_linear_a_nobias_double_weight = None
    view_611 = torch.ops.aten.view.default(add_197, [44160, 128]);  add_197 = None
    mm_167 = torch.ops.aten.mm.default(view_611, t_199);  view_611 = t_199 = None
    view_612 = torch.ops.aten.view.default(mm_167, [5, 8832, 512]);  mm_167 = None
    split_tensor_89 = torch.ops.aten.split.Tensor(view_612, 256, dim = -1);  view_612 = None
    getitem_466 = split_tensor_89[0]
    getitem_467 = split_tensor_89[1];  split_tensor_89 = None
    silu_25 = torch.ops.aten.silu.default(getitem_466);  getitem_466 = None
    mul_149 = torch.ops.aten.mul.Tensor(silu_25, getitem_467);  silu_25 = getitem_467 = None
    t_200 = torch.ops.aten.t.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_2_linear_s_biasinit_m2_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_2_linear_s_biasinit_m2_weight = None
    clone_107 = torch.ops.aten.clone.default(view_546, memory_format = torch.contiguous_format);  view_546 = None
    _unsafe_view_105 = torch.ops.aten._unsafe_view.default(clone_107, [44160, 128]);  clone_107 = None
    mm_168 = torch.ops.aten.mm.default(_unsafe_view_105, t_200);  _unsafe_view_105 = t_200 = None
    view_613 = torch.ops.aten.view.default(mm_168, [5, 8832, 128]);  mm_168 = None
    add_198 = torch.ops.aten.add.Tensor(view_613, atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_2_linear_s_biasinit_m2_bias);  view_613 = atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_2_linear_s_biasinit_m2_bias = None
    sigmoid_43 = torch.ops.aten.sigmoid.default(add_198);  add_198 = None
    t_201 = torch.ops.aten.t.default(atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_2_linear_b_nobias_weight);  atom_attention_decoder_atom_transformer_local_diffn_transformer_transitions_2_linear_b_nobias_weight = None
    view_614 = torch.ops.aten.view.default(mul_149, [44160, 256]);  mul_149 = None
    mm_169 = torch.ops.aten.mm.default(view_614, t_201);  view_614 = t_201 = None
    view_615 = torch.ops.aten.view.default(mm_169, [5, 8832, 128]);  mm_169 = None
    mul_150 = torch.ops.aten.mul.Tensor(sigmoid_43, view_615);  sigmoid_43 = view_615 = None
    add_199 = torch.ops.aten.add.Tensor(masked_fill_42, mul_150);  masked_fill_42 = mul_150 = None
    add_200 = torch.ops.aten.add.Tensor(add_199, mul_147);  add_199 = mul_147 = None
    native_layer_norm_default_76 = torch.ops.aten.native_layer_norm.default(add_200, [128], atom_attention_decoder_to_pos_updates_0_weight, atom_attention_decoder_to_pos_updates_0_bias, 1e-05);  add_200 = atom_attention_decoder_to_pos_updates_0_weight = atom_attention_decoder_to_pos_updates_0_bias = None
    getitem_468 = native_layer_norm_default_76[0];  native_layer_norm_default_76 = None
    t_202 = torch.ops.aten.t.default(atom_attention_decoder_to_pos_updates_1_weight);  atom_attention_decoder_to_pos_updates_1_weight = None
    view_616 = torch.ops.aten.view.default(getitem_468, [44160, 128]);  getitem_468 = None
    mm_170 = torch.ops.aten.mm.default(view_616, t_202);  view_616 = t_202 = None
    view_617 = torch.ops.aten.view.default(mm_170, [5, 8832, 3]);  mm_170 = None
    view_618 = torch.ops.aten.view.default(view_617, [1, 5, 8832, 3]);  view_617 = None
    mul_151 = torch.ops.aten.mul.Tensor(view_31, 16.0)
    pow_3 = torch.ops.aten.pow.Tensor_Scalar(view_31, 2)
    add_201 = torch.ops.aten.add.Tensor(pow_3, 256.0);  pow_3 = None
    pow_4 = torch.ops.aten.pow.Tensor_Scalar(add_201, -0.5);  add_201 = None
    mul_152 = torch.ops.aten.mul.Tensor(mul_151, pow_4);  mul_151 = pow_4 = None
    mul_153 = torch.ops.aten.mul.Tensor(view_618, mul_152);  view_618 = mul_152 = None
    pow_5 = torch.ops.aten.pow.Tensor_Scalar(view_31, 2);  view_31 = None
    add_202 = torch.ops.aten.add.Tensor(pow_5, 256.0);  pow_5 = None
    reciprocal = torch.ops.aten.reciprocal.default(add_202);  add_202 = None
    mul_154 = torch.ops.aten.mul.Tensor(reciprocal, 256.0);  reciprocal = None
    mul_155 = torch.ops.aten.mul.Tensor(arg354_1, mul_154);  arg354_1 = mul_154 = None
    add_203 = torch.ops.aten.add.Tensor(mul_155, mul_153);  mul_155 = mul_153 = None
    view_619 = torch.ops.aten.view.default(add_203, [5, 8832, 3]);  add_203 = None
    return pytree.tree_unflatten((view_619,), self._out_spec)
