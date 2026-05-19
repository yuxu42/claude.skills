// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "modeling_qwen25.hpp"

#include "modeling/ops/kv_cache.hpp"
#include "modeling/ops/llm.hpp"
#include "modeling/ops/op_policy.hpp"
#include "modeling/ops/ops.hpp"
#include "modeling/ops/shape.hpp"
#include "modeling/weights/weight_loader.hpp"

#include <cmath>
#include <openvino/core/except.hpp>
#include <openvino/openvino.hpp>
#include <string>

namespace {

auto set_name = [](auto node, const std::string& name) {
    node->output(0).set_names({name});
    node->set_friendly_name(name);
};

}  // namespace

namespace ov {
namespace pipeline {
namespace models {

// =====================================================================
// Qwen25Attention
// =====================================================================

Qwen25Attention::Qwen25Attention(
    BuilderContext& ctx, const std::string& name, const Qwen25Config& cfg, int32_t layer_idx, Module* parent)
    : Module(name, ctx, parent),
      layer_idx_(layer_idx),
      num_heads_(cfg.num_attention_heads),
      num_kv_heads_(cfg.kv_heads()),
      head_dim_(cfg.resolved_head_dim()),
      hidden_size_(cfg.hidden_size),
      scaling_(1.0f / std::sqrt(static_cast<float>(head_dim_))),
      attention_bias_(cfg.attention_bias) {
    if (num_heads_ <= 0 || head_dim_ <= 0 || num_kv_heads_ <= 0) {
        OPENVINO_THROW("Invalid Qwen2.5 attention configuration");
    }
    if (num_heads_ % num_kv_heads_ != 0) {
        OPENVINO_THROW("num_attention_heads must be divisible by num_key_value_heads");
    }

    q_proj_param_ = &register_parameter("q_proj.weight");
    k_proj_param_ = &register_parameter("k_proj.weight");
    v_proj_param_ = &register_parameter("v_proj.weight");
    o_proj_param_ = &register_parameter("o_proj.weight");

    q_bias_param_ = &register_parameter("q_proj.bias");
    k_bias_param_ = &register_parameter("k_proj.bias");
    v_bias_param_ = &register_parameter("v_proj.bias");

    if (!attention_bias_) {
        q_bias_param_->set_optional(true);
        k_bias_param_->set_optional(true);
        v_bias_param_->set_optional(true);
    }
}

const Tensor& Qwen25Attention::q_proj_weight() const {
    OPENVINO_ASSERT(q_proj_param_, "Qwen25Attention q_proj parameter is not registered");
    return q_proj_param_->value();
}

const Tensor& Qwen25Attention::k_proj_weight() const {
    OPENVINO_ASSERT(k_proj_param_, "Qwen25Attention k_proj parameter is not registered");
    return k_proj_param_->value();
}

const Tensor& Qwen25Attention::v_proj_weight() const {
    OPENVINO_ASSERT(v_proj_param_, "Qwen25Attention v_proj parameter is not registered");
    return v_proj_param_->value();
}

const Tensor& Qwen25Attention::o_proj_weight() const {
    OPENVINO_ASSERT(o_proj_param_, "Qwen25Attention o_proj parameter is not registered");
    return o_proj_param_->value();
}

const Tensor* Qwen25Attention::q_proj_bias() const {
    return (q_bias_param_ && q_bias_param_->is_bound()) ? &q_bias_param_->value() : nullptr;
}

const Tensor* Qwen25Attention::k_proj_bias() const {
    return (k_bias_param_ && k_bias_param_->is_bound()) ? &k_bias_param_->value() : nullptr;
}

const Tensor* Qwen25Attention::v_proj_bias() const {
    return (v_bias_param_ && v_bias_param_->is_bound()) ? &v_bias_param_->value() : nullptr;
}

Tensor Qwen25Attention::forward(const Tensor& hidden_states,
                                const Tensor& beam_idx,
                                const Tensor& rope_cos,
                                const Tensor& rope_sin,
                                const Tensor* causal_mask) const {
    auto* policy = &ctx().op_policy();

    // Project Q, K, V
    auto q = ops::linear(hidden_states, q_proj_weight());
    auto k = ops::linear(hidden_states, k_proj_weight());
    auto v = ops::linear(hidden_states, v_proj_weight());

    // Add bias if present
    if (auto* bias = q_proj_bias()) {
        q = q + *bias;
    }
    if (auto* bias = k_proj_bias()) {
        k = k + *bias;
    }
    if (auto* bias = v_proj_bias()) {
        v = v + *bias;
    }

    // Reshape to multi-head format and transpose
    auto q_heads = q.reshape({0, 0, num_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto k_heads = k.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});
    auto v_heads = v.reshape({0, 0, num_kv_heads_, head_dim_}).permute({0, 2, 1, 3});

    // Apply RoPE
    q_heads = ops::llm::apply_rope(q_heads, rope_cos, rope_sin, head_dim_, policy);
    k_heads = ops::llm::apply_rope(k_heads, rope_cos, rope_sin, head_dim_, policy);

    // KV Cache
    const std::string cache_prefix = full_path().empty() ? name() : full_path();
    auto cached = modeling::ops::append_kv_cache(k_heads, v_heads, beam_idx,
                                                  num_kv_heads_, head_dim_, cache_prefix, ctx());

    // GQA expansion
    auto k_expanded = ops::llm::repeat_kv(cached.first, num_heads_, num_kv_heads_, head_dim_);
    auto v_expanded = ops::llm::repeat_kv(cached.second, num_heads_, num_kv_heads_, head_dim_);

    // Use precomputed causal mask (shared across all layers)
    auto attn = ops::llm::sdpa(q_heads, k_expanded, v_expanded, scaling_, 3, causal_mask, false, policy);

    // Merge heads: [B, H, S, D] -> [B, S, H*D]
    const int64_t attn_hidden = static_cast<int64_t>(num_heads_) * static_cast<int64_t>(head_dim_);
    auto merged = attn.permute({0, 2, 1, 3}).reshape({0, 0, attn_hidden});

    return ops::linear(merged, o_proj_weight());
}

// =====================================================================
// Qwen25MLP
// =====================================================================

Qwen25MLP::Qwen25MLP(BuilderContext& ctx, const std::string& name, const Qwen25Config& cfg, Module* parent)
    : Module(name, ctx, parent) {
    if (!cfg.hidden_act.empty() && cfg.hidden_act != "silu") {
        OPENVINO_THROW("Unsupported Qwen2.5 MLP activation: ", cfg.hidden_act);
    }
    gate_proj_param_ = &register_parameter("gate_proj.weight");
    up_proj_param_ = &register_parameter("up_proj.weight");
    down_proj_param_ = &register_parameter("down_proj.weight");
}

const Tensor& Qwen25MLP::gate_proj_weight() const {
    OPENVINO_ASSERT(gate_proj_param_, "Qwen25MLP gate_proj parameter is not registered");
    return gate_proj_param_->value();
}

const Tensor& Qwen25MLP::up_proj_weight() const {
    OPENVINO_ASSERT(up_proj_param_, "Qwen25MLP up_proj parameter is not registered");
    return up_proj_param_->value();
}

const Tensor& Qwen25MLP::down_proj_weight() const {
    OPENVINO_ASSERT(down_proj_param_, "Qwen25MLP down_proj parameter is not registered");
    return down_proj_param_->value();
}

Tensor Qwen25MLP::forward(const Tensor& x) const {
    auto gate = ops::linear(x, gate_proj_weight());
    auto up = ops::linear(x, up_proj_weight());
    auto gated = ops::silu(gate) * up;
    return ops::linear(gated, down_proj_weight());
}

// =====================================================================
// Qwen25DecoderLayer
// =====================================================================

Qwen25DecoderLayer::Qwen25DecoderLayer(
    BuilderContext& ctx, const std::string& name, const Qwen25Config& cfg, int32_t layer_idx, Module* parent)
    : Module(name, ctx, parent),
      input_layernorm_(ctx, "input_layernorm", cfg.rms_norm_eps, this),
      self_attn_(ctx, "self_attn", cfg, layer_idx, this),
      post_attention_layernorm_(ctx, "post_attention_layernorm", cfg.rms_norm_eps, this),
      mlp_(ctx, "mlp", cfg, this) {
    register_module("input_layernorm", &input_layernorm_);
    register_module("self_attn", &self_attn_);
    register_module("post_attention_layernorm", &post_attention_layernorm_);
    register_module("mlp", &mlp_);
}

Tensor Qwen25DecoderLayer::forward(const Tensor& hidden_states,
                                   const Tensor& beam_idx,
                                   const Tensor& rope_cos,
                                   const Tensor& rope_sin,
                                   const Tensor* causal_mask) const {
    // Pre-attention norm
    auto normed = input_layernorm_.forward(hidden_states);

    // Self-attention + residual
    auto attn_out = self_attn_.forward(normed, beam_idx, rope_cos, rope_sin, causal_mask);
    auto residual = hidden_states + attn_out;

    // Post-attention norm + MLP + residual
    auto post_norm = post_attention_layernorm_.forward(residual);
    auto mlp_out = mlp_.forward(post_norm);
    return residual + mlp_out;
}

// =====================================================================
// Qwen25Model
// =====================================================================

Qwen25Model::Qwen25Model(BuilderContext& ctx, const Qwen25Config& cfg, Module* parent)
    : Module("model", ctx, parent),
      cfg_(cfg),
      embed_tokens_(ctx, "embed_tokens", this),
      layers_(),
      norm_(ctx, "norm", cfg.rms_norm_eps, this),
      head_dim_(cfg.resolved_head_dim()),
      rope_theta_(cfg.rope_theta) {
    if (head_dim_ <= 0) {
        OPENVINO_THROW("Invalid Qwen2.5 head dimension");
    }

    register_module("embed_tokens", &embed_tokens_);
    register_module("norm", &norm_);

    layers_.reserve(static_cast<size_t>(cfg.num_hidden_layers));
    for (int32_t i = 0; i < cfg.num_hidden_layers; ++i) {
        const std::string layer_name = "layers[" + std::to_string(i) + "]";
        layers_.emplace_back(ctx, layer_name, cfg, i, this);
        register_module(layer_name, &layers_.back());
    }
}

Tensor Qwen25Model::forward(const Tensor& input_ids,
                            const Tensor& position_ids,
                            const Tensor& beam_idx,
                            const Tensor& attention_mask) {
    auto hidden_states = embed_tokens_.forward(input_ids);

    // Build RoPE cos/sin from position_ids (computed once, shared across layers)
    auto cos_sin = ops::llm::rope_cos_sin(position_ids, head_dim_, rope_theta_, &ctx().op_policy());

    // Build causal mask ONCE from input dimensions (shared across all layers).
    // q_len = dim(input_ids, 1), kv_len = dim(attention_mask, 1)
    // This avoids per-layer ShapeOf/Gather/Range ops that break GPU fusion.
    auto q_len = Tensor(shape::dim(input_ids, 1), input_ids.context());
    auto kv_len = Tensor(shape::dim(attention_mask, 1), attention_mask.context());
    auto causal_mask = ops::llm::build_kv_causal_mask_with_attention_from_q_len(
        q_len, kv_len, attention_mask);

    // Pass through decoder layers
    for (auto& layer : layers_) {
        hidden_states = layer.forward(hidden_states, beam_idx, cos_sin.first, cos_sin.second, &causal_mask);
    }

    // Final RMSNorm
    return norm_.forward(hidden_states);
}

VocabEmbedding& Qwen25Model::embed_tokens() {
    return embed_tokens_;
}

// =====================================================================
// Qwen25ForCausalLM
// =====================================================================

Qwen25ForCausalLM::Qwen25ForCausalLM(BuilderContext& ctx, const Qwen25Config& cfg, Module* parent)
    : Module("", ctx, parent), cfg_(cfg), model_(ctx, cfg, this), lm_head_(ctx, "lm_head", this) {
    register_module("model", &model_);
    register_module("lm_head", &lm_head_);

    if (cfg_.tie_word_embeddings) {
        lm_head_.tie_to(model_.embed_tokens().weight_param());
    }
}

Tensor Qwen25ForCausalLM::forward(const Tensor& input_ids,
                                  const Tensor& position_ids,
                                  const Tensor& beam_idx,
                                  const Tensor& attention_mask) {
    auto hidden = model_.forward(input_ids, position_ids, beam_idx, attention_mask);
    return lm_head_.forward(hidden);
}

// =====================================================================
// Factory: create_qwen25_model
// =====================================================================

std::shared_ptr<ov::Model> create_qwen25_model(const Qwen25Config& cfg,
                                               weights::WeightSource& source,
                                               weights::WeightFinalizer& finalizer) {
    OpPolicy policy;
    policy.use_internal_rope = true;
    BuilderContext ctx(policy);

    Qwen25ForCausalLM model(ctx, cfg);

    // Load weights
    weights::load_model(model, source, finalizer);

    // Re-establish tie after loading (in case lm_head.weight was separately loaded)
    if (cfg.tie_word_embeddings) {
        model.get_parameter("lm_head.weight").tie_to(
            model.get_parameter("model.embed_tokens.weight"));
    }

    // Create model inputs
    auto input_ids = ctx.parameter("input_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto attention_mask = ctx.parameter("attention_mask", ov::element::i64, ov::PartialShape{-1, -1});
    auto position_ids = ctx.parameter("position_ids", ov::element::i64, ov::PartialShape{-1, -1});
    auto beam_idx = ctx.parameter("beam_idx", ov::element::i32, ov::PartialShape{-1});

    // Forward pass
    auto logits = model.forward(input_ids, position_ids, beam_idx, attention_mask);

    // Build ov::Model
    auto result = std::make_shared<ov::op::v0::Result>(logits.output());
    set_name(result, "logits");
    auto ov_model = ctx.build_model({result->output(0)});

    // Set runtime hints for optimal performance
    ov_model->set_rt_info(ov::element::f16, {"runtime_options", ov::hint::kv_cache_precision.name()});
    ov_model->set_rt_info(8.0f, {"runtime_options", ov::hint::activations_scale_factor.name()});

    return ov_model;
}

}  // namespace models
}  // namespace pipeline
}  // namespace ov
