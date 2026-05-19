// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "modeling/builder_context.hpp"
#include "modeling/layers/lm_head.hpp"
#include "modeling/layers/rms_norm.hpp"
#include "modeling/layers/vocab_embedding.hpp"
#include "modeling/module.hpp"
#include "modeling/ops/tensor.hpp"
#include "modeling/weights/weight_finalizer.hpp"
#include "modeling/weights/weight_loader.hpp"
#include "modeling/weights/weight_source.hpp"

namespace ov {
class Model;
}

namespace ov {
namespace pipeline {
namespace models {

struct Qwen25Config {
    int32_t hidden_size = 896;
    int32_t num_attention_heads = 14;
    int32_t num_key_value_heads = 2;
    int32_t head_dim = 64;
    int32_t intermediate_size = 4864;
    int32_t num_hidden_layers = 24;
    int32_t vocab_size = 151936;
    int32_t max_position_embeddings = 131072;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    std::string hidden_act = "silu";
    bool tie_word_embeddings = true;
    bool attention_bias = true;

    int32_t resolved_head_dim() const {
        if (head_dim > 0)
            return head_dim;
        if (num_attention_heads > 0)
            return hidden_size / num_attention_heads;
        return 0;
    }

    int32_t kv_heads() const {
        return num_key_value_heads > 0 ? num_key_value_heads : num_attention_heads;
    }
};

class Qwen25Attention : public Module {
public:
    Qwen25Attention(BuilderContext& ctx,
                    const std::string& name,
                    const Qwen25Config& cfg,
                    int32_t layer_idx,
                    Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin,
                   const Tensor* causal_mask) const;

private:
    const Tensor& q_proj_weight() const;
    const Tensor& k_proj_weight() const;
    const Tensor& v_proj_weight() const;
    const Tensor& o_proj_weight() const;
    const Tensor* q_proj_bias() const;
    const Tensor* k_proj_bias() const;
    const Tensor* v_proj_bias() const;

    int32_t layer_idx_ = 0;
    int32_t num_heads_ = 0;
    int32_t num_kv_heads_ = 0;
    int32_t head_dim_ = 0;
    int32_t hidden_size_ = 0;
    float scaling_ = 1.0f;
    bool attention_bias_ = false;

    WeightParameter* q_proj_param_ = nullptr;
    WeightParameter* k_proj_param_ = nullptr;
    WeightParameter* v_proj_param_ = nullptr;
    WeightParameter* o_proj_param_ = nullptr;

    WeightParameter* q_bias_param_ = nullptr;
    WeightParameter* k_bias_param_ = nullptr;
    WeightParameter* v_bias_param_ = nullptr;
};

class Qwen25MLP : public Module {
public:
    Qwen25MLP(BuilderContext& ctx, const std::string& name, const Qwen25Config& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& x) const;

private:
    const Tensor& gate_proj_weight() const;
    const Tensor& up_proj_weight() const;
    const Tensor& down_proj_weight() const;

    WeightParameter* gate_proj_param_ = nullptr;
    WeightParameter* up_proj_param_ = nullptr;
    WeightParameter* down_proj_param_ = nullptr;
};

class Qwen25DecoderLayer : public Module {
public:
    Qwen25DecoderLayer(BuilderContext& ctx,
                       const std::string& name,
                       const Qwen25Config& cfg,
                       int32_t layer_idx,
                       Module* parent = nullptr);

    Tensor forward(const Tensor& hidden_states,
                   const Tensor& rope_cos,
                   const Tensor& rope_sin,
                   const Tensor* causal_mask) const;

private:
    RMSNorm input_layernorm_;
    Qwen25Attention self_attn_;
    RMSNorm post_attention_layernorm_;
    Qwen25MLP mlp_;
};

class Qwen25Model : public Module {
public:
    Qwen25Model(BuilderContext& ctx, const Qwen25Config& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& attention_mask);

    VocabEmbedding& embed_tokens();

private:
    Qwen25Config cfg_;
    VocabEmbedding embed_tokens_;
    std::vector<Qwen25DecoderLayer> layers_;
    RMSNorm norm_;
    int32_t head_dim_ = 0;
    float rope_theta_ = 1000000.0f;
};

class Qwen25ForCausalLM : public Module {
public:
    Qwen25ForCausalLM(BuilderContext& ctx, const Qwen25Config& cfg, Module* parent = nullptr);

    Tensor forward(const Tensor& input_ids,
                   const Tensor& position_ids,
                   const Tensor& attention_mask);

private:
    Qwen25Config cfg_;
    Qwen25Model model_;
    LMHead lm_head_;
};

std::shared_ptr<ov::Model> create_qwen25_model(const Qwen25Config& cfg,
                                               weights::WeightSource& source,
                                               weights::WeightFinalizer& finalizer);

}  // namespace models
}  // namespace pipeline
}  // namespace ov
