// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Standalone runner for Qwen2.5 model using OpenVINO modeling API.
// Loads weights from HuggingFace safetensors, builds the OV model graph,
// compiles it, and runs greedy text generation.

#include "modeling_qwen25.hpp"

#include "safetensors_utils/safetensors_loader.hpp"
#include "safetensors_utils/safetensors_weight_finalizer.hpp"
#include "safetensors_utils/safetensors_weight_source.hpp"

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace ov::pipeline;
namespace st = ov::pipeline::safetensors;

namespace {

/// Compress model weights to FP16.
/// Converts FP32 and BF16 Constant nodes to FP16.
/// Intel Arc GPUs have native FP16 support but not BF16, so this avoids
/// runtime BF16→FP32 conversions on GPU and halves memory bandwidth vs FP32.
void compress_weights_to_fp16(std::shared_ptr<ov::Model>& model) {
    size_t converted = 0;
    size_t total_bytes_before = 0;
    size_t total_bytes_after = 0;

    for (auto& op : model->get_ordered_ops()) {
        auto constant = std::dynamic_pointer_cast<ov::opset13::Constant>(op);
        if (!constant)
            continue;

        auto src_type = constant->get_element_type();
        if (src_type != ov::element::f32 && src_type != ov::element::bf16)
            continue;

        // Only compress weight-like tensors (>= 512 elements)
        auto shape = constant->get_shape();
        size_t num_elements = ov::shape_size(shape);
        if (num_elements < 512)
            continue;

        total_bytes_before += num_elements * src_type.size();

        // Convert to FP16
        std::vector<ov::float16> fp16_data(num_elements);
        if (src_type == ov::element::f32) {
            const float* src = constant->get_data_ptr<float>();
            for (size_t i = 0; i < num_elements; ++i) {
                fp16_data[i] = ov::float16(src[i]);
            }
        } else {
            // BF16 -> FP16: go through float to handle range correctly
            const auto* src = reinterpret_cast<const uint16_t*>(constant->get_data_ptr());
            for (size_t i = 0; i < num_elements; ++i) {
                uint32_t fp32_bits = static_cast<uint32_t>(src[i]) << 16;
                float f;
                std::memcpy(&f, &fp32_bits, sizeof(f));
                fp16_data[i] = ov::float16(f);
            }
        }

        auto new_constant = std::make_shared<ov::opset13::Constant>(
            ov::element::f16, shape, fp16_data.data());
        new_constant->set_friendly_name(constant->get_friendly_name());

        // Replace the constant in-place. Keep Convert to FP32 only if the
        // original was FP32 (to preserve graph type consistency). For BF16
        // sources, the existing Convert node in the graph already handles it.
        if (src_type == ov::element::f32) {
            auto convert = std::make_shared<ov::opset13::Convert>(
                new_constant->output(0), ov::element::f32);
            ov::replace_node(constant, convert);
        } else {
            // BF16 case: the graph already has a Convert(BF16->F32) after the
            // constant. Replace the BF16 constant with FP16 — the existing
            // Convert node will now do FP16->F32 instead (same semantics).
            constant->output(0).replace(new_constant->output(0));
        }
        total_bytes_after += num_elements * sizeof(ov::float16);
        converted++;
    }

    std::cout << "[Compress] Converted " << converted << " weight tensors to FP16" << std::endl;
    std::cout << "[Compress] Memory: " << (total_bytes_before / (1024 * 1024)) << "MB -> "
              << (total_bytes_after / (1024 * 1024)) << "MB ("
              << (100 - 100 * total_bytes_after / std::max(total_bytes_before, size_t(1))) << "% reduction)" << std::endl;
}

/// Parse Qwen2.5 config from HuggingFace config.json
models::Qwen25Config load_qwen25_config(const std::filesystem::path& model_dir) {
    auto config_path = model_dir / "config.json";
    if (!std::filesystem::exists(config_path)) {
        throw std::runtime_error("config.json not found in " + model_dir.string());
    }

    std::ifstream f(config_path);
    nlohmann::json j;
    f >> j;

    models::Qwen25Config cfg;
    cfg.hidden_size = j.value("hidden_size", cfg.hidden_size);
    cfg.num_attention_heads = j.value("num_attention_heads", cfg.num_attention_heads);
    cfg.num_key_value_heads = j.value("num_key_value_heads", cfg.num_key_value_heads);
    cfg.head_dim = j.value("head_dim", 0);
    if (cfg.head_dim == 0 && cfg.num_attention_heads > 0) {
        cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
    }
    cfg.intermediate_size = j.value("intermediate_size", cfg.intermediate_size);
    cfg.num_hidden_layers = j.value("num_hidden_layers", cfg.num_hidden_layers);
    cfg.vocab_size = j.value("vocab_size", cfg.vocab_size);
    cfg.max_position_embeddings = j.value("max_position_embeddings", cfg.max_position_embeddings);
    cfg.rms_norm_eps = j.value("rms_norm_eps", cfg.rms_norm_eps);
    cfg.rope_theta = j.value("rope_theta", cfg.rope_theta);
    cfg.hidden_act = j.value("hidden_act", cfg.hidden_act);
    cfg.tie_word_embeddings = j.value("tie_word_embeddings", cfg.tie_word_embeddings);

    // Qwen2/Qwen2.5 has attention bias on Q, K, V projections by default.
    // The config.json may not explicitly list this field.
    if (j.contains("attention_bias")) {
        cfg.attention_bias = j["attention_bias"].get<bool>();
    } else {
        // Qwen2 architecture defaults to having Q/K/V bias
        cfg.attention_bias = true;
    }

    return cfg;
}

/// Simple greedy generation loop (no tokenizer - uses raw token IDs)
void greedy_generate(ov::InferRequest& infer_request,
                     const std::vector<int64_t>& input_ids,
                     int max_new_tokens,
                     int eos_token_id) {
    int64_t batch_size = 1;
    int64_t seq_len = static_cast<int64_t>(input_ids.size());

    // Set input_ids
    auto input_ids_tensor = ov::Tensor(ov::element::i64, {1, static_cast<size_t>(seq_len)});
    std::copy(input_ids.begin(), input_ids.end(), input_ids_tensor.data<int64_t>());
    infer_request.set_tensor("input_ids", input_ids_tensor);

    // Set attention_mask (all 1s for prefill)
    auto attn_mask_tensor = ov::Tensor(ov::element::i64, {1, static_cast<size_t>(seq_len)});
    std::fill_n(attn_mask_tensor.data<int64_t>(), seq_len, 1);
    infer_request.set_tensor("attention_mask", attn_mask_tensor);

    // Set position_ids (0, 1, 2, ..., seq_len-1)
    auto pos_ids_tensor = ov::Tensor(ov::element::i64, {1, static_cast<size_t>(seq_len)});
    for (int64_t i = 0; i < seq_len; ++i) {
        pos_ids_tensor.data<int64_t>()[i] = i;
    }
    infer_request.set_tensor("position_ids", pos_ids_tensor);

    // Set beam_idx
    auto beam_idx_tensor = ov::Tensor(ov::element::i32, {1});
    beam_idx_tensor.data<int32_t>()[0] = 0;
    infer_request.set_tensor("beam_idx", beam_idx_tensor);

    std::vector<int64_t> generated_tokens;
    int64_t next_pos = seq_len;

    // Prefill
    std::cout << "[Generate] Running prefill with " << seq_len << " tokens..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    infer_request.infer();
    auto prefill_end = std::chrono::high_resolution_clock::now();
    auto prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(prefill_end - start).count();
    std::cout << "[Generate] Prefill done in " << prefill_ms << "ms" << std::endl;

    // Get logits and sample greedily
    auto logits_tensor = infer_request.get_tensor("logits");
    auto logits_shape = logits_tensor.get_shape();
    size_t vocab_size = logits_shape.back();
    const float* logits_data = logits_tensor.data<float>();

    // Take argmax of last token position
    size_t last_pos_offset = (logits_shape[1] - 1) * vocab_size;
    int64_t next_token = 0;
    float max_logit = logits_data[last_pos_offset];
    for (size_t v = 1; v < vocab_size; ++v) {
        if (logits_data[last_pos_offset + v] > max_logit) {
            max_logit = logits_data[last_pos_offset + v];
            next_token = static_cast<int64_t>(v);
        }
    }
    generated_tokens.push_back(next_token);
    std::cout << "[Generate] Token 0: " << next_token << std::endl;

    // Decode loop
    for (int step = 1; step < max_new_tokens; ++step) {
        if (next_token == eos_token_id) {
            std::cout << "[Generate] EOS reached at step " << step << std::endl;
            break;
        }

        // Single token input
        auto step_ids = ov::Tensor(ov::element::i64, {1, 1});
        step_ids.data<int64_t>()[0] = next_token;
        infer_request.set_tensor("input_ids", step_ids);

        // Extend attention mask
        auto step_attn = ov::Tensor(ov::element::i64, {1, static_cast<size_t>(next_pos + 1)});
        std::fill_n(step_attn.data<int64_t>(), next_pos + 1, 1);
        infer_request.set_tensor("attention_mask", step_attn);

        // Position for this token
        auto step_pos = ov::Tensor(ov::element::i64, {1, 1});
        step_pos.data<int64_t>()[0] = next_pos;
        infer_request.set_tensor("position_ids", step_pos);

        infer_request.infer();

        // Get next token (logits shape is [1, 1, vocab_size] for decode step)
        logits_tensor = infer_request.get_tensor("logits");
        const float* decode_logits = logits_tensor.data<float>();
        next_token = 0;
        max_logit = decode_logits[0];
        for (size_t v = 1; v < vocab_size; ++v) {
            if (decode_logits[v] > max_logit) {
                max_logit = decode_logits[v];
                next_token = static_cast<int64_t>(v);
            }
        }
        generated_tokens.push_back(next_token);
        next_pos++;

        if (step % 10 == 0) {
            std::cout << "[Generate] Step " << step << ", token: " << next_token << std::endl;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - prefill_end).count();

    std::cout << "\n[Generate] Done!" << std::endl;
    std::cout << "[Generate] Generated " << generated_tokens.size() << " tokens" << std::endl;
    std::cout << "[Generate] Total time: " << total_ms << "ms" << std::endl;
    std::cout << "[Generate] Prefill: " << prefill_ms << "ms, Decode: " << decode_ms << "ms" << std::endl;
    if (generated_tokens.size() > 1) {
        double tps = static_cast<double>(generated_tokens.size() - 1) * 1000.0 / static_cast<double>(decode_ms);
        std::cout << "[Generate] Decode throughput: " << tps << " tokens/sec" << std::endl;
    }

    std::cout << "\n[Generate] Output token IDs: ";
    for (auto t : generated_tokens) {
        std::cout << t << " ";
    }
    std::cout << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_dir> [options] [prompt_token_ids...]\n"
                  << "  model_dir: path to HuggingFace Qwen2.5 model directory with safetensors\n"
                  << "\nOptions:\n"
                  << "  --device DEVICE      OpenVINO device (CPU, GPU, GPU.0, GPU.1) [default: CPU]\n"
                  << "  --export-ir PATH     Export model to OpenVINO IR format (.xml + .bin)\n"
                  << "  --load-ir PATH       Load model from IR format (.xml file)\n"
                  << "  --compress-weights   Compress FP32 weights to FP16 (halves memory, faster on GPU)\n"
                  << "\n"
                  << "  prompt_token_ids: space-separated token IDs (default: BOS token 151643)\n"
                  << "\nExamples:\n"
                  << "  # Run with FP16 weights on GPU\n"
                  << "  " << argv[0] << " /path/to/model --compress-weights --device GPU 151643\n"
                  << "\n"
                  << "  # Export compressed IR\n"
                  << "  " << argv[0] << " /path/to/model --compress-weights --export-ir model_fp16.xml\n"
                  << "\n"
                  << "  # Run from IR\n"
                  << "  " << argv[0] << " /path/to/model --load-ir model_fp16.xml --device GPU 151643\n";
        return 1;
    }

    std::filesystem::path model_dir = argv[1];
    std::string device = "CPU";
    std::string export_ir_path;
    std::string load_ir_path;
    bool compress_weights = false;

    // Parse arguments
    std::vector<int64_t> prompt_tokens;
    int arg_idx = 2;
    while (arg_idx < argc) {
        std::string arg = argv[arg_idx];
        if (arg == "--device" && arg_idx + 1 < argc) {
            device = argv[arg_idx + 1];
            arg_idx += 2;
        } else if (arg == "--export-ir" && arg_idx + 1 < argc) {
            export_ir_path = argv[arg_idx + 1];
            arg_idx += 2;
        } else if (arg == "--load-ir" && arg_idx + 1 < argc) {
            load_ir_path = argv[arg_idx + 1];
            arg_idx += 2;
        } else if (arg == "--compress-weights") {
            compress_weights = true;
            arg_idx++;
        } else {
            prompt_tokens.push_back(std::stoll(arg));
            arg_idx++;
        }
    }
    if (prompt_tokens.empty()) {
        prompt_tokens = {151643};  // <|im_start|>
    }

    try {
        ov::Core core;
        std::shared_ptr<ov::Model> ov_model;

        // Step 1: Load or build model
        if (!load_ir_path.empty()) {
            // Load from IR
            std::cout << "[Main] Loading model from IR: " << load_ir_path << std::endl;
            auto load_start = std::chrono::high_resolution_clock::now();
            ov_model = core.read_model(load_ir_path);
            auto load_end = std::chrono::high_resolution_clock::now();
            auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();
            std::cout << "[Main] Model loaded in " << load_ms << "ms" << std::endl;
        } else {
            // Build from safetensors
            std::cout << "[Main] Loading config from: " << model_dir << std::endl;
            auto cfg = load_qwen25_config(model_dir);
            std::cout << "[Main] Model config:" << std::endl;
            std::cout << "  hidden_size: " << cfg.hidden_size << std::endl;
            std::cout << "  num_hidden_layers: " << cfg.num_hidden_layers << std::endl;
            std::cout << "  num_attention_heads: " << cfg.num_attention_heads << std::endl;
            std::cout << "  num_key_value_heads: " << cfg.kv_heads() << std::endl;
            std::cout << "  head_dim: " << cfg.resolved_head_dim() << std::endl;
            std::cout << "  vocab_size: " << cfg.vocab_size << std::endl;
            std::cout << "  rope_theta: " << cfg.rope_theta << std::endl;
            std::cout << "  tie_word_embeddings: " << (cfg.tie_word_embeddings ? "true" : "false") << std::endl;
            std::cout << "  attention_bias: " << (cfg.attention_bias ? "true" : "false") << std::endl;

            std::cout << "\n[Main] Loading safetensors weights..." << std::endl;
            auto load_start = std::chrono::high_resolution_clock::now();
            auto st_data = st::load_safetensors(model_dir);
            auto load_end = std::chrono::high_resolution_clock::now();
            auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();
            std::cout << "[Main] Weights loaded in " << load_ms << "ms ("
                      << st_data.tensor_infos.size() << " tensors)" << std::endl;

            std::cout << "\n[Main] Building Qwen2.5 model graph..." << std::endl;
            auto build_start = std::chrono::high_resolution_clock::now();
            st::SafetensorsWeightSource source(std::move(st_data));
            st::SafetensorsWeightFinalizer finalizer;
            ov_model = models::create_qwen25_model(cfg, source, finalizer);
            auto build_end = std::chrono::high_resolution_clock::now();
            auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start).count();
            std::cout << "[Main] Model graph built in " << build_ms << "ms" << std::endl;
        }

        // Step 2: Compress weights to FP16 if requested
        if (compress_weights) {
            std::cout << "\n[Main] Compressing weights to FP16..." << std::endl;
            auto compress_start = std::chrono::high_resolution_clock::now();
            compress_weights_to_fp16(ov_model);
            auto compress_end = std::chrono::high_resolution_clock::now();
            auto compress_ms = std::chrono::duration_cast<std::chrono::milliseconds>(compress_end - compress_start).count();
            std::cout << "[Main] Compression done in " << compress_ms << "ms" << std::endl;
        }

        // Step 3: Export to IR if requested
        if (!export_ir_path.empty()) {
            std::cout << "\n[Main] Exporting model to IR: " << export_ir_path << std::endl;
            auto export_start = std::chrono::high_resolution_clock::now();
            ov::serialize(ov_model, export_ir_path);
            auto export_end = std::chrono::high_resolution_clock::now();
            auto export_ms = std::chrono::duration_cast<std::chrono::milliseconds>(export_end - export_start).count();
            std::cout << "[Main] Model exported in " << export_ms << "ms" << std::endl;

            // If only exporting, exit here
            if (prompt_tokens.size() == 1 && prompt_tokens[0] == 151643) {
                std::cout << "[Main] Export complete. Use --load-ir to run inference." << std::endl;
                return 0;
            }
        }

        // Step 3: Compile with OpenVINO
        std::cout << "\n[Main] Compiling model for device: " << device << std::endl;
        auto compile_start = std::chrono::high_resolution_clock::now();
        ov::AnyMap compile_props;
        if (device.find("GPU") != std::string::npos) {
            compile_props[ov::hint::inference_precision.name()] = ov::element::f16;
            compile_props[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::PERFORMANCE;
        }
        auto compiled_model = core.compile_model(ov_model, device, compile_props);
        auto compile_end = std::chrono::high_resolution_clock::now();
        auto compile_ms = std::chrono::duration_cast<std::chrono::milliseconds>(compile_end - compile_start).count();
        std::cout << "[Main] Compilation done in " << compile_ms << "ms" << std::endl;

        // Step 4: Run inference
        auto infer_request = compiled_model.create_infer_request();

        std::cout << "\n[Main] Starting generation with " << prompt_tokens.size()
                  << " prompt tokens..." << std::endl;
        int eos_token_id = 151645;  // <|im_end|> for Qwen2.5
        greedy_generate(infer_request, prompt_tokens, 50, eos_token_id);

    } catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
