#include <stdio.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include "progress.h"
#include "llama.h"

static void log_silencioso(ggml_log_level level, const char* text, void* user_data) {
    // não faz nada
}

int main()
{
    llama_log_set(log_silencioso, nullptr);

    std::cout << "Iniciando programa..." << std::flush;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "\r\033[2K" << std::flush;

    std::cout << "Carregando modelo..." << std::flush;

    // Init llama.cpp
    llama_backend_init();

    // Load the model
    llama_model_params model_params = llama_model_default_params();
    const char *model_path = "./llama.cpp/models/qwen2-0.5b-instruct-q4_k_m.gguf";
    llama_model *model = llama_model_load_from_file(model_path, model_params);

    // Check if the model was loaded successfully
    if (model == nullptr)
    {
        std::cout << "\r\033[2K" << std::flush;
        std::cerr << "Erro ao carregar o modelo!" << std::endl;
        return 1;
    }

    std::cout << "\r\033[2K" << std::flush;
    std::cout << "\rModelo carregado com sucesso!" << std::flush;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "\r\033[2K" << std::flush;

    std::cout << "Criando contexto..." << std::flush;

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;

    llama_context *ctx = llama_init_from_model(model, ctx_params);

    std::cout << "\r\033[2K" << std::flush;
    std::cout << "Contexto criado com sucesso!" << std::flush;
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "\r\033[2K" << std::flush;

    std::cout << "Criando exemplo..." << std::flush;

    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    // Simulate a message
    // Tokenize a prompt
    // std::string prompt = "You are a helpful assistant. Give a detailed answer to: Hi, what can you do?";
    std::string prompt = "Você é um assistente útil. Dê uma resposta detalhada para: Olá, o que você pode fazer?";
    std::vector<llama_token> tokens(prompt.size() * 4); // Allocate enough space for tokens
    int n_tokens = llama_tokenize(
        vocab, prompt.c_str(), prompt.size(),
        tokens.data(), tokens.size(), true, false);
    tokens.resize(n_tokens); // Resize to actual number of tokens

    // Prepare the batch
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());

    // Processa e gera tokens
    llama_sampler *sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    // llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::cout << std::endl;

    for (int i = 0; i < 200; i++)
    { // gera até 200 tokens
        if (llama_decode(ctx, batch) != 0)
            break;

        llama_token token = llama_sampler_sample(sampler, ctx, -1);

        if (llama_vocab_is_eog(vocab, token))
            break;

        char buf[256];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, false);
        if (n > 0)
            std::cout << std::string(buf, n) << std::flush;

        batch = llama_batch_get_one(&token, 1);
    }

    std::cout << std::endl;

    // Clean resources
    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}