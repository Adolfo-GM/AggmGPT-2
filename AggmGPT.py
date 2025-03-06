from llama_cpp import Llama

def RunAggmGPT():
    model_path = "AggmGPT.gguf"  
    model = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=35)
    while True:
        prompt = input("\nEnter your prompt: ")

        messages = [
            {"role": "system", "content": f"You are AggmGPT, an advanced AI assistant created by Adolfo GM. The user said: '{prompt}'."},
            {"role": "user", "content": prompt}
        ]

        output = model.create_chat_completion(messages, max_tokens=1050, temperature=0.7)

        print("\nAggmGPT:")
        print(output["choices"][0]["message"]["content"])

def AskAggmGPT(question, model_path="AggmGPT.gguf"):
    model = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=35)
    messages = [
        {"role": "system", "content": f"You are AggmGPT, an advanced AI assistant created by Adolfo GM.  The user asked: '{question}'."},
        {"role": "user", "content": question}
    ]

    output = model.create_chat_completion(messages, max_tokens=1050, temperature=0.7)

    return output["choices"][0]["message"]["content"]

if __name__ == "__main__":
    print(AskAggmGPT("What is the meaning of life?"))
    RunAggmGPT()
