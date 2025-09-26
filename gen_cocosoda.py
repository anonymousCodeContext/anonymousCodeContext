import os
import jsonlines
import json
import heapq
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, RobertaModel
import tiktoken


class CoCoSoDaEncoder(nn.Module):   
    def __init__(self, encoder):
        super(CoCoSoDaEncoder, self).__init__()
        self.encoder = encoder
      
    def forward(self, input_ids=None):
        attention_mask = input_ids.ne(self.encoder.config.pad_token_id)
        outputs = self.encoder(input_ids, attention_mask=attention_mask)[0]
        outputs = (outputs * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        return F.normalize(outputs, p=2, dim=1)

def load_deveval_data(file_path):
    """
    Loads and preprocesses data from the DevEval JSONL file into the required format.
    """
    data = []
    print(f"Loading data from {file_path}...")
    with jsonlines.open(file_path) as f:
        for obj in f:
            docstring_tokens = f"{obj['requirement']['Functionality']}\n{obj['requirement']['Arguments']}"
            signature_requirement = f"Signature: {obj['signature']}\nRequirement:\n    {docstring_tokens}"
    
            # Your logic to read the ground truth code
            ground_truth, indent_space = read_code(obj) 
            
            # Extract function name
            func_signature = obj['signature']
            function_name = func_signature[func_signature.find('def ') + 4 : func_signature.find('(')]
            
            data.append({
                'signature_requirement': signature_requirement, 
                'namespace': obj['namespace'], 
                'ground_truth': ground_truth,
                'function_name': function_name,
                'indent_space': indent_space,
            })
    print(f"Loaded {len(data)} examples.")
    return data

def load_prompt_elements(file_path='./data/DevEval/prompt_elements_source_code2.jsonl'):
    import utils
    return utils.load_json_data(file_path)

def get_embeddings(data_list, model, tokenizer, device, batch_size=32, key_name='input_code'):
    """
    Generates embeddings for a list of data objects using the CoCoSoDa model.
    """
    model.eval()
    all_embeddings = []
    
    # Prepare texts for embedding
    texts = [d[key_name] for d in data_list]

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            # --- CRITICAL: Apply CoCoSoDa's specific input format ---
            tokenized_inputs = []
            for text in batch_texts:
                # Truncate to a max length to prevent errors, 256 is a safe default
                code_tokens = tokenizer.tokenize(text)[:256-4] 
                tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
                tokenized_inputs.append(tokenizer.convert_tokens_to_ids(tokens))

            # Pad the batch to the longest sequence
            max_len = max(len(inp) for inp in tokenized_inputs)
            padded_inputs = [inp + [tokenizer.pad_token_id] * (max_len - len(inp)) for inp in tokenized_inputs]
            
            input_ids = torch.tensor(padded_inputs).to(device)
            
            # Get embeddings from our custom pooling model
            batch_embeddings = model(input_ids=input_ids)
            all_embeddings.extend(batch_embeddings.cpu().numpy())
            
    return np.array(all_embeddings)

def run_cocosoda_retrieval(corpus_data, query_data, model_path, output_dir, k):
    """
    Orchestrates the entire retrieval process: loading model, generating embeddings,
    and finding the top-k most similar items for each query.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoder = RobertaModel.from_pretrained(model_path)
    # Ensure the encoder knows its pad token id for the custom model
    encoder.config.pad_token_id = tokenizer.pad_token_id
    model = CoCoSoDaEncoder(encoder).to(device)

    # Generate embeddings for the entire corpus (the "database" to search in)
    corpus_embeddings = get_embeddings(corpus_data, model, tokenizer, device)
    
    # Generate embeddings for the queries (the items we need candidates for)
    query_embeddings = get_embeddings(query_data, model, tokenizer, device)

    # Perform retrieval
    processed_data = []
    print("Performing similarity search...")
    for i in tqdm(range(len(query_embeddings)), desc="Retrieving Candidates"):
        query_emb = torch.tensor(query_embeddings[i]).unsqueeze(0)
        corpus_embs_tensor = torch.tensor(corpus_embeddings)
        
        cos_sim = F.cosine_similarity(query_emb, corpus_embs_tensor).numpy()
        
        # To avoid retrieving the item itself, we can find its index and set similarity to -inf
        query_namespace = query_data[i]['namespace']
        try:
            self_index = [idx for idx, d in enumerate(corpus_data) if d['namespace'] == query_namespace][0]
            cos_sim[self_index] = -np.inf
        except IndexError:
            pass # Query not in corpus, no need to exclude

        # Get top-k indices using heapq for efficiency
        top_k_indices = heapq.nlargest(k, range(len(cos_sim)), cos_sim.take)
        
        # Assemble candidates
        code_candidates = []
        for rank, corpus_id in enumerate(top_k_indices):
            data = corpus_data[corpus_id]
            code_candidates.append({
                'signature_requirement': data['signature_requirement'],
                'score': float(cos_sim[corpus_id]),
                'function_name': data['function_name'],
                'ground_truth': data['ground_truth'],
                'namespace': data['namespace']
            })
        
        obj = query_data[i]
        processed_data.append({
            'signature_requirement': obj['signature_requirement'],
            'code_candidates': code_candidates,
            'ground_truth': obj['ground_truth'],
            'namespace': obj['namespace'],
            'function_name': obj['function_name'],
            'indent_space': obj['indent_space']
        })

    # Save the retrieval results
    output_file = os.path.join(output_dir, f'retrieval_results_cocosoda_{k}.jsonl')
    with jsonlines.open(output_file, 'w') as f:
        f.write_all(processed_data)
    print(f"Retrieval results saved to {output_file}")
    return output_file

def create_prompts(retrieval_file, output_path, max_token_length, k):
    """
    Generates the final LLM prompts from the retrieval results file.
    This function is adapted from your gen_similarity.py.
    """
    with jsonlines.open(retrieval_file) as f:
        retrieved_data = list(f)

    tokenizer = tiktoken.encoding_for_model("gpt-4")
    prompts = []
    print(f"Creating prompts from {retrieval_file}...")
    for obj in tqdm(retrieved_data, desc="Creating Prompts"):
        instruction_template = 'Please complete the {function_name} function based on the provided Signature and Requirement \n'
        input_instruction = ' [INPUT]\n'
        output_instruction = ' [OUTPUT]\n'

        final_prompt_part = instruction_template.format(function_name=obj['function_name']) + \
                            input_instruction + obj['signature_requirement'] + '\n' + output_instruction
        
        remaining_token_budget = max_token_length - len(tokenizer.encode(final_prompt_part))
        
        prompt_prefix = ""
        examples_included = 0
        for sample in obj['code_candidates']:
            sample_prompt = instruction_template.format(function_name=sample['function_name']) + \
                            input_instruction + sample['signature_requirement'] + '\n' + \
                            output_instruction + sample['ground_truth'] + '\n\n\n'
            
            sample_token_len = len(tokenizer.encode(sample_prompt))
            if remaining_token_budget >= sample_token_len:
                prompt_prefix += sample_prompt
                remaining_token_budget -= sample_token_len
                examples_included += 1
            else:
                print(f"Token limit reached for {obj['namespace']}. Included {examples_included}/{k} examples.")
                break
        
        full_prompt = prompt_prefix + final_prompt_part
        prompts.append({
            'prompt': full_prompt,
            'function_name': obj['function_name'],
            'label': obj['ground_truth'],
            'namespace': obj['namespace'],
            'count': examples_included,
            'indent_space': obj['indent_space']
        })

    with jsonlines.open(output_path, 'w') as f:
        f.write_all(prompts)
    print(f"Final prompts saved to {output_path}")

def read_code(json_obj):
    """
    Placeholder for your function that reads source code files.
    """
    #TODO: Implement read_code

    return "pass # TODO: Implement read_code", "    "

def save_embeddings(data, 
                    output_path='data/DevEval/similarity/cocosoda_embeddings_prompt_elements_source_code2.npy'):
    import utils
    MODEL_PATH = "/root/workspace/data/DeepSoftwareAnalytics/CoCoSoDa"
    
    TOP_K = 8
    MAX_PROMPT_TOKENS = 8000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    encoder = RobertaModel.from_pretrained(MODEL_PATH)
    # Ensure the encoder knows its pad token id for the custom model
    encoder.config.pad_token_id = tokenizer.pad_token_id
    model = CoCoSoDaEncoder(encoder).to(device)
    result = get_embeddings(data, 
                            model, 
                            tokenizer, 
                            device, 
                            batch_size=4,
                            key_name='input_code')

    # print(result.shape)
    import numpy as np
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    np.save(output_path, result)
    print(f"Embeddings have been saved to {output_path}")
    

if __name__ == '__main__':
    
    MODEL_PATH = "/root/workspace/data/DeepSoftwareAnalytics/CoCoSoDa"
    DATA_PATH = '/root/workspace/code/DevEval/data_fixed2.jsonl'
    OUTPUT_DIR = '.data/DevEval/similarity/cocosoda_output'
    TOP_K = 8
    MAX_PROMPT_TOKENS = 8000

    os.makedirs(OUTPUT_DIR, exist_ok=True)


    all_data = load_deveval_data(DATA_PATH)
    corpus_data = all_data
    query_data = all_data
    
    
    # 2. Run the CoCoSoDa retrieval process
    retrieval_results_file = run_cocosoda_retrieval(
        corpus_data=corpus_data,
        query_data=query_data,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        k=TOP_K
    )

    final_prompts_file = os.path.join(OUTPUT_DIR, f'prompts_cocosoda_{TOP_K}.jsonl')
    create_prompts(
        retrieval_file=retrieval_results_file,
        output_path=final_prompts_file,
        max_token_length=MAX_PROMPT_TOKENS,
        k=TOP_K
    )

    print("\nCoCoSoDa processing complete!")