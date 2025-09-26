import json

model_dict = {
    'gpt': ['gpt-4o-2024-11-20'],
    'gpt-4': ['gpt-4o-2024-11-20'],
    'gemini': ['gemini-2.5-pro'],
    'claude': ['claude-sonnet-4-20250514', 'claude-sonnet-4-20250514-thinking'],
    'deepseek': ['deepseek-ai/DeepSeek-R1', 'deepseek-r1-250528', 'deepseek-ai/DeepSeek-R1-0528'],
    'gpt-oss': ['openai/gpt-oss-120b'],
    'qwen3_coder': ['Qwen/Qwen3-Coder-480B-A35B-Instruct'],
    'llama': ['meta-llama/Llama-3.3-70B-Instruct'],
    'qwen3': ['Qwen/Qwen3-235B-A22B-Instruct-2507'],
}




def stat_efficiency(model_key, llm_output_file):
    tokens_in = []
    tokens_out = []
    times = []
    # model_full_name = model_dict[model_name]
    with open(llm_output_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            # print(js['usage'])
            # exit()
            # name = model_full_name
            # if model_name == 'claude' and 'claude-sonnet-4-20250514-thinking' in js['usage']:
            #     name = 'claude-sonnet-4-20250514-thinking'
            
            # if name not in js['usage']:
            #     print(js['usage'])
            #     exit()
                
            name_set = model_dict[model_key]
            model_name = None
            for name in name_set:
                if name in js['usage']:
                    model_name = name
                    break
            if model_name is None:
                # print(f'{model_name} not found !!!!!!!!!!!')
                print(js['usage'])
                raise Exception(f'model_name not found !!!!!!!!!!!')
                
            usgae = js['usage'][model_name]
            # [model_dict[model_name]]
            tokens_in.append(usgae['input_tokens'])
            tokens_out.append(usgae['output_tokens'])
            times.append(js['time_cost'])
            
    print(f'{model_key} tokens_in: {sum(tokens_in)/len(tokens_in)}')
    print(f'{model_key} tokens_out: {sum(tokens_out)/len(tokens_out)}')
    print(f'{model_key} tokens_total: {sum(tokens_in)/len(tokens_in) + sum(tokens_out)/len(tokens_out)}')
    print(f'{model_key} times: {sum(times)/len(times)}')
    return sum(tokens_in)/len(tokens_in), sum(tokens_out)/len(tokens_out), sum(times)/len(times)
    
    




