import json
from utils import load_json_data
from argparse import ArgumentParser
import tiktoken


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--prompt_element_file", type=str, default='prompt/prompt_elements.jsonl')
    parser.add_argument("--setting", type=str, choices=['baseline', 'local_completion', 'local_infilling'])
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--context_window", type=int, default=16384)
    parser.add_argument("--max_tokens", type=int, default=500)
    return parser.parse_args()


def produce_prompt(context_window, max_tokens, setting, d, tokenizer, dataset='DevEval', language='python'):
    
    # print('input_code:', d['input_code'])
    # print('d:',d)
    input_ids = tokenizer.encode(d['input_code'])
    max_context_length = context_window - len(input_ids) - max_tokens
    template = open(f'prompt/template/{setting}/ChatLM_na.txt', 'r').read()
    if dataset == 'CoderEval':
        template = open(f'prompt/CoderEval/ChatLM_na.txt', 'r').read()
    if setting == 'baseline':
        if d['class_name']:
            input_code = f"class {d['class_name']}:\n" + d['input_code']
        else:
            input_code = d['input_code']
        prompt = template.format(
            function_name=d['function_name'],
            input_code=input_code
        )
    elif setting == 'local_completion':
        context_above_ids = tokenizer.encode(d['contexts_above'])
        if len(context_above_ids) > max_context_length:
            context_above_ids = context_above_ids[-max_context_length:]
            context_above = tokenizer.decode(context_above_ids)
        prompt = template.format(
            function_name=d['function_name'],
            contexts_above=context_above,
            input_code=d['input_code']
        )
    elif setting == 'local_infilling' or setting == 'infilling':
        prompt = template.format(
            function_name=d['function_name'],
            contexts_above=d['contexts_above'],
            contexts_below=d['contexts_below'],
            input_code=d['input_code'],
            language=language
        )
        prompt_ids = tokenizer.encode(prompt)
        if len(prompt_ids) > context_window:
            context_above_ids = tokenizer.encode(d['contexts_above'])
            context_below_ids = tokenizer.encode(d['contexts_below'])
            # Truncate context to fit within context window
            context_above_ratio = len(context_above_ids) / (len(context_above_ids) + len(context_below_ids))
            context_below_ratio = len(context_below_ids) / (len(context_above_ids) + len(context_below_ids))
            max_context_above_length = int(max_context_length * context_above_ratio)
            max_context_below_length = int(max_context_length * context_below_ratio)
            context_above_ids = context_above_ids[-max_context_above_length:]
            context_below_ids = context_below_ids[:max_context_below_length]
            context_above = tokenizer.decode(context_above_ids)
            context_below = tokenizer.decode(context_below_ids)
            prompt = template.format(
                function_name=d['function_name'],
                contexts_above=context_above,
                contexts_below=context_below,
                input_code=d['input_code']
            )
    return prompt

 
def make_prompt(prompt_element_file, setting, output_file, context_window, max_tokens, dataset='DevEval', language='python'):
    prompt_elements = load_json_data(prompt_element_file)
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    with open(output_file, 'w') as f:
        # print('prompt_elements:', prompt_elements)
        for d in prompt_elements:
            # print(d)
            prompt = produce_prompt(d=d, 
                                    setting=setting, 
                                    context_window=context_window, 
                                    max_tokens=max_tokens, 
                                    tokenizer=tokenizer,
                                    dataset=dataset,
                                    language=language)
            d['prompt'] = prompt
            f.write(json.dumps(d) + '\n')
    print(f"MakePrompt done, save file:{output_file},")
    return True
 
def main():
    args = parse_args()
    prompt_element_file = args.prompt_element_file
    setting = args.setting
    output_file = args.output_file
    context_window = args.context_window
    max_tokens = args.max_tokens
    
    make_prompt(prompt_element_file, setting, output_file, context_window, max_tokens)
    

if __name__ == '__main__':
    main()
