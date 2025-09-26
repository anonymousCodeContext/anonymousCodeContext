import os
import logging
from argparse import ArgumentParser
import yaml
import re
import gen_similarity
import gen_process_elements
import make_prompt
import regex
import langChain_inference_async
import func_extractor
import sys
from na_utils.timer import Timer
import time

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default='config/template.yaml')
    return parser.parse_args()

def merge_dict(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(value, dict):
                merge_dict(dict1[key], value)
            elif value is not None:
                dict1[key] = value
        else:
            dict1[key] = value
    return dict1

def split_dict(data):
    determine = {}
    not_determine = {}
    
    for key, value in data.items():
        if isinstance(value, dict):
            d, not_d =split_dict(value)
            determine[key] = d
            if not_d is not None and len(not_d) > 0:
                not_determine[key] = not_d
        else:
            if isinstance(value, str) and "${" in value:
                not_determine[key] = value
            else:
                determine[key] = value
                
    return determine, not_determine

def find_top_level_vars(text):
    pattern = r'\$\{(?:[^{}]*|(?:\$\{(?:[^{}]|\{[^{}]*\})*\}))*\}'
    matches = regex.findall(pattern, text)
    
    outer_vars = [match[2:-1] for match in matches]
    return outer_vars 

def decode_placeholder_cycle(determine, value):
    init_value = value
    c = 0
    while True:
        value = decode_placeholder(determine, value)
        if value == init_value:
            break
        else:
            init_value = value
        c += 1
    return value

def decode_placeholder(determine, value):
    holder = find_top_level_vars(value)
    if isinstance(holder, list) and len(holder) == 1:
        holder = holder[0]
    
    if isinstance(holder, list):
        for h in holder:
            v = decode_placeholder(determine, '${'+h+'}')
            value = value.replace('${'+h+'}', v)
        return value
    
    init_holder = holder
    if '$' in holder:
        init_holder1 = holder
        holder = decode_placeholder(determine, holder)
        holder = str(holder)
        value = value.replace(init_holder, holder)
    elif '.' in holder:
        init_holder2 = holder
        d = determine
        flag = True
        for h in holder.split('.'):
            if h in d:
                d = d[h]
            else:
                flag = False
                break
        if flag:
            holder = d
            value = value.replace('${'+init_holder+'}', str(holder))

    elif holder in determine:
        v = determine[holder]
        value = value.replace('${'+holder+'}', v)
    return value
    
def query_replace_dynamic_syntax(determine, not_d):
    for key, value in not_d.items():
        if isinstance(value, dict):
            not_d[key] = query_replace_dynamic_syntax(determine, value)
        else: # str
            replacement = decode_placeholder_cycle(determine, value)
            if isinstance(replacement, int):
                replacement = int(replacement)
            elif isinstance(replacement, bool):
                replacement = bool(replacement)
            not_d[key] = replacement
    
    return not_d

def query_replace(determine, not_d):
    for key, value in not_d.items():
        if isinstance(value, dict):
            not_d[key] = query_replace(determine, value)
        else: # str
            vars_in_value = re.findall(r"\${(.*?)}", value)
            
            print(vars_in_value)
            for var in vars_in_value:
                if var in determine:
                    replacement = determine[var]
                    not_d[key] = not_d[key].replace('${'+var+'}', str(replacement))
                    
                elif '.' in var:
                    var_list = var.split('.')
                    d = determine
                    flag = True
                    for v in var_list:
                        if v in d:
                            d = d[v]
                        else:
                            flag = False
                            break
                    if flag:
                        replacement = d
                        not_d[key] = not_d[key].replace('${'+var+'}', str(replacement))
                    replacement = not_d[key]
                    if isinstance(replacement, int):
                        replacement = int(replacement)
                    elif isinstance(replacement, bool):
                        replacement = bool(replacement)
                    not_d[key] = replacement
                    
    return not_d
    
def load_config(config_path):
    with open(config_path) as f:
        data = yaml.safe_load(f)
    determine, not_determine = split_dict(data)
    N = 20
    while N > 0:
        if len(not_determine) == 0:
            break
        # not_determine = query_replace(determine, not_determine)
        not_determine = query_replace_dynamic_syntax(determine, not_determine)
        # print(not_determine)
        # print(not_determine)
        # break
        d, not_determine = split_dict(not_determine)
        determine = merge_dict(determine, d)
        N -= 1
    if N == 0:
        print(f"Warning: confing: {config_path} has not been fully resolved, some variables may be unresolved")
        print(not_determine)
    return determine


def ensure_path(path):
    """
    If the path is a directory and does not exist, create it.
    If the path is a file path, create the directory and an empty file.
    """
    if os.path.isdir(path) or (not os.path.splitext(path)[1] and not os.path.exists(path)):
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        if not os.path.exists(path):
            with open(path, 'w', encoding='utf-8') as f:
                pass

def get_time_str():
    """
    Returns the current timestamp in the format: YYYY-MM-DD HH:MM:SS
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def delete_file(path):
    if os.path.exists(path) is False:
        return
    import shutil
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)

def main():
    args = parse_args()
    config = load_config(args.config)
    
    EXEC_DATA_DIR = config['EXEC_DATA_DIR']
    EXEC_DATA_TASK_PATH = config['EXEC_DATA_TASK_PATH']
    MODEL = config['MODEL']
    TASK = config['TASK']
    MODA = config['MODA']
    Dataset = config['Dataset']
    dataset_name = Dataset['name']
    language = 'python'
    if language in Dataset:
        language = Dataset['language']

    ensure_path(EXEC_DATA_TASK_PATH)
    os.system(f"cp {args.config} {EXEC_DATA_TASK_PATH}/{Dataset['name']}-{TASK}-{MODEL}-{MODA}-config.yaml")
    
    ensure_path(config['RUN_LOG_PATH'])
    
    f = Tee(sys.__stdout__, open(config['RUN_LOG_PATH'], 'a'))
    sys.stdout = f
    sys.stderr = f
    
    print(f"\n\n===============Start running at {get_time_str()}===============\n")
    print(config)
    # exit()
    Exec = config['Exec']
    Preprocess = config['Preprocess']
    MakePrompt = config['MakePrompt']
    GenerateCode = config['GenerateCode']
    Eval = config['Eval']
    Stat = config['Stat']
    # Preprocess
    if Exec['preprocess'] > 0:
        print(f"\n====Preprocess start, at {get_time_str()}====")
        Preprocess = config['Preprocess']
        if Preprocess['type'] == 'baseline' or Preprocess['type'] == 'infilling':
            
            output_path = Preprocess['baseline']['output_path']
            if Preprocess['type'] == 'infilling':
                output_path = Preprocess['infilling']['output_path']
            
                
            if Exec['preprocess'] == 2:
                delete_file(output_path)
            
            if Exec['preprocess'] == 1 and os.path.exists(output_path):
                print(f"Preprocess file {output_path} already exists, skip preprocess")
            else:
                meta_path = Dataset['meta_data']
                ensure_path(output_path)
                source_code_path = Dataset['source_code']
                gen_process_elements.raw_gen_prompt_elements(meta_path, source_code_path, output_path)
                print(f"Preprocess done, file:{output_path}, at {get_time_str()}")
        elif Preprocess['type'] == 'similarity':
            # meta_path, output_path, number, method, output_dir
            output_dir = Preprocess['similarity']['output_dir']
            if Exec['preprocess'] == 2:
                delete_file(output_dir) 
            if Exec['preprocess'] == 1 and os.path.exists(output_dir):
                print(f"Preprocess dir {output_dir} already exists, skip preprocess")
            else:
                train_prompt_elem_path = Preprocess['similarity']['train_prompt_elem_path']
                test_meta_data_path = Preprocess['similarity']['test_meta_data_path']
                number = Preprocess['similarity']['number']
                method = Preprocess['similarity']['method']
                meta_path = Dataset['meta_data']
                ensure_path(output_dir)
                
                if method == 'bm25':
                    gen_similarity.bm25_indexing(train_prompt_elem_path, 
                                                test_meta_data_path, 
                                                output_dir, 
                                                number,
                                                key_name=Dataset['id_key'])
                elif method == 'cocosoda' or method == 'unixcoder':
                    train_embedding_path = Preprocess['similarity']['train_embedding_path']
                    test_embedding_path = Preprocess['similarity']['test_embedding_path']
                    re_embedding = Preprocess['similarity']['re_embedding']
                    if method == 'cocosoda':
                        func = gen_similarity.cocosoda_embeding_and_indexing
                    else:
                        func = gen_similarity.unixcoder_embeding_and_indexing
                    func(train_embedding_path,
                        test_embedding_path,
                        train_prompt_elem_path,
                        test_meta_data_path,
                        output_dir,
                        re_embedding)
                    # gen_similarity.cocosoda_embeding_and_indexing(train_embedding_path,
                    #                                               test_embedding_path,
                    #                                               train_prompt_elem_path,
                    #                                               test_meta_data_path,
                    #                                               output_dir,
                    #                                               re_embedding)
                
                
                # gen_similarity.preprocess(meta_path, output_path, number, method, output_dir)
                
                print(f"Preprocess done, dir:{output_dir}, at {get_time_str()}")
        elif Preprocess['type'] == 'navigation':
            output_path = Preprocess['navigation']['output_path']
            if Exec['preprocess'] == 2:
                print(f"Delete output_path: {output_path},")
                delete_file(output_path)
            ensure_path(output_path)
            
            from gen_navigation2 import get_prompt_ctx
            NA = Preprocess['navigation']
            prompt_elem_file = NA['prompt_elem_file']
            meta_data_path = NA['meta_data_path']
            run_base_dir = NA['run_base_dir']
            source_code_path = NA['source_code_path']
            model = MODEL
            max_iterations = NA['max_iterations']
            api_key= NA['api_key']
            open_key = NA['open_key']
            min_count = NA['min_count']
            sample_id_key = Dataset['id_key']
            output_path = NA['output_path']
            plus = False
            if 'plus' in NA:
                plus = NA['plus']
            
            max_concurrent_num = 20
            if 'max_concurrent_num' in NA:
                max_concurrent_num = NA['max_concurrent_num']
            
            # re_run = NA['re_run']
            
            get_prompt_ctx(prompt_elem_file=prompt_elem_file,
                            meta_data_path=meta_data_path,
                            run_base_dir=run_base_dir,
                            source_code_path=source_code_path,
                            model=model,
                            max_iterations=max_iterations,
                            api_key=api_key,
                            open_key=open_key,
                            min_count=min_count,
                            sample_id_key=sample_id_key,
                            output_path=output_path,
                            dataset_name=dataset_name,
                            plus=plus,
                            max_concurrent_num=max_concurrent_num)
                
            print(f"Preprocess done, file:{output_path}, at {get_time_str()}")
    if Exec['make_prompt'] > 0:
        timer = Timer()
        timer.start()
        print(f"\n====MakePrompt start, at {get_time_str()}====")
        print(MakePrompt['type'])
        MakePrompt = config['MakePrompt']
        if MakePrompt['type'] == 'baseline' or MakePrompt['type'] == 'infilling':
            #             prompt_element_file = args.prompt_element_file
            # setting = args.setting
            # output_file = args.output_file
            # context_window = args.context_window
            # max_tokens = args.max_tokens
            output_path = MakePrompt['output_path']
            if Exec['make_prompt'] == 2:
                delete_file(output_path) 
            if Exec['make_prompt'] == 1 and os.path.exists(output_path):
                print(f"MakePrompt file {output_path} already exists, skip preprocess")
            else:
                ensure_path(output_path)
                prompt_element_file = MakePrompt['preprocess_file']
                setting = MakePrompt['type']
                context_window = MakePrompt['context_window']
                max_tokens = MakePrompt['max_token_length']
                
                make_prompt.make_prompt(prompt_element_file=prompt_element_file, 
                                        setting=setting, 
                                        output_file=output_path, 
                                        context_window=context_window, 
                                        max_tokens=max_tokens,
                                        dataset=Dataset['name'],
                                        language=language)
                print(f"MakePrompt done, file:{output_path}, at {get_time_str()}")
        elif MakePrompt['type'] == 'similarity':
            # method, max_token_length, number, preprocess_file, output_path
            output_path = MakePrompt['output_path']
            if Exec['make_prompt'] == 2:
                delete_file(output_path) 
            if Exec['make_prompt'] == 1 and os.path.exists(output_path):
                print(f"MakePrompt file {output_path} already exists, skip preprocess")
            else:
                ensure_path(output_path)
                method = MakePrompt['type']
                max_token_length = MakePrompt['max_token_length']
                number = MakePrompt['N']
                preprocess_file = MakePrompt['preprocess_file']
                template_path = MakePrompt['template_path']
                template_item_path = MakePrompt['template_item_path']
                input_data_path = MakePrompt['input_data_path']
                elem_indexing_path = MakePrompt['elem_indexing_path']
                gen_similarity.make_prompt2(template_path, 
                                            template_item_path,
                                            input_data_path,
                                            elem_indexing_path,
                                            number,
                                            output_path,
                                            language=language)
                # gen_similarity.make_prompt(method, max_token_length, number, preprocess_file, output_path)
                print(f"MakePrompt done, file:{output_path}, at {get_time_str()}")
        elif MakePrompt['type'] == 'sa':
            output_path = MakePrompt['output_path']
            if Exec['make_prompt'] == 2:
                delete_file(output_path) 
            if Exec['make_prompt'] == 1 and os.path.exists(output_path):
                print(f"MakePrompt file {output_path} already exists, skip preprocess")
            else:
                # print('..................')
                import gen_static
                # template_path='/root/workspace/code/anonymousCodeContext/prompt/template/sa_template_refine.txt', 
                # element_path='/root/workspace/code/DevEval/prompt_elements_source_code2.jsonl', 
                # meta_data_path='/root/workspace/code/anonymousCodeContext/data/DevEval/data_fixed2_sample_pre_proj.jsonl',
                # depandancy_dir='/root/workspace/code/anonymousCodeContext/data/DevEval/similarity/sa',
                # output_path='/root/workspace/code/anonymousCodeContext/data/DevEval/prompt_sa.jsonl'
                template_path = MakePrompt['template_path']
                element_path = MakePrompt['element_path']
                meta_data_path = MakePrompt['meta_data_path']
                depandancy_dir = MakePrompt['depandancy_dir']
                output_path = MakePrompt['output_path']
                gen_static.make_prompt(
                    template_path=template_path,
                    element_path=element_path,
                    meta_data_path=meta_data_path,
                    depandancy_dir=depandancy_dir,
                    output_path=output_path)
        elif MakePrompt['type'] == 'navigation':
            import gen_navigation2
            prompt_ctx_file = MakePrompt['preprocess_file']
            elem_data_path = MakePrompt['elem_data_path']
            meta_data_path = MakePrompt['meta_data_path']
            output_path = MakePrompt['output_path']
            key=Dataset['id_key']
            template_path = MakePrompt['template_path']
            gen_navigation2.make_prompt(prompt_ctx_file=prompt_ctx_file,
                                        template_path=template_path,
                                        elem_data_path=elem_data_path,
                                        meta_data_path=meta_data_path,
                                        output_file=output_path,
                                        key=key)
            # gen_navigation2.get_prompt_ctx()
        else:
            
            print(f"MakePrompt type {MakePrompt['type']} not supported")
            exit()
        timer.end()
        print(f"MakePrompt Time cost: {timer.get_elapsed_ms()}ms, file:{output_path}")
        
    if Exec['generate_code'] > 0:
        print(f"\n====GenerateCode start, at {get_time_str()}====")
        GenerateCode = config['GenerateCode']
        output_dir = GenerateCode['output_dir']
        if Exec['generate_code'] == 2:
            # time.sleep(10)
            print(f"Delete output_dir: {output_dir},")
            delete_file(output_dir)
        ensure_path(output_dir)
        
        prompt_file = GenerateCode['prompt_file']
        api_key_file = GenerateCode['api_key_file']
        max_concurrent = GenerateCode['max_concurrent']
        min_count = GenerateCode['min_count']
        gen_type = GenerateCode['type']
        meta_data_path = Dataset['meta_data']
        open_key = None
        if 'open_key' in GenerateCode:
            open_key = GenerateCode['open_key']
        if os.path.exists(api_key_file):
            with open(api_key_file, 'r', encoding='utf-8') as f:
                api_key_content = f.read()
            print(f"Token usage will be tracked based on this key. Content of api_key_file:\n{api_key_content}")
        else:
            print(f"api_key_file not found: {api_key_file}")
        TC = 5
        tolerance = TC
        fail_count = 1
        while fail_count > 0 and tolerance > 0:
            fail_count = langChain_inference_async.run(prompt_file, 
                                    output_dir, 
                                    MODEL, 
                                    MODA, 
                                    api_key_file,
                                    open_key, 
                                    max_concurrent, 
                                    Dataset['id_key'], 
                                    min_count,
                                    task=gen_type,
                                    meta_data_path=meta_data_path)
            tolerance -= 1
            if fail_count > 0:
                st = min(2 * ((TC - tolerance) ** 2), 600)
                print(f'sleep {st}s, then try again')
                time.sleep(st)
        print(f"GenerateCode done, file:{output_dir}, at {get_time_str()}")
      
    if Exec['postprocess'] > 0:
        print(f"\n====Postprocess start, at {get_time_str()}====")
        Postprocess = config['Postprocess']
        output_file = Postprocess['output_file']
        if Exec['postprocess'] == 2:
            print(f"Delete output_dir: {output_file},")
            delete_file(output_file) 
        # if Exec['postprocess'] == 1 and os.path.exists(output_file):
        #     print(f"Postprocess file {output_file} already exists, skip preprocess")
        # else:
        ensure_path(output_file)
        GenerateCode = config['GenerateCode']
        llm_output_file = GenerateCode['output_dir'] + '/latest.jsonl'
        if Dataset['name'] == 'CoderEval':
            func_extractor.postprocess_coderEval2(llm_output_file, output_file, language=language)
        elif Dataset['name'] == 'DevEval':
            func_extractor.postprocess_deveval(llm_output_file, output_file)
            
        print(f"Postprocess done, file:{output_file}, at {get_time_str()}")
    
    if Exec['eval'] > 0:
        print(f"\n====Eval start, at {get_time_str()}====")
        
        if Dataset['name'] == 'DevEval':
            source_code_root = Eval['source_code_root']
            from data.DevEval import check_source_code
            check_source_code.reset_source_code(source_code_root)
            os.environ['NO_SQLITE'] = '1'
            Eval = config['Eval']
            log_file = Eval['log_file']
            if Exec['eval'] == 2:
                delete_file(log_file)
                print(f"Delete log_file: {log_file}")
            
            ensure_path(log_file)
            output_file = Eval['output_file']
            source_code_root = Eval['source_code_root']
            data_file = Eval['data_file']
            n = Eval['n']
            k = Eval['k']
            class Args:
                pass
            args = Args()
            args.output_file = output_file
            args.log_file = log_file
            args.source_code_root = source_code_root
            args.data_file = data_file
            args.n = n
            args.k = k
            import eval_pass_k
            eval_pass_k.main(args)
            print(f"Eval done, file:{log_file}, at {get_time_str()}")
    
    if Exec['efficiency_stat'] > 0:
        import efficiency_stat
        print(f"\n====EfficiencyStat start, at {get_time_str()}====")
        llm_output_file = Stat['llm_output_file']
        efficiency_stat.stat_efficiency(MODEL, llm_output_file)
        print(f"EfficiencyStat done, at {get_time_str()}")
        
if __name__ == "__main__":
    main()
