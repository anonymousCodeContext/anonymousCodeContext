import jsonlines
import os
from tqdm import tqdm
from gensim.summarization import bm25
import random
# from openai import OpenAI
# from langchain_openai import OpenAIEmbeddings
import numpy as np
import json
# import ast
import tiktoken
from na_utils.timer import Timer
import utils
import copy


# def gen_signature_requirement(train):
@DeprecationWarning
def bm25_preprocess(train, test, output_path, number):
    timer = Timer()
    timer.start()
    code = [' '.join(obj['signature_requirement']) for obj in train]
    bm25_model = bm25.BM25(code)
    average_idf = sum(map(lambda k: float(bm25_model.idf[k]), bm25_model.idf.keys())) / len(bm25_model.idf.keys())
    
    processed = []
    for obj in tqdm(test, total=len(test)):
        query = obj['signature_requirement']
        score = bm25_model.get_scores(query,average_idf) # one-dimensional array
        rtn = sorted(enumerate(score), key=lambda x: x[1], reverse=True)[:number + 1]
        code_candidates = []
        for i in range(len(rtn)):
            data = train[rtn[i][0]]
            if data['namespace'] == obj['namespace']:
                continue
            code_candidates.append({'signature_requirement': data['signature_requirement'], 
                                           'score': rtn[i][1], 
                                        #    'idx':i+1,
                                           'function_name': data['function_name'],
                                           'ground_truth': data['ground_truth'],
                                           'namespace': data['namespace']
                                           })            
        processed.append({'signature_requirement': obj['signature_requirement'], 
                          'code_candidates': code_candidates,
                          'ground_truth': obj['ground_truth'],
                          'namespace': obj['namespace'],
                          'function_name': obj['function_name'],
                          'indent_space': obj['indent_space']
                          })
    timer.end()
    print(f"BM25 Time cost: {timer.get_elapsed_ms()}ms")
    with jsonlines.open(os.path.join(output_path, 'test_bm25_'+str(number)+'.jsonl'),'w') as f:
        f.write_all(processed)


        
def bm25_indexing(train_prompt_elem_path, 
                  test_meta_data_path, 
                  output_dir,
                  number=5,
                  key_name='namespace'):
    timer = Timer()
    timer.start()
    train = utils.load_json_data(train_prompt_elem_path)
    test_meta_data_dict = utils.load_json_data_as_dict(test_meta_data_path)
    test = [d for d in train if d[key_name] in test_meta_data_dict]
    
    corpus = [ data['input_code'].split()  for data in train]
    bm25_model = bm25.BM25(corpus)
    average_idf = sum(float(val) for val in bm25_model.idf.values()) / len(bm25_model.idf)
    
    processed = []
    for obj in tqdm(test, total=len(test)):
        query = obj['input_code'].split()
        scores = bm25_model.get_scores(query, average_idf) # one-dimensional array
        scores_arr = np.argsort(scores)[::-1]
        can_count = 0
        for i in range(len(scores_arr)):
            data = copy.deepcopy(train[scores_arr[i]])
            path1 = data['completion_path']
            path2 = obj['completion_path']
            if data[key_name] == obj[key_name] or path1 == path2:
                continue
            data['score'] = scores[scores_arr[i]]
            code_candidates.append(data)
            can_count += 1
            if can_count >= number:
                break

        e = copy.deepcopy(obj)
        e['candidates'] = code_candidates                
        processed.append(e)
        
    timer.end()
    print(f"BM25 Time cost: {timer.get_elapsed_ms()}ms")
    with jsonlines.open(os.path.join(output_dir, 'indexing_bm25_'+str(number)+'.jsonl'),'w') as f:
        f.write_all(processed)
        
def embedding_indexing(train_embedding_path, 
                       test_embedding_path,
                       train_prompt_elem_path, 
                       test_meta_data_path, 
                        output_dir,
                        number=5,
                        key_name='namespace',
                        method='cocosoda'):
    import torch
    import torch.nn.functional as F
    timer = Timer()
    timer.start()
    print(f'train_embedding_path: {train_embedding_path}')
    print(f'test_embedding_path: {test_embedding_path}')
    print(f'train_prompt_elem_path: {train_prompt_elem_path}')
    print(f'method: {method}')
    corpus_embs = np.load(train_embedding_path)
    query_emb = np.load(test_embedding_path)
    
    corpus_embs = torch.tensor(corpus_embs)
    query_embs = torch.tensor(query_emb)
    print(corpus_embs.shape, query_emb.shape)
    
    train = utils.load_json_data(train_prompt_elem_path)
    test_meta_data_dict = utils.load_json_data_as_dict(test_meta_data_path)
    test = [d for d in train if d[key_name] in test_meta_data_dict]
    
    processed = []
    index = 0
    for obj in tqdm(test, total=len(test)):
        q = query_embs[index].unsqueeze(0)
        scores = F.cosine_similarity(q, corpus_embs, dim=1).numpy()
        scores_arr = np.argsort(scores)[::-1]
        code_candidates = []
        can_count = 0
        for i in range(len(scores_arr)):
            # data = train[rtn[i][0]]
            data = copy.deepcopy(train[scores_arr[i]])
            path1 = data['completion_path']
            path2 = obj['completion_path']
            if data[key_name] == obj[key_name] or path1 == path2:
                continue
            data['score'] = float(scores[scores_arr[i]])
            code_candidates.append(data)
            can_count += 1
            if can_count >= number:
                break

        e = copy.deepcopy(obj)
        e['candidates'] = code_candidates                
        processed.append(e)
        index += 1
        
    timer.end()
    print(f"{method} indexing Time cost: {timer.get_elapsed_ms()}ms")
    with jsonlines.open(os.path.join(output_dir, f'indexing_{method}_'+str(number)+'.jsonl'),'w') as f:
        f.write_all(processed)

def get_cocosoda_embeddings(train_embedding_output_path='data/DevEval/similarity/cocosoda_embeddings_prompt_elements_source_code2.npy',
                            test_embedding_output_path='data/DevEval/similarity/cocosoda_embeddings_prompt_elements_source_code2_sample_per_proj.npy',
                            train_data_path='data/DevEval/prompt_elements_source_code2.jsonl',
                            test_meta_data_path='data/DevEval/data_fixed2_sample_per_proj.jsonl'):
    '''Save the CoCoSoDa embeddings for train and test sets locally'''
    import gen_cocosoda
    print(f'train_embedding_output_path: {train_embedding_output_path}')
    print(f'test_embedding_output_path: {test_embedding_output_path}')
    print(f'train_data_path: {train_data_path}')
    print(f'test_meta_data_path: {test_meta_data_path}')
    train_data = utils.load_json_data(train_data_path)
    gen_cocosoda.save_embeddings(data=train_data, 
                                 output_path=train_embedding_output_path)
    test_dict = utils.load_json_data_as_dict(test_meta_data_path)
    test_data = [d for d in train_data if d['namespace'] in test_dict]
    gen_cocosoda.save_embeddings(data=test_data, 
                                 output_path=test_embedding_output_path)

    
    # exit()

def cocosoda_embeding_and_indexing(train_embedding_path='data/DevEval/similarity/cocosoda_embeddings_prompt_elements_source_code2.npy',
                                   test_embedding_path='data/DevEval/similarity/cocosoda_embeddings_prompt_elements_source_code2_sample_per_proj.npy',
                                   train_prompt_elem_path='./data/DevEval/prompt_elements_source_code2.jsonl',
                                   test_meta_data_path='./data/DevEval/data_fixed2_sample_pre_proj.jsonl',
                                   output_dir='./data/DevEval/similarity',
                                   re_embedding=False):
    timer = Timer()
    timer.start()
    if re_embedding:
        get_cocosoda_embeddings(train_embedding_path,
                                test_embedding_path,
                                train_prompt_elem_path,
                                test_meta_data_path)
    
    
    embedding_indexing(train_embedding_path, 
                       test_embedding_path,
                       train_prompt_elem_path, 
                       test_meta_data_path, 
                       output_dir,
                       number=5,
                       key_name='namespace')
    timer.end()
    print(f"CoCoSoDa embedding and indexing Time cost: {timer.get_elapsed_ms()}ms")

def get_unixcoder_embeddings(train_embedding_output_path='data/DevEval/similarity/unixcoder_embeddings_prompt_elements_source_code2.npy',
                            test_embedding_output_path='data/DevEval/similarity/unixcoder_embeddings_prompt_elements_source_code2_sample_per_proj.npy',
                            train_data_path='data/DevEval/prompt_elements_source_code2.jsonl',
                            test_meta_data_path='data/DevEval/data_fixed2_sample_pre_proj.jsonl'):
    '''Save the unixcoder embeddings for train and test sets locally'''
    import gen_unixcoder
    print(f'train_embedding_output_path: {train_embedding_output_path}')
    print(f'test_embedding_output_path: {test_embedding_output_path}')
    print(f'train_data_path: {train_data_path}')
    print(f'test_meta_data_path: {test_meta_data_path}')
    train_data = utils.load_json_data(train_data_path)
    gen_unixcoder.save_embeddings(data=train_data, 
                                 output_path=train_embedding_output_path)
    test_dict = utils.load_json_data_as_dict(test_meta_data_path)
    test_data = [d for d in train_data if d['namespace'] in test_dict]
    gen_unixcoder.save_embeddings(data=test_data, 
                                 output_path=test_embedding_output_path)

def unixcoder_embeding_and_indexing(train_embedding_path='data/DevEval/similarity/unixcoder_embeddings_prompt_elements_source_code2.npy',
                                   test_embedding_path='data/DevEval/similarity/unixcoder_embeddings_prompt_elements_source_code2_sample_per_proj.npy',
                                   train_prompt_elem_path='./data/DevEval/prompt_elements_source_code2.jsonl',
                                   test_meta_data_path='./data/DevEval/data_fixed2_sample_pre_proj.jsonl',
                                   output_dir='./data/DevEval/similarity',
                                   re_embedding=False):
    timer = Timer()
    timer.start()
    if re_embedding:
        get_unixcoder_embeddings(train_embedding_path,
                                test_embedding_path,
                                train_prompt_elem_path,
                                test_meta_data_path)
    
    
    embedding_indexing(train_embedding_path, 
                       test_embedding_path,
                       train_prompt_elem_path, 
                       test_meta_data_path, 
                       output_dir,
                       number=5,
                       key_name='namespace',
                       method='unixcoder')
    timer.end()
    print(f"UniXCoder embedding and indexing Time cost: {timer.get_elapsed_ms()}ms")
           
def embedding_preprocess_cache(train, output_path):
    cache_file = os.path.join(output_path, 'embedding_cache.jsonl')
    cache_dict = {}

    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    cache_dict[item['namespace']] = item['embedding']
                except Exception as e:
                    print(f"Error reading cache: {e}")
                    return

    to_process = [ obj for obj in train if obj['namespace'] not in cache_dict ]

    print('embedding_preprocess_cache start, to_process:', len(to_process))
    emb = Embedding()
    with open(cache_file, 'a', encoding='utf-8') as f:
        for obj in tqdm(to_process, desc="Embedding encoding"):
            ns = obj['namespace']
            try:
                embedding = emb.embedding(obj['signature_requirement'])
                cache_dict[ns] = embedding
                f.write(json.dumps({'namespace': ns, 'embedding': embedding}) + '\n')
                f.flush()
            except Exception as e:
                print(f"namespace={ns} embedding failed: {e}")
                break
    print('embedding_preprocess_cache done')

def embedding_preprocess(train, output_path, number, cache_file):
    cache_dict = {}
    with open(cache_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                cache_dict[item['namespace']] = item['embedding']
            except Exception as e:
                print(f"Error reading cache: {e}")
    
    for obj in train:
        obj['embedding'] = cache_dict[obj['namespace']]
    

    scores = Embedding().get_scores(train)
    print(scores.shape)
    
    processed = []
    index = 0
    for obj in tqdm(train, total=len(train)):
        score = scores[index]
        rtn = sorted(enumerate(score), key=lambda x: x[1], reverse=True)[:number + 1]
        code_candidates = []
        for i in range(len(rtn)):
            data = train[rtn[i][0]]
            if data['namespace'] == obj['namespace']:
                continue
            code_candidates.append({'signature_requirement': data['signature_requirement'], 
                                           'score': rtn[i][1], 
                                        #    'idx':i+1,
                                           'function_name': data['function_name'],
                                           'ground_truth': data['ground_truth'],
                                           'namespace': data['namespace']
                                           })            
        processed.append({'signature_requirement': obj['signature_requirement'], 
                          'code_candidates': code_candidates,
                          'ground_truth': obj['ground_truth'],
                          'namespace': obj['namespace'],
                          'function_name': obj['function_name'],
                          'indent_space': obj['indent_space']
                          })
        index += 1
        
    with jsonlines.open(os.path.join(output_path, 'test_embedding_'+str(number)+'.jsonl'),'w') as f:
        f.write_all(processed)


      
        
def instance_selection(method, max_token_length, number, insturction, input_insturction, output_insturction, preprocess_file, output_path):
    '''
    insturction = 'Generate Python code based on the requirement \n'
    input_insturction = ' [INPUT] \n'
    output_insturction = ' [OUTPUT]v\n'
    preprocess_file = './preprocess/test_bm25_64.jsonl'
    output_path = './data'
    '''
    test = []
    with jsonlines.open(preprocess_file) as f:
        for i in f:
            test.append(i)

    tokenizer = tiktoken.encoding_for_model("gpt-4")
    prompts = []
    for obj in tqdm(test, total=len(test)):
        last_prompt = insturction.format(function_name=obj['function_name']) +\
            input_insturction + obj['signature_requirement'] + '\n' + output_insturction
        rest_token_length = max_token_length
        rest_token_length -= len(tokenizer.encode(last_prompt))
        
        topk = []
        topk = obj['code_candidates'][:number]
        prompt = ''
        count = 0
        
        for sample in topk:
            # print(sample['score'])
            sample_prompt = insturction.format(function_name=sample['function_name']) + \
                input_insturction + \
                sample['signature_requirement'] + '\n' + \
                output_insturction + \
                sample['ground_truth'] + '\n\n\n'
            len_sample_prompt = len(tokenizer.encode(sample_prompt))
            if len_sample_prompt > rest_token_length:
                print(obj['namespace'], 'overflow', count)
                break
            else:
                rest_token_length -= len_sample_prompt
                prompt += sample_prompt
                count += 1
        # print(count)    
        tmp_prompt = prompt + last_prompt
        prompts.append({
            'prompt': tmp_prompt,
            'function_name': obj['function_name'],
            'label': obj['ground_truth'],
            'namespace': obj['namespace'],
            'count': count,
            'indent_space': obj['indent_space']
            })
        
        # exit()
        
    # with jsonlines.open(os.path.join(output_path, method+'_prompt_'+str(number)+'.jsonl'),'w') as f:
    with jsonlines.open(output_path,'w') as f:
        f.write_all(prompts)
        
        
class BGE_Embedding:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.embeddings = SentenceTransformer('BAAI/bge-base-en-v1.5')
        
    def embedding(self, data):
        return self.embeddings.encode(data)
    
class Ada_Embedding:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = 'your key'
        os.environ["OPENAI_API_BASE"] = 'your url'
        from langchain_openai import OpenAIEmbeddings
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
    def embedding(self, data):
        return self.embeddings.embed_query(data)

class Embedding:
    def __init__(self, model="bge"):
        if model == 'bge':
            self.embeddings = BGE_Embedding()
        elif model == 'ada':
            self.embeddings = Ada_Embedding()
        else:
            raise ValueError(f"Invalid model: {model}")
    
    def get_scores(self, train_data):
        # embs = [self.embedding(d['signature_requirement']) for d in data]
        embeddings = [d['embedding'] for d in train_data]
        embs_np = np.array(embeddings) # (num, dim)
        print('embs_np.shape', embs_np.shape)
        norms = np.linalg.norm(embs_np, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        embs_norm = embs_np / norms
        scores = np.dot(embs_norm, embs_norm.T)
        return scores
        
    def embedding(self, data):
        return self.embeddings.embedding(data)
        

def read_code(json_obj):
    path = os.path.join('/root/workspace/code/DevEval/Source_Code2', json_obj['completion_path'])
    with open(path, 'r') as f:
        code = f.read()
    code_lines = code.split('\n')
    code_lines = code_lines[json_obj['signature_position'][0] - 1:json_obj['body_position'][1]]
    space_num = len(code_lines[0]) - len(code_lines[0].lstrip(' '))
    if space_num > 0:
        tmp = []
        for line in code_lines:
            if len(line) > space_num:
                tmp.append(line[space_num:])
            else:
                tmp.append('')
        code_lines = tmp
        # code_lines = [l[space_num:] for l in code_lines]
    ground_true = '\n'.join(code_lines)
    # if space_num > 0:
    #     print(ground_true)
    #     print('========================================')
        # exit()
    return ground_true, ' ' * space_num


def preprocess(meta_path, output_path, number, method, output_dir):
    train_path = meta_path
    train = []
    with jsonlines.open(train_path) as f:
        for obj in f:
            # signature_requirement, docstring_tokens, namespace， ground_truth
            docstring_tokens = f'''{obj['requirement']['Functionality']}
{obj['requirement']['Arguments']}'''
            signature_requirement = f'''Signature: {obj['signature']}
Requirement:
    {docstring_tokens}'''
    
            ground_truth, space = read_code(obj)
            func_signature = obj['signature']
            function_name = func_signature[func_signature.index('def ')+4 : func_signature.index('(')]
            d = {'signature_requirement': signature_requirement, 
                 'namespace': obj['namespace'], 
                 'ground_truth': ground_truth,
                 'function_name': function_name,
                 'indent_space': space,
                 }
            train.append(d)
    
    if method == 'bm25':
        bm25_preprocess(train, train, output_path, number)
    elif method == 'embedding':
        cache_file = os.path.join(output_dir, 'embedding_cache.jsonl')
        embedding_preprocess_cache(train, cache_file)
        embedding_preprocess(train, output_path, number, cache_file)
    else:
        raise ValueError(f"Invalid type: {method}")

@DeprecationWarning    
def make_prompt(method, max_token_length, number, preprocess_file, output_path):
    insturction = 'Please complete the {function_name} function based on the provided Signature and Requirement \n'
    input_insturction = ' [INPUT]\n'
    output_insturction = ' [OUTPUT]\n'
    instance_selection(method=method, 
                       max_token_length=max_token_length,
                       number=number, 
                       insturction=insturction, 
                       input_insturction=input_insturction, 
                       output_insturction=output_insturction, 
                       preprocess_file=preprocess_file, 
                       output_path=output_path)


def make_prompt2(template_path, 
                 template_item_path,
                 input_data_path, # meta_data
                 elem_indexing_path, # prompt_elements, with indexing
                 number=5,
                 output_path='./data/DevEval/similarity/similarity_prompt.jsonl',
                 key_name='namespace',
                 language='python'):
    input_data = utils.load_json_data(input_data_path)
    print(f'input_data_path: {input_data_path}')
    print(f'elem_indexing_path: {elem_indexing_path}')
    elem_index_dict = utils.load_json_data_as_dict(elem_indexing_path, key_name=key_name)
    
    # output_path = os.path.join(output_dir, f'{method}_prompt_{number}.jsonl')
    with open(output_path, 'w') as f:
        for meta in tqdm(input_data, total=len(input_data)):
            template = open(template_path, 'r').read()
            elem = elem_index_dict[meta[key_name]]
            candidates = elem['candidates']
            del elem['candidates']
            items = []
            for i in range(number):
                can = candidates[i]
                template_item = open(template_item_path, 'r').read()
                template_item = template_item.format(
                    index=i+1,
                    function_name=can['function_name'],
                    contexts_above=can['contexts_above'],
                    contexts_below=can['contexts_below'],
                    input_code=can['input_code'],
                    function_code=can['ground_truth'],
                    language=language
                )
                items.append(template_item)
            
            examples = '\n\n'.join(items)
            prompt = template.format(
                examples=examples,
                function_name=elem['function_name'],
                contexts_above=elem['contexts_above'],
                contexts_below=elem['contexts_below'],
                input_code=elem['input_code'],
                language=language
            )
            elem['prompt'] = prompt
            f.write(json.dumps(elem) + '\n')
        
    print(f'make_prompt2 done, output_path: {output_path}')
            
        
def main():
    train_path = '/root/workspace/code/DevEval/data_fixed2.jsonl'
    # test_path = train_path
    output_path = './similarity'
    number = 64
    train = []
    with jsonlines.open(train_path) as f:
        for obj in f:
            # signature_requirement, docstring_tokens, namespace， ground_truth
            docstring_tokens = f'''{obj['requirement']['Functionality']}
{obj['requirement']['Arguments']}'''
            signature_requirement = f'''Signature: {obj['signature']}
Requirement:
    {docstring_tokens}'''
    
            ground_truth, space = read_code(obj)
            func_signature = obj['signature']
            function_name = func_signature[func_signature.index('def ')+4 : func_signature.index('(')]
            d = {'signature_requirement': signature_requirement, 
                 'namespace': obj['namespace'], 
                 'ground_truth': ground_truth,
                 'function_name': function_name,
                 'indent_space': space,
                 }
            train.append(d)
            
    # bm25_preprocess(train, train, output_path, number)
    # exit()
    # embedding_preprocess_cache(train, output_path)
    # embedding_preprocess(train, output_path, number, './similarity/embedding_cache.jsonl')
    # exit()
    # insturction = 'Generate Python code based on the requirement \n'
    insturction = 'Please complete the {function_name} function based on the provided Signature and Requirement \n'
    input_insturction = ' [INPUT]\n'
    output_insturction = ' [OUTPUT]\n'
    # method = 'bm25'
    # preprocess_file = './similarity/test_bm25_64.jsonl'
    method = 'embedding'
    preprocess_file = './similarity/test_embedding_64.jsonl'
    output_path = './similarity'
    instance_selection(method=method, 
                       max_token_length=5000,
                       number=8, 
                       insturction=insturction, 
                       input_insturction=input_insturction, 
                       output_insturction=output_insturction, 
                       preprocess_file=preprocess_file, 
                       output_path=output_path)


if __name__ == '__main__':
    # main()
    # bm25_indexing(train_prompt_elem_path='./data/DevEval/prompt_elements_source_code2.jsonl',
    #               test_meta_data_path='./data/DevEval/data_fixed2_sample_pre_proj.jsonl',
    #               output_dir='./data/DevEval/similarity',
    #               number=5,
    #               key_name='namespace')
    
    # make_prompt2(template_path='./prompt/template/similarity_template.txt',
    #              template_item_path='./prompt/template/similarity_template_item.txt',
    #              input_data_path='./data/DevEval/data_fixed2_sample_pre_proj.jsonl',
    #              elem_indexing_path='./data/DevEval/similarity/indexing_bm25_5.jsonl',
    #              number=5,
    #              output_path='./data/DevEval/similarity/prompt_bm25_5.jsonl')
    
    # ----------CoCoSoDa------------
    timer = Timer()
    timer.start()
    get_cocosoda_embeddings()
    train_embedding_path = 'data/DevEval/similarity/cocosoda_embeddings_prompt_elements_source_code2.npy' 
    test_embedding_path = 'data/DevEval/similarity/cocosoda_embeddings_prompt_elements_source_code2_sample_per_proj.npy'
    train_prompt_elem_path = './data/DevEval/prompt_elements_source_code2.jsonl'
    test_meta_data_path = './data/DevEval/data_fixed2_sample_pre_proj.jsonl'
    output_dir = './data/DevEval/similarity'
    embedding_indexing(train_embedding_path, 
                       test_embedding_path,
                       train_prompt_elem_path, 
                       test_meta_data_path, 
                       output_dir,
                       number=5,
                       key_name='namespace')
    timer.end()
    print(f"CoCoSoDa embedding and indexing Time cost: {timer.get_elapsed_ms()}ms")
    
    # make_prompt2(template_path='./prompt/template/similarity_template.txt',
    #              template_item_path='./prompt/template/similarity_template_item.txt',
    #              input_data_path='./data/DevEval/data_fixed2_sample_pre_proj.jsonl',
    #              elem_indexing_path='./data/DevEval/similarity/indexing_CoCoSoDa_5.jsonl',
    #              number=3,
    #              output_path='./data/DevEval/similarity/prompt_cocosoda_3.jsonl')