from pathlib import Path
import json
import subprocess
import psutil
from subprocess import run
from tqdm import tqdm
import os
import numpy as np
from argparse import ArgumentParser
import textwrap
from func_timeout import func_set_timeout
import func_timeout
import time
import utils


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--output_file', type=Path)
    parser.add_argument('--log_file', type=Path)
    parser.add_argument('--source_code_root', type=Path, default=Path('Source_Code'))
    parser.add_argument('--data_file', type=Path, default=Path('data.jsonl')) # data.jsonl
    parser.add_argument('--k', type=str, default='1,3,5,10') # k in pass_at_k
    parser.add_argument('--n', type=int, default=1) # number of completions per task
    return parser.parse_args()


def adjust_indent(code, new_indent):
    # remove original indentation
    dedented_code = textwrap.dedent(code)
    # add new indentation
    indented_code = textwrap.indent(dedented_code, ' ' * new_indent)
    return indented_code

@func_set_timeout(180)
def execution_tests_na_4ground_true(args, data):
    project_path = os.path.join(args.source_code_root, data['project_path'])
    
    for test in data['tests']:
        cmd = f"""source myenv/bin/activate
pytest {test};
status=$?
if [ $status -eq 0 ]; then
    echo "na echo: Tests passed successfully."
else
    echo "na echo 1024999: Tests failed with status code $status."
fi
deactivate;
exit $status"""
        
        test_cmd = f"cd {project_path} && {cmd} "
        process = subprocess.Popen(cmd, 
                                   cwd=project_path,
                                   shell=True, 
                                   text=True, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        try:
            while True:
                process_id = process.pid
                process_memory = psutil.Process(process_id).memory_info().rss
                if process_memory > 5 * 1024 * 1024 * 1024: # 5GB memory usage per test
                    process.terminate()
                    process.wait()
                    print('OOM', test_cmd)
                    return 'OOM' # Out of Memory
                return_code = process.poll()
                
                if return_code is not None:
                    if return_code != 0:
                        stdout, stderr = process.communicate()
                        
                        process.terminate()
                        process.wait()
                        print('cmd:  ', test_cmd)
                        print('Out:  ', stdout)
                        print('Error:', stderr)
                        
                        return 'Error', test_cmd, stderr # Execution Error
                    else:
                        stdout, stderr = process.communicate()
                        # process.terminate()
                        # process.wait()
                        # print('cmd:  ', test_cmd)
                        # print('Out:  ', stdout)
                        # print(stdout)
                        break
        except Exception as e:
            print('test:',test,'exception:',e, test_cmd)
            process.terminate()
            process.wait()
            return 'Error', test_cmd, e # Other Error
        finally:
            process.terminate()
            process.wait()
    return 'Pass', '', '' # Pass

@func_set_timeout(180)
def execution_tests_na(args, data):
    project_path = os.path.join(args.source_code_root, data['project_path'])
    
    for test in data['tests']:
        cmd = f"""source myenv/bin/activate
pytest {test};
status=$?
if [ $status -eq 0 ]; then
    echo "na echo: Tests passed successfully."
else
    echo "na echo 1024999: Tests failed with status code $status."
fi
deactivate;
exit $status"""
        
        test_cmd = f"cd {project_path} && {cmd} "
        process = subprocess.Popen(cmd, 
                                   cwd=project_path,
                                   shell=True, 
                                   text=True, 
                                   stdout=subprocess.PIPE, 
                                   stderr=subprocess.PIPE)
        try:
            while True:
                process_id = process.pid
                process_memory = psutil.Process(process_id).memory_info().rss
                if process_memory > 5 * 1024 * 1024 * 1024: # 5GB memory usage per test
                    process.terminate()
                    process.wait()
                    print('OOM', test_cmd)
                    return 'OOM' # Out of Memory
                return_code = process.poll()
                
                if return_code is not None:
                    if return_code != 0:
                        # stdout, stderr = process.communicate()
                        
                        # process.terminate()
                        # process.wait()
                        # print('cmd:  ', test_cmd)
                        # print('Out:  ', stdout)
                        # print('Error:', stderr)
                        
                        return 'Error' # Execution Error
                    else:
                        # stdout, stderr = process.communicate()
                        # process.terminate()
                        # process.wait()
                        # print('cmd:  ', test_cmd)
                        # print('Out:  ', stdout)
                        break
        except Exception as e:
            # print('test:',test,'exception:',e, test_cmd)
            process.terminate()
            process.wait()
            return 'Error' # Other Error
        finally:
            process.terminate()
            process.wait()
    return 'Pass' # Pass

def compute_pass_at_k(n, c, k):
    """
    n: total number of completions per task
    c: number of completions that pass all tests
    k: k in pass_at_k
    """
    if n - c < k:
        return 1
    else:
        return 1.0 - np.prod(1.0 - k / np.arange(n-c+1, n+1))


def SetUp_evaluation(args, data, completion):
    # print('''data['completion_path']''', data['completion_path'])
    completion_path = Path(data['completion_path'])
    # print('''args.source_code_root''', args.source_code_root)
    completion_path = os.path.join(args.source_code_root, completion_path)
    
    head_tail = os.path.split(completion_path)
    completion_tmp_path = os.path.join(head_tail[0], 'tmp_' + head_tail[1])

    # rename the original completion file as tmp_completion
    run(['cp', completion_path, completion_tmp_path])

    # write the new completion file
    sos, eos = data['body_position'][0]-1, data['body_position'][1]
    
    # print(data)
    
    with open(completion_path, 'r') as f:
        # print('kkk', completion_path)
        # exit()
        file_lines = f.readlines()
    file_lines = file_lines[:sos] + ['\n', completion, '\n'] + file_lines[eos:]
    
    # print(''.join(file_lines))
    
    with open(completion_path, 'w') as f:
        f.write(''.join(file_lines))


def TearDown_evaluation(args, data):
    completion_path = Path(data['completion_path'])
    completion_path = os.path.join(args.source_code_root, completion_path)
    head_tail = os.path.split(completion_path)
    completion_tmp_path = os.path.join(head_tail[0], 'tmp_' + head_tail[1])
    run(['mv', completion_tmp_path, completion_path])


def check_correctness(args, data):
    completion = data['completion']
    if completion == "    pass\n":
        return 'Error'
    completion = adjust_indent(completion, data['indent'])
    
    SetUp_evaluation(args, data, completion)
    try:
        flag = execution_tests_na(args, data)
    except func_timeout.exceptions.FunctionTimedOut:
        print('timeout')
        flag = 'TimeOut'
    TearDown_evaluation(args, data)
    return flag


def report_results(args, benchmark_data):
    if not os.path.exists(args.log_file):
        raise ValueError(f'{args.log_file} does not exist')
    
    # Collect passed completions for each namespace
    passed_completion = {}
    with open(args.log_file, 'r') as f:
        for line in f:
            # print(line)
            js = json.loads(line)
            if 'pass' in js:
                js['Result'] = js['pass']
            if js['Result'] == 'Pass':
                namespace, completion = js['namespace'], js['completion']
                if namespace not in passed_completion:
                    passed_completion[namespace] = set()
                passed_completion[namespace].add(completion)

    # Iterate through all completions and count the number of passed completions for each namespace
    results = {}
    with open(args.output_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace, completion = js['namespace'], js['completion']
            if namespace not in benchmark_data:
                continue
            if namespace not in results:
                results[namespace] = 0
            if namespace in passed_completion and completion in passed_completion[namespace]:
                results[namespace] += 1
            
    # Compute Pass@k
    k_list = [int(k) for k in args.k.split(',')]
    for k in k_list:
        if k > args.n:
            continue
        pass_at_k = np.mean([compute_pass_at_k(args.n, pass_num, k) for namespace, pass_num in results.items()])
        print(f'pass_at_{k}: {pass_at_k*100}%')
        

def report_results_simple(log_file, benchmark_data, namespace_only=None):
    if not os.path.exists(log_file):
        raise ValueError(f'{log_file} does not exist')
    
    log_data = utils.load_jsonl(log_file)
    # Collect passed completions for each namespace
    passed_completion = {}
    with open(log_file, 'r') as f:
        for line in f:
            # print(line)
            js = json.loads(line)
            if 'pass' in js:
                js['Result'] = js['pass']
            if js['Result'] == 'Pass':
                namespace, completion = js['namespace'], js['completion']
                if namespace not in passed_completion:
                    passed_completion[namespace] = set()
                passed_completion[namespace].add(completion)

    # Iterate through all completions and count the number of passed completions for each namespace
    results = {}
    with open(args.output_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace, completion = js['namespace'], js['completion']
            if namespace not in benchmark_data:
                continue
            if namespace not in results:
                results[namespace] = 0
            if namespace in passed_completion and completion in passed_completion[namespace]:
                results[namespace] += 1
            
    # Compute Pass@k
    k_list = [int(k) for k in args.k.split(',')]
    for k in k_list:
        if k > args.n:
            continue
        pass_at_k = np.mean([compute_pass_at_k(args.n, pass_num, k) for namespace, pass_num in results.items()])
        print(f'pass_at_{k}: {pass_at_k*100}%')


def load_finished_data(args):
    finished_data = {}
    if os.path.exists(args.log_file):
        with open(args.log_file, 'r') as f:
            for line in f:
                # if len(line) > 755:
                #     print(line[754: 757])
                # # print(len(line))
                # print(line)
                js = json.loads(line)
                namespace, completion = js['namespace'], js['completion']
                if namespace not in finished_data:
                    finished_data[namespace] = set()
                finished_data[namespace].add(completion)
                # finished_data[namespace].add(js['id'])
    return finished_data


def main(args):
    finished_data = load_finished_data(args)#  { namespace:[complection1, completion2]}
    # finished_data = {}
    
    benchmark_data = {}
    with open(args.data_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace = js['namespace']
            benchmark_data[namespace] = js
            
    failed_set, projs = get_ground_true_failed()
    # print(len(failed_set))
    # exit()
    benchmark_data = {key: benchmark_data[key] for key in benchmark_data if key not in failed_set}

    sorted_items = sorted(benchmark_data.items())
    benchmark_data = dict(sorted_items)
    
    print('benchmark_data count:', len(benchmark_data))
    
    todo_output_data = [] 
    exclude_outputs = []
    # print('finished_data:=============')
    # print(finished_data)
    with open(args.output_file, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace, completion = js['namespace'], js['completion']
            
            # if namespace in benchmark_data and benchmark_data[namespace]['project_path'] in exclude_repos:
            #     exclude_outputs.append(completion)
            #     continue
            if namespace in failed_set or namespace not in benchmark_data:
                exclude_outputs.append(completion)
                # print('exclude:', namespace)
                continue
            # print(js['id'])
            if namespace not in finished_data:
                # print(f'{namespace} not in finished_data')
                todo_output_data.append(js)
                finished_data[namespace] = set()
                finished_data[namespace].add(completion) 
            elif completion not in finished_data[namespace]:
                todo_output_data.append(js)
                finished_data[namespace].add(completion)         
    del finished_data
    print("TODO Completions: ", len(todo_output_data))
    print("Exclude Completions: ", len(exclude_outputs))

    
    i = 0
    with open(args.log_file, 'a') as f:
        for output in tqdm(todo_output_data):
            if output['namespace'] in benchmark_data:
                i+=1
                data = benchmark_data[output['namespace']]
                data['completion'] = output['completion'] 
                flag = check_correctness(args, data)
                # exit()
                output['Result'] = flag
                # print('flag:',flag)
                f.write(json.dumps(output) + '\n')
                f.flush()
            # if i > 10:
            #     break

    report_results(args, benchmark_data)


def test_ground_truth(args):
    data = open(args.data_file, 'r').readlines()
    # data = open('failed_samples_na_fixed2_venv_fix_pytest.jsonl', 'r').readlines()
    # output_f = open('failed_samples.jsonl', 'w')
    output_f = open('failed_samples_groundtrue_SC2_20250820.jsonl', 'w')
    i = 0
    for line in tqdm(data):
        i+=1
        # if i < 141:
        #     continue
        js = json.loads(line)
        tests = set(js['tests'])
        js['tests'] = list(tests)
        try:
            flag, test_cmd, e = execution_tests_na_4ground_true(args, js)
        except func_timeout.exceptions.FunctionTimedOut:
            flag = 'TimeOut'
        if flag != 'Pass':
            print(js['namespace'], js['project_path'], flag)
            js['test_cmd'] = test_cmd
            js['error'] = e
            output_f.write(json.dumps(js) + '\n')
        # print(flag)
        # if i > 10:
        #     break

def get_ground_true_failed():
    # fail_path = '/root/workspace/code/DevEval/env_failed_samples.jsonl'
    fail_path = '/root/workspace/code/DevEval/failed_samples_groundtrue_SC2_20250820.jsonl'
    failed_samples = set()
    projs = set()
    with open(fail_path, 'r') as f:
        for line in f:
            js = json.loads(line)
            namespace = js['namespace']
            failed_samples.add(namespace)
            projs.add(js['project_path'])
            # benchmark_data[namespace] = js
    
    from data.DevEval.make_fail_completion import load_invalid_set_from_file
    invalid_set = load_invalid_set_from_file('./data/DevEval/invalid_set.json')
    failed_samples = failed_samples.union(invalid_set)
    
    print(f'failed: sample count:{len(failed_samples)}, failed project count: {len(projs)}')
    return failed_samples, projs
    
    

if __name__ == '__main__':
    args = get_parser()
    # print(args)
    # exit()
    if args.output_file is None:
        test_ground_truth(args)
    else:
        main(args)