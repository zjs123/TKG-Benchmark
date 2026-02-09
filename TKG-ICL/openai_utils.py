import json
from time import sleep

import numpy as np
import openai
from openai import OpenAI
import re


def parse_results_chatgpt(result):
    if ', ...' in result:
        result = result.replace(', ...', '')
    try:
        return_list = eval(result)
    except:
        if '.' in result:
            result = re.sub(r'\.\w+', '', result)
        pattern = r'\[\d+,\d+\]'  # 匹配[数字,数字]格式
        matches = list(re.finditer(pattern, result))
        # 获取最后一个匹配项
        try:
            last_match = matches[-1]
             # 返回最后一个匹配项之前的所有字符
            result =  result[:last_match.end()]+ "]"
            return_list = eval(result)
        except:
            try:
                pattern = r'\[\d+, \d+\]'  # 匹配[数字,数字]格式
                matches = list(re.finditer(pattern, result))
                last_match = matches[-1]
                 # 返回最后一个匹配项之前的所有字符
                result =  result[:last_match.end()]+ "]"
                return_list = eval(result)
            except:
                return_list = []
    
    if type(return_list) != list:
        return_list =list(return_list)
    
    parse_return_list = []
    for item in return_list:
        if type(item[0]) == list:
            for in_item in item:
                parse_return_list.append(in_item)
        else:
            parse_return_list.append(item)
    
    new_parse_return_list = []
    for item in parse_return_list:
        new_parse_return_list.append((item[0],item[1]))
    
    return new_parse_return_list

def parse_results_chatgpt_span(result):
    parse_return = -1
    if 'Yes' in result or 'yes' in result or 'YES' in result:
        parse_return = 1
    elif 'No' in result or 'no' in result or 'NO' in result:
        parse_return = 0
   
    return parse_return

def predict_chatgpt(prompt, args):
    sys_instruction = "You must be able to correctly predict the next {relation_label} and {object_label}, from a given history consisting of multiple quadruplets in the form of {time}:[{subject}, {relation}, {object_label}] and the query in the form of {time}:[{subject}, in the end. You must generate only two numbers for {relation_label}, {object_label} without any explanation. In the form of [[{relation_label}, {object_label}], [{relation_label}, {object_label}], ...] and give me 500 of the most likely candidate results. Rank them with possibility"
    prompt = [
        {"role": "system", "content": sys_instruction},
        {"role": "user", "content": prompt},
    ]
    
    client = OpenAI(api_key="", base_url="")
    completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=prompt,
                max_tokens=512,
                temperature=0.0,
                top_p=1,
                n=1,
                stop=["]]"],
                )
    print(completion)
    parsed_results = parse_results_chatgpt(completion.choices[0].message.content)  # type: ignore
    return parsed_results
    '''
    try:
        completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=prompt,
                max_tokens=512,
                temperature=0.0,
                top_p=1,
                n=1,
                stop=["]]"],
                )
        print(completion)
        parsed_results = parse_results_chatgpt(completion.choices[0].message.content)  # type: ignore
        return parsed_results
    except:
        pass
    '''

def predict_chatgpt_span(prompt, args):
    sys_instruction = "You must determine whether the given knowledge will ends at the current timestamp. Knowledges are in the form of [{subject_entity}, {relation}, {object_entity}].Respond only with Yes or No without any explanation."
    prompt = [
        {"role": "system", "content": sys_instruction},
        {"role": "user", "content": prompt},
    ]
    
    client = OpenAI(api_key="", base_url="")
    try:
        completion = client.chat.completions.create(
                model="gpt-4o-mini", #"gpt-3.5-turbo",
                messages=prompt,
                max_tokens=512,
                temperature=0.0,
                top_p=1,
                n=1,
                stop=["]]"],
                )
        #print(completion)
        parsed_results = parse_results_chatgpt_span(completion.choices[0].message.content)  # type: ignore
        return parsed_results
    except:
        pass
    

