import json
from time import sleep
import numpy as np
import openai
from openai import OpenAI
import re

def parse_important_entities(result):
    try:
        # 尝试直接解析列表
        entities = eval(result)
        if isinstance(entities, list) and all(isinstance(ent, str) for ent in entities):
            return entities
    except:
        # 正则匹配实体列表（兼容非标准格式）
        pattern = r'\[(.*?)\]'
        matches = re.findall(pattern, result, re.DOTALL)
        if matches:
            try:
                entities = [ent.strip().strip("'\"") for ent in matches[-1].split(',')]
                return [e for e in entities if e] 
            except:
                pass
    return []

def predict_chatgpt_cot_step1(prompt, args):
    sys_instruction = """You need to select important entities from the given first-order historical sequence of knowledge triplets. 
The knowledge triplets are in the form of {time}:[{subject}, {relation}, {object}]. 
You must return ONLY a list of important entity names (strings) without any explanation, e.g., ["entity1", "entity2"].
Do not return any other content except the entity list."""
    prompt = [
        {"role": "system", "content": sys_instruction},
        {"role": "user", "content": prompt},
    ]
    
    client = OpenAI(api_key="", base_url="")
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            max_tokens=512,
            temperature=0.0,
            top_p=1,
            n=1,
            stop=None,
        )
        # 解析重要实体
        entities = parse_important_entities(completion.choices[0].message.content)
        return entities
    except Exception as e:
        print(f"COT Step1 (chatgpt) error: {e}")
        return []

def predict_chatgpt_span_cot_step1(prompt, args):
    sys_instruction = """You need to select important entities from the given first-order historical sequence of knowledge triplets. 
The knowledge triplets are in the form of [{subject_entity}, {relation}, {object_entity}] with timestamps. 
You must return ONLY a list of important entity names (strings) without any explanation, e.g., ["entity1", "entity2"].
Do not return any other content except the entity list."""
    prompt = [
        {"role": "system", "content": sys_instruction},
        {"role": "user", "content": prompt},
    ]
    
    client = OpenAI(api_key="", base_url="")
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            max_tokens=512,
            temperature=0.0,
            top_p=1,
            n=1,
            stop=None,
        )
        # 解析重要实体
        entities = parse_important_entities(completion.choices[0].message.content)
        return entities
    except Exception as e:
        print(f"COT Step1 (chatgpt_span) error: {e}")
        return []

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

def predict_chatgpt(prompt, args, important_entities=None):
    cot_context = ""
    if important_entities and len(important_entities) > 0:
        cot_context = f"First, the important entities selected from historical sequences are: {', '.join(important_entities)}. You must use the historical information of these entities to assist your reasoning. "
    
    sys_instruction = f"{cot_context}You must be able to correctly predict the next {relation_label} and {object_label}, from a given history consisting of multiple quadruplets in the form of {time}:[{subject}, {relation}, {object_label}] and the query in the form of {time}:[{subject}, in the end. You must generate only two numbers for {relation_label}, {object_label} without any explanation. In the form of [[{relation_label}, {object_label}], [{relation_label}, {object_label}], ...] and give me 500 of the most likely candidate results. Rank them with possibility"
    
    prompt = [
        {"role": "system", "content": sys_instruction},
        {"role": "user", "content": prompt},
    ]
    
    client = OpenAI(api_key="", base_url="")
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
    except Exception as e:
        print(f"COT Step2 (chatgpt) error: {e}")
        return []

def predict_chatgpt_span(prompt, args, important_entities=None):
    cot_context = ""
    if important_entities and len(important_entities) > 0:
        cot_context = f"First, the important entities selected from historical sequences are: {', '.join(important_entities)}. You must use the historical information of these entities to assist your judgment. "
    
    sys_instruction = f"{cot_context}You must determine whether the given knowledge will ends at the current timestamp. Knowledges are in the form of [{subject_entity}, {relation}, {object_entity}].Respond only with Yes or No without any explanation."
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
        parsed_results = parse_results_chatgpt_span(completion.choices[0].message.content)  # type: ignore
        return parsed_results
    except Exception as e:
        print(f"COT Step2 (chatgpt_span) error: {e}")
        return -1