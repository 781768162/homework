#-*- coding:utf-8 -*-
from openai import OpenAI
from functools import wraps
import time

def time_counter(func):
#一个装饰器，用来计算函数耗时用的
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        used_time = end - start
        print(f"\n{func.__name__}函数用时{used_time:.6f}s\n")
        return result
    return wrapper

def extract_columns(input_file=r"posts_groundtruth.txt"):
#读取数据集，提取text和label
    text, label = [], []
    with open(input_file, 'r', encoding='utf-8') as f:     
        for line in f:
            columns = line.strip().split('\t')

            if len(columns) >= 6:
                text.append(columns[1])
                label.append(columns[-1])
            else:
                pass
    return text[1:], label[1:]

@time_counter
def chat_model(text: list): 
#利用openai库的api与模型交互
    client = OpenAI(
    base_url = "http://localhost:11434/v1",
    api_key = "ollama"
    )

    res, emotion = [], []
    for content in text:
        response = client.chat.completions.create(
            model="qwen3:4b",
            messages=[
                {"role": "system", "content": "你是一个辨别新闻情感色彩的ai助手，你需要阅读所给的文本，判断这段新闻的情感色彩，用一个两字的中文词汇概括，你的输出只能包含一个中文词汇。/no_think"},
                {"role": "user", "content": f"{content}"}
            ],
            temperature=0.7,
            max_tokens=5000
        )
        emotion.append(response.choices[0].message.content[-2:])
        response = client.chat.completions.create(
            model="qwen3:4b",
            messages=[
                {"role": "system", "content": "你是一个辨别真假新闻的ai助手，你需要阅读所给的文本，判断这段新闻的真实性，你的输出只能包含一个单词 ，新闻真实为real，虚假为fake。/no_think"},
                {"role": "user", "content": f"{content}"}
            ],
            temperature=0.7,
            max_tokens=5000
        )
        res.append(response.choices[0].message.content[-4:])
    
    return res, emotion
    
def count_acc(label: list, res: list):
    total_len = min(len(label), len(res))
    correct_num_real, correct_num_fake = 0, 0
    for i in range(total_len):
        if label[i] == res[i]:
            if label[i] == 'fake':
                correct_num_fake += 1
            else:
                correct_num_real += 1
    acc = (correct_num_real + correct_num_fake) / total_len
    acc_real = correct_num_real / total_len
    acc_fake = correct_num_fake / total_len
    return acc, acc_real, acc_fake

if __name__ == "__main__":
    text, label = extract_columns()
    res, emotion = chat_model(text)
    print(emotion)
    acc, acc_real, acc_fake = count_acc(label, res)
    print(f"正确率：{acc}，真新闻正确率：{acc_real}， 假新闻正确率：{acc_fake}")
