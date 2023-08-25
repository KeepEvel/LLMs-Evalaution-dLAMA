import re
import time
import json
import csv
import unicodedata

import numpy as np
import pandas as pd

from tqdm import tqdm
from pprint import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer
from file_save import *

BLOOM_MODEL = ["bloomz-1b1", "bloomz-1b7", "bloomz-560m", "bloom-560m", "bloom-1b1", "bloom-1b7", "bloom-3b", "bloomz-3b", "bloom-7b1", "bloomz-7b1"]

SYSTEM_DESCRIPTION = {"zh": "你是一个百科全书", 'en': "You are an encyclopedia."}

PRETICATES = ["P1412", "P1376", "P1303", "P530", "P495", "P449", "P364", "P264", "P190", "P136", "P106", "P103", "P47", "P37", "P36", "P30", "P27", "P20", "P19", "P17"]
PROMPTS = {
    "P1412": {
        "zh": '问题: {}用什么国家语言交流? 答案是:',
        'en': 'Question: What language does {} communicate? The answer is:',
    },
    "P1376": {
        "zh": '问题: 哪个国家的首都是{}? 答案是:',
        'en': 'Question: Which country has {} as its capital? The answer is:',
    },
    "P1303": {
        "zh": "问题：{}玩什么乐器？答案是：",
        'en': 'Question: What instrument does {} play? The answer is:',
    },
    "P530": {
        "zh": '问题：与{}保持外交关系的国家是? 答案是：',
        'en': 'Question: What is the country that maintains diplomatic relations with {}? The answer is:',
    },
    "P495": {
        "zh": '问题：哪一个国家创造了{}? 答案是：',
        'en': 'Question: Which country created {}? The answer is:',
    },
    "P449": {
        "zh": "问题：{}最初是在哪里播出的? 答案是：",
        'en': "Question: Where was {} originally aired on? The answer is:",
    },
    "P364": {
        "zh": '问题：{}的起源语言是什么? 答案是：',
        'en': "Question: What is the language of origin of {}? The answer is:",
    },
    "P264": {
        "zh": '问题：{}与哪个唱片公司签约? 答案是:',
        'en': 'Question: Which record label is {} signed to? The answer is:',
    },
     "P190": {
        "zh": '问题：{}的姐妹城市是? 答案是：',
        'en': 'Question: What is the sister city of {}? The answer is: ',
    },
     "P136": {
        "zh": '问题：{}与哪种音乐流派有关？答案是:',
        'en': 'Question: Which musical genre is {} associated with? The answer is:',
    },
    
     "P17": {
        "zh": '问题：{}位于哪个国家？答案是：',
        'en': 'Question: In which country is {} located? The answer is: ',
    },
    "P19":{
        "zh": '问题：{}出生于哪个城市？答案是：',
        'en': 'Question: In which city was {} born? The answer is: ',
    },
    "P20":{
        "zh": '问题：{}在哪个城市去世？答案是：' ,
        'en':'Question: In which city did {} die? The answer is: ',
    },
    "P27":{
        "zh": '问题：{}是哪个国家的公民？答案是：' ,
        'en':'Question: What country is {} a citizen of? The answer is: ',
    },
    "P30": {
        "zh": '问题：{}位于哪个大洲？答案是：' ,
        'en': 'Question: Which continent is {} located in? The answer is:',
    },
    "P36": {
        "zh": '问题：{}的首都是？答案是：' ,
        'en': 'Question: What is the capital of {}? The answer is:',
    },
    "P37": {
        "zh": '问题：{}的官方语言是什么？ 答案是：' ,
        'en': 'Question: What is the official language of {}? The answer is: ',
    },
    "P47": {
        "zh": '问题：{}与哪个国家接壤？答案是：' ,
        'en': 'Question：What is the country that shares border with {}? The answer is:',
    },
    "P103":{
        "zh": '问题：{}的母语是什么？答案是：' ,
        'en':'Question: What is the native language of {}? The answer is: ',
    },
    "P106":{
        "zh": '问题：{}的职业是什么？答案是：' ,
        'en':'Question: What is the profession of {}? The answer is: ',
    }

}

def get_bloom_output(predicate, entity, lang, model, tokenizer): # one question per call
    """Answer a question using bloom model by GPU."""
    
    prompt = PROMPTS[predicate][lang].format(entity)

    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs, max_new_tokens=5)
    
    outputs_decode = tokenizer.decode(outputs[0])
    return [get_key_answers(get_answer_only(prompt, outputs_decode))]


def probe_bloom_using_DLAMA_jsonl(predicate, lang, input_file_path, model, tokenizer):
    """Probe Bloom using a DLAMA predicate and save the tuples with the answers to a jsonl file."""
    output_triples = []
    count = 0

    lang1 = lang
    if lang == "en-ar" or lang == "en-es" or lang == "en-zh":
        lang1 = "en"
    for triple in tqdm(load_jsonl_file(input_file_path)):
        # Handle rate-limiting in a naive way!
        while True:
            try:
                choices = get_bloom_output(predicate, triple["sub_label"], lang1, model, tokenizer)
                break
            except:
                print("There is bug")
                # TODO: Tune the time to wait before sending another request
                time.sleep(5)

        triple["bloom_choices"] = choices
        output_triples.append(triple)        
    return output_triples

# the original method in dLAMA
# def compute_accuracy(tuples):
#     """Measure the BLOOM model's accuracy."""
#     n_correct = int(0)
#     n_correct_substr = int(0)
#     n_tuples = len(tuples)
#     for t in tuples:
#         gpt_choices = t["bloom_choices"]
#         obj_labels = t["obj_label"]

#         # An answer is correct if it is one of candidate answers
#         n_correct += int(any([c in obj_labels for c in gpt_choices]))

#         # An answer is correct if any of the candidate answers is a substr of the answer
#         n_correct_substr += int(any([o in c for o in obj_labels for c in gpt_choices]))

#     return {
#         "Total number of tuples": n_tuples,
#         "# of correct answers (Exact match)": n_correct,
#         "% of correct answers (Exact match)": round(100 * n_correct / n_tuples, 1),
#         "# of correct answers (Overlap)": n_correct_substr,
#         "% of correct answers (Overlap)": round(100 * n_correct_substr / n_tuples, 1),
#     }

def compute_accuracy(tuples):
    """Measure the model's accuracy."""
    n_correct = int(0)
    n_correct_substr = int(0)
    n_tuples = len(tuples)
    for t in tuples:
        gpt_choices = t["bloom_choices"]
        #gpt_choices = t["gpt3.5-turbo_choices"]
        for item in range(len(gpt_choices)):
            gpt_choices[item] = gpt_choices[item].strip()
            
        obj_labels = t["obj_label"]

        # An answer is correct if it is one of candidate answers
        n_correct += int(any([c in obj_labels for c in gpt_choices]))

        # An answer is correct if any of the candidate answers is a substr of the answer
        count = 0
        
        for c in gpt_choices:
            for o in obj_labels:
                if not o.isupper():
                    o = o.lower()
                    if o in c.lower():
                        count = 1
                        break
                else:
                    if o in c:
                        count = 1
                        break
            if count == 1:
                break
        
        n_correct_substr += count         
        #n_correct_substr += int(any([o.lower() in c.lower() for o in obj_labels for c in gpt_choices]))
        #n_correct_substr += int(any([is_string_in(o.lower(), c.lower().replace(".", " ").replace(",", " ")) for o in obj_labels for c in gpt_choices]))
    return {
        "Total number of tuples": n_tuples,
        "# of correct answers (Exact match)": n_correct,
        "% of correct answers (Exact match)": round(100 * n_correct / n_tuples, 1),
        "# of correct answers (Overlap)": n_correct_substr,
        "% of correct answers (Overlap)": round(100 * n_correct_substr / n_tuples, 1),
    }

def main():
    #df_list = []
    for model_name in BLOOM_MODEL:
        # load the model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_name == "bloomz-7b1":
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", offload_folder="offload")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        
        # initialise the data frame
        model_name_list = [model_name]
        file_name_list = []
        total_list = []

        df_info = {}

        lang = ["en-zh", "zh", "en-es", "en-ar"]
        for i in lang:
            info = {}
            info["lang_list"] = []
            info["N_c_e"] = []
            info["P_c_e"] = []
            info["N_c_o"] = []
            info["P_c_o"] = []
            info["time_cost"] = []

            df_info[i] = info
        
        # ------------------  do the probe -----------------------------------------------------------------------------
        for predicate in PRETICATES:
            start = time.perf_counter()
            print("current perticate is " + predicate)
            print("current model is " + model_name)

            # find all file paths of datasets
            file_paths = {}
            file_names = []
            for i in lang:
                if i == "en-zh" or i == "zh":
                    file_names = [predicate+ "_general_ASIA.jsonl", predicate+"_general_WESTERN_COUNTRIES.jsonl"]
                elif i == "en-es":
                    file_names = [predicate + "_general_SOUTH_AMERICA.jsonl", predicate+"_general_WESTERN_COUNTRIES.jsonl"]
                elif i == "en-ar":
                    file_names = [predicate + "_general_ARAB_REGION.jsonl", predicate+"_general_WESTERN_COUNTRIES.jsonl"]     
                
                file_path_temp = {}
                for j in file_names:
                    file_path_temp[j] = "dataset/dlama-v1/"+ i + "/"+ j
                file_paths[i] = file_path_temp

            # do probe
            for i in lang:
                output_triple = []

                
                if i == "en-zh" or i == "zh":
                    file_names = [predicate+ "_general_ASIA.jsonl", predicate+"_general_WESTERN_COUNTRIES.jsonl"]
                elif i == "en-es":
                    file_names = [predicate + "_general_SOUTH_AMERICA.jsonl", predicate+"_general_WESTERN_COUNTRIES.jsonl"]
                elif i == "en-ar":
                    file_names = [predicate + "_general_ARAB_REGION.jsonl", predicate+"_general_WESTERN_COUNTRIES.jsonl"]    

                for j in file_names:
                    output_triple = probe_bloom_using_DLAMA_jsonl(predicate, i, file_paths[i][j], model, tokenizer)

                    if i == "zh":
                        save_jsonl_file_no_EN("all_result/bloom/"+ model_name + "/" + i + "/" + model_name + "_"+ j.replace(".jsonl", "_output.jsonl"), output_triple)
                    else:
                        save_jsonl_file("all_result/bloom/"+ model_name + "/" + i + "/" + model_name + "_"+ j.replace(".jsonl", "_output.jsonl"), output_triple)

                    # store the dataframe data needed
                    current_file_name= j.replace(".jsonl", "")
                    current_result_list = list(compute_accuracy(output_triple).values())
                    if i == lang[0]:
                        file_name_list.append(current_file_name)
                        total_list.append(current_result_list[0])
                    
                    end = time.perf_counter()
                    elapsed = end - start
                    df_info[i]["lang_list"].append(i)
                    df_info[i]["N_c_e"].append(current_result_list[1])
                    df_info[i]["P_c_e"].append(current_result_list[2])
                    df_info[i]["N_c_o"].append(current_result_list[3])
                    df_info[i]["P_c_o"].append(current_result_list[4])
                    df_info[i]["time_cost"].append(elapsed)
                    
                    # print result
                    print(i + " " + j + ": ")
                    print(compute_accuracy(output_triple))
                    #save_jsonl_file_no_EN("result/"+ i + "/" + model_name + "_" + j.replace(".jsonl", "_result"), compute_accuracy(output_triple))

                    # create excel for df
                    columns = ['model name','file name','tuples numbers']
                    data = [model_name_list,file_name_list, total_list]
                    for ii in lang:
                        columns.extend(['language','e', 'e%', 'o', 'o%', "time cost"])
                        data.append(df_info[ii]["lang_list"])
                        data.append(df_info[ii]["N_c_e"])
                        data.append(df_info[ii]["P_c_e"])
                        data.append(df_info[ii]["N_c_o"])
                        data.append(df_info[ii]["P_c_o"])
                        data.append(df_info[ii]["time_cost"])
                    data = tuple(data)
                        
                    df = pd.DataFrame(data,index=columns).T
                    df.to_excel("all_result/result_table/"+ model_name + ".xlsx")


if __name__ == "__main__":

    main()
        
        

