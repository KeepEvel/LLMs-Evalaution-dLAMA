import re
import time
import json
import csv
import unicodedata
import torch
import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from file_save import *
from llama import Llama

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'

torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)


Llama2_MODEL = ["llama-2-7b-chat"]
ckpt_dir = "llama-2-7b-chat/"
tokenizer_path = "tokenizer.model"
temperature = torch.tensor(0)
top_p = torch.tensor(0.9)
max_seq_len = torch.tensor(512)
max_batch_size = torch.tensor(4)
max_gen_len = None


SYSTEM_DESCRIPTION = {"zh": "你是一个百科全书", 'en': "You are an encyclopedia."}

PRETICATES = ["P1412", "P1376", "P1303", "P530", "P495", "P449", "P364", "P264", "P190", "P136", "P106", "P103", "P47", "P37", "P36", "P30", "P27", "P20", "P19", "P17"]

PROMPTS = {
    "P1412": {
        "zh": '{}使用什么语言? 只用一个语言的名字来回复，用中文。',
        'en': 'What language does {} used to communicate in? Reply with a language name only, no other words.',
    },
    "P1376": {
        "zh": '哪个国家的首都是{}? 只回复一个国家的中文中文名称。',
        'en': 'what is the country of which the captital is {}? Reply only to the name of a country, no other words.',
    },
    "P1303": {
        "zh": '{}演奏什么乐器?只回复一个乐器的中文名称。',
        'en': 'What instrument does {} play? Reply with only one instrument name, no other words, no other words.',
    },
    "P530": {
        "zh": '哪个国家与{}保持外交关系? 仅回复一个国家的中文名字',
        'en': 'Which country maintains diplomatic relations with {}? Reply with a name of the country only, no other words.',
    },
    "P495": {
        "zh": '哪一个国家创造了{}? 仅回复一个国家的中文名字',
        'en': 'Which country that creats the {}? Reply to only one country name, no other word.',
    },
    "P449": {
        "zh": '{}最初是在什么地方播出的? 仅仅回复一个播出地点的中文名字',
        'en': 'What does {} originally aired on?  Reply with one full name only, no other word.',
    },
    "P364": {
        "zh": '{}的起源语言是什么?仅回复一个语言的中文名字',
        'en': 'What is the language of origin of {}? Reply to the name in one language only, no other word.',
    },
    "P264": {
        "zh": '{}和哪个唱片公司签约了? 仅回复一个唱片公司的中文名字',
        'en': 'Which record label is {} signed to? Answer one record label name only, no other word.',
    },
     "P190": {
        "zh": '哪个城市是{}的姐妹城市? 仅回复一个城市的中文名字',
        'en': 'Which city is the twin city of {}? Reply with a city name only, no other word.',
    },
     "P136": {
        "zh": '{}与哪种音乐流派有关？仅回复音乐类型的中文名字',
        'en': 'Which musical genre is {} associated with? Reply only to the name of the music genre, no other word.',
    },
    
    "P17": {
        "zh": '{}位于哪个国家？仅回答一个国家的中文名字',
        'en': ' In which country is {} located? Reply one name of country only, no other word.',
    },
    "P19":{
        "zh": '{}出生于哪个城市？仅回答一个城市的中文名字',
        'en': ' In which city was {} born? Reply one name of city only, no other word.',
    },
    
    "P20":{
        "zh": '{}在哪个城市去世？仅回答一个城市的中文名字' ,
        'en':' In which city did {} die? Reply one name of city only, no other word.',
    },
    "P27":{
        "zh": '{}是哪个国家的公民？仅回答一个国家的中文名字' ,
        'en':' What country is {} a citizen of? Reply one name of country only, no other word.',
    },
    "P30": {
        "zh": '{}位于哪个大洲？仅回答一个大洲的中文名字' ,
        'en': ' Which continent is {} located in? Reply one name of continent only, no other word.',
    },
    "P36": {
        "zh": '{}的首都是？仅回答一个城市的中文名字' ,
        'en': ' What is the capital of {}? Reply one name of city only, no other word.',
    },
    "P37": {
        "zh": '{}的官方语言是什么？ 仅回答一个语言的中文名字' ,
        'en': ' What is the official language of {}? Reply one name of language only, no other word. ',
    },
    "P47": {
        "zh": '{}与哪个国家接壤？仅回答一个国家的中文名字' ,
        'en': 'What is the country that shares border with {}? Reply one name of country only, no other word.',
    },
    "P103":{
        "zh": '{}的母语是什么？仅回答一个语言的中文名字' ,
        'en':' What is the native language of {}? Reply one name of language only, no other word.',
    },
    "P106":{
        "zh": '{}的职业是什么？仅回答一个职业的中文名字' ,
        'en':' What is the profession of {}? Reply one name of profession only, no other word.',
    }

}


def get_Llama2_output(predicate, entity, lang, generator): # one question per call
    """Answer a question using Llama-v2 model by GPU."""
    
    # Compose the prompt
    sys_dict = {"role": "system"}
    sys_dict["content"] = SYSTEM_DESCRIPTION[lang]
    
    user_dict = {"role": "user"}
    user_dict["content"] = PROMPTS[predicate][lang].format(entity)
    prompt = [[sys_dict, user_dict]]
      
    # Get result
    outputs = generator.chat_completion(
        prompt,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    
    result = outputs[0]['generation']['content']
    return [result]


def probe_Llama2_using_DLAMA_jsonl(predicate, lang, input_file_path, generator):
    """Probe Llama2 using a DLAMA predicate and save the tuples with the answers to a jsonl file."""
    output_triples = []

    lang1 = lang
    if lang == "en-ar" or lang == "en-es" or lang=="en-zh":
        lang1 = "en"
    for triple in tqdm(load_jsonl_file(input_file_path)):
        # Handle rate-limiting in a naive way!
        while True:
            try:
                choices = get_Llama2_output(predicate, triple["sub_label"], lang1, generator)
                break
            except:
                print("Rate-limited")
                # TODO: Tune the time to wait before sending another request
                time.sleep(5)

        triple["Llama2_choices"] = choices.strip()
        
        if lang1 == "en":
            triple["Llama2_choices_pure"] = [re.sub(r"[^a-zA-Z]", "", n) for n in choices]
        elif lang1 == "zh":
            triple["Llama2_choices_pure"] = [re.sub(r"[^\u4e00-\u9fff]+", "", n) for n in choices]
        
        output_triples.append(triple)

            
    return output_triples

# the original method in dLAMA
# def compute_accuracy(tuples):
#     """Measure the BLOOM model's accuracy."""
#     n_correct = int(0)
#     n_correct_substr = int(0)
#     n_tuples = len(tuples)
#     for t in tuples:
#         Llama_choices = t["bloom_choices"]
#         obj_labels = t["obj_label"]

#         # An answer is correct if it is one of candidate answers
#         n_correct += int(any([c in obj_labels for c in Llama_choices]))

#         # An answer is correct if any of the candidate answers is a substr of the answer
#         n_correct_substr += int(any([o in c for o in obj_labels for c in Llama_choices]))

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
        Llama_choices = t["Llama2_choices"]
        for item in range(len(Llama_choices)):
            Llama_choices[item] = Llama_choices[item].strip()
            
        obj_labels = t["obj_label"]

        # An answer is correct if it is one of candidate answers
        n_correct += int(any([c in obj_labels for c in Llama_choices]))

        # An answer is correct if any of the candidate answers is a substr of the answer
        count = 0
        
        for c in Llama_choices:
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
        #n_correct_substr += int(any([o.lower() in c.lower() for o in obj_labels for c in Llama_choices]))
        #n_correct_substr += int(any([is_string_in(o.lower(), c.lower().replace(".", " ").replace(",", " ")) for o in obj_labels for c in Llama_choices]))
    return {
        "Total number of tuples": n_tuples,
        "# of correct answers (Exact match)": n_correct,
        "% of correct answers (Exact match)": round(100 * n_correct / n_tuples, 1),
        "# of correct answers (Overlap)": n_correct_substr,
        "% of correct answers (Overlap)": round(100 * n_correct_substr / n_tuples, 1),
    }

def compute_accuracy(tuples):
    """Measure the Llama2 model's accuracy."""
    n_correct = int(0)
    n_correct_substr = int(0)
    n_tuples = len(tuples)
    for t in tuples:
        Llama_choices = t["Llama2_choices"]
        obj_labels = t["obj_label"]

        # An answer is correct if it is one of candidate answers
        n_correct += int(any([c in obj_labels for c in Llama_choices]))

        # An answer is correct if any of the candidate answers is a substr of the answer
        n_correct_substr += int(any([o in c for o in obj_labels for c in Llama_choices]))

    return {
        "Total number of tuples": n_tuples,
        "# of correct answers (Exact match)": n_correct,
        "% of correct answers (Exact match)": round(100 * n_correct / n_tuples, 1),
        "# of correct answers (Overlap)": n_correct_substr,
        "% of correct answers (Overlap)": round(100 * n_correct_substr / n_tuples, 1),
    }
def main():
    #df_list = []
    for model_name in Llama2_MODEL:
        # load the model
        ckpt_dir = model_name + "/"
        generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
         )

        # initialise the data frame
        model_name_list = [model_name]
        file_name_list = []
        total_list = []

        df_info = {}

        #lang = ["en-zh", "zh", "en-es", "en-ar"]
        lang = ["en-zh", "en-es", "en-ar"]
        for i in lang:
            info = {}
            info["lang_list"] = []
            info["N_c_e"] = []
            info["P_c_e"] = []
            info["N_c_o"] = []
            info["P_c_o"] = []
            info["time_cost"] = []

            df_info[i] = info
        
        for predicate in PRETICATES:
            start = time.perf_counter()
            print("current perticate is " + predicate)
            print("current model is " + model_name)

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
          
            for i in lang:
                output_triple = []

                if i == "en-zh" or i == "zh":
                    file_names = [predicate+ "_general_ASIA.jsonl", predicate+"_general_WESTERN_COUNTRIES.jsonl"]
                elif i == "en-es":
                    file_names = [predicate + "_general_SOUTH_AMERICA.jsonl", predicate+"_general_WESTERN_COUNTRIES.jsonl"]
                elif i == "en-ar":
                    file_names = [predicate + "_general_ARAB_REGION.jsonl", predicate+"_general_WESTERN_COUNTRIES.jsonl"]    

                for j in file_names:
                    output_triple = probe_Llama2_using_DLAMA_jsonl(predicate, i, file_paths[i][j], generator)

                    if i == "zh":
                        save_jsonl_file_no_EN("all_result/llama2-7b-chat/"+ i + "/" + model_name + "_"+ j.replace(".jsonl", "_output.jsonl"), output_triple)
                    else:
                        save_jsonl_file("all_result/llama2-7b-chat/"+ i + "/" + model_name + "_"+ j.replace(".jsonl", "_output.jsonl"), output_triple)

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
        
        

