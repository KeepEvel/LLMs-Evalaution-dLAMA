import re
import json


def load_jsonl_file(file_path):
    """Load a jsonl file."""
    data = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl_file(file_path, tuples):
    """Save tuples as a jsonl file."""
    with open(file_path, "w") as f:
        for t in tuples:
            f.write(json.dumps(t) + "\n")
            
def save_jsonl_file_no_EN(file_path, tuples):
    """Save tuples as a jsonl file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for t in tuples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def get_answer_only(inputs, outputs):
    
    new_outputs = outputs.replace(inputs, "")
    
    end_to_remove = "</s>"
    if end_to_remove in new_outputs:
        new_outputs = new_outputs.replace(end_to_remove, "")
    if len(new_outputs)>0:
        if new_outputs[0] == " ":
            return new_outputs[1:]
 
    return new_outputs

def get_key_answers(outputs):
    result_temp = outputs
    pattern = re.compile(r'[\u4e00-\u9fff]+')
    if bool(pattern.search(outputs)):       
        if "." in outputs:
            result_temp = result_temp.replace(".", "。")
        if " " in outputs:
            result_temp = result_temp.replace(" ", "。")
        if "、" in outputs:
            result_temp = result_temp.replace("、", ",")
        result = result_temp.split("。")[0]
    else:
        result = result_temp.split(".")[0]
    return result