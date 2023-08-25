#  Translate a DLAMA dataset into another language
import copy
import glob
import json
from tqdm import tqdm
from utils import get_wikidata_labels
from os import makedirs

from pathlib import Path
from argparse import ArgumentParser


def load_file(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = [json.loads(l) for l in f]
    return data


import pywikibot

def get_aliases(site, labels, lang):
    repo = site.data_repository()  # the Wikibase repository for given site
    alian_dict = {}
    for label in set(labels):
        item = pywikibot.ItemPage(repo, label)  # a repository item
        if lang in item.aliases:
            alian_dict[label] = item.aliases[lang]
        else:
            alian_dict[label] = []
    return alian_dict
            
        

    
def main(BASE_DIR, other_lang):
    """Translate data files of DLAMA into a new language"""
    
    site = pywikibot.Site('wikipedia:es')
    makedirs(str(Path(BASE_DIR, other_lang)), exist_ok=True)
    files = sorted([f for f in glob.glob(str(Path(BASE_DIR, other_lang, "*")))])

    for file in tqdm(files):
        samples = load_file(file)
        #subjects_uris = [s["sub_uri"] for s in samples]
        objects_uris = [obj_uri for s in samples for obj_uri in s["obj_uri"]]

        # Find translation dictionaries
        #subjects_labels = get_wikidata_labels(subjects_uris, [other_lang])
        
        alian_dict = get_aliases(site, objects_uris, other_lang)

        # Translate samples
        translated_samples = []
        for sample in samples:
            obj_uris = sample["obj_uri"]
            translated_sample = copy.deepcopy(sample)
            for i in translated_sample["obj_uri"]:
                translated_sample["obj_label"] += alian_dict.get(i, [])
            translated_sample["obj_label"] = list(set(translated_sample["obj_label"]))
            translated_samples.append(translated_sample)
        # Export translated file
        filename = file.split("\\")[-1]
        output_filename = str(Path(BASE_DIR, 'extended', other_lang, filename))
        makedirs(str(Path(BASE_DIR, 'extended', other_lang)), exist_ok=True)
        with open(output_filename, "w", encoding="utf-8") as f:
            for sample in translated_samples:
                f.write(json.dumps(sample, ensure_ascii=False))
                f.write("\n")


if __name__ == "__main__":
    args_parser = ArgumentParser("Translate the labels of a dataset to a new langauges")
    args_parser.add_argument(
        "--lang", required=True, help="New language to translate labels to"
    )
    args_parser.add_argument(
        "--dir", required=True, help="Base directory of the dataset"
    )
    args = args_parser.parse_args()

    main(BASE_DIR=args.dir, other_lang=args.lang)
 #python translate_dlama_dataset.py --lang "zh" --dir ../data/asia-west/dlama/
