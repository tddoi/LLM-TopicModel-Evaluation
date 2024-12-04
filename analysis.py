import os
import json
import argparse
import numpy as np
import pandas as pd
import utils
from collections import defaultdict
import glob

#metrics_list = ["Mean_Conc", "Max_Conc", "Min_Conc"]
#metrics_list = ["Cov_in_datasets/20NG/train_texts.txt","Cov_in_datasets/20NG_talk.politics/train_texts.txt", "Cov_in_datasets/20NG_comp/train_texts.txt",  "Cov_in_datasets/20NG_sci/train_texts.txt",  "Cov_in_datasets/20NG_rec.sport/train_texts.txt"]

def extract_metrics(outputs_dir_path):
    score_file_paths = glob.glob(os.path.join(outputs_dir_path,"**","scores.json"), recursive=True)
    if score_file_paths:
        with open(score_file_paths[0]) as f: score_dict = json.load(f)
        return score_dict.keys()
    else:
        return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o","--outputs_dir_path", type=str)
    parser.add_argument("--results_file_path", type=str, default=None)
    args = parser.parse_args()
    if not args.results_file_path:
        experiment_title = args.outputs_dir_path.split(os.sep)[1]
        dataset = args.outputs_dir_path.split(os.sep)[2]
        args.results_file_path = os.path.join("results", experiment_title, dataset, "results.tsv")
    return args

def main():
    args = parse_args()
    model_scores_dict = defaultdict(lambda: dict())
    for metric in extract_metrics(args.outputs_dir_path):
        for model_dir_name in os.listdir(args.outputs_dir_path):
            scores = []
            for seed_dir_name in os.listdir(os.path.join(args.outputs_dir_path, model_dir_name)):
                with open(os.path.join(args.outputs_dir_path, model_dir_name, seed_dir_name, "scores.json")) as f:
                    score_dict = json.load(f)
                scores.append(score_dict[metric])
            model_scores_dict[f"{metric}_Mean"][model_dir_name] = utils.mean(scores)
            model_scores_dict[f"{metric}_Std"][model_dir_name] = utils.std(scores)
    
    os.makedirs(os.path.dirname(args.results_file_path), exist_ok=True)
    pd.DataFrame.from_dict(model_scores_dict).to_csv(args.results_file_path, sep="\t")

if __name__ == "__main__":
    main()