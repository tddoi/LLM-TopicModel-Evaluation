from llm_topic_model import LLMTopicModel
from evaluation import EvaluationSystem, Evaluator
import argparse
import os
import yaml
import json
from typing import List

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, default="test")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0125")
    parser.add_argument("-d", "--documents_file_path", type=str, default="datasets/GoogleNewsT/seed1/Part01/train_texts.txt")
    parser.add_argument("-k", "--num_topics", type=int, default=None)
    parser.add_argument("-t", "--num_top_words", type=int, default=None)
    parser.add_argument("--prompt_style", type=str, default="description", choices=["description","simulation","reverse"])
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--eval", action="store_true")
    #parser.add_argument("-i", "--trial_index", type=int, default=None)
    args = parser.parse_args()
    return args

def generate_outputs_dir_path(experiment_title, dataset_name, model_name, nof_run=1):
    return os.path.join("outputs", experiment_title, dataset_name, model_name, str(nof_run))
    
def run(model, num_topics, num_top_words, documents_file_path, prompt_style, insertion_phrase, outputs_dir_path):
    llm_topic_model = LLMTopicModel(model=model, num_topics=num_topics, num_top_words=num_top_words, prompt_style=prompt_style)
    llm_topic_model.set_prompt(documents_file_path, insertion_phrase)
    llm_topic_model.run()
    llm_topic_model.save(outputs_dir_path)
    return llm_topic_model.topics

def eval(evaluators:List[Evaluator], outputs_dir_path):
    evaluation_system = EvaluationSystem(evaluators)
    topics = evaluation_system.read_topics(outputs_dir_path)
    evaluation_system.eval(topics)
    evaluation_system.save(outputs_dir_path)
    return evaluation_system.scores


def main():
    args = parse_argument()
    
    insertion_phrase_dict = \
    {
        "none": None,
        "broad":    "NOTE: Identify broad topics.",
        "narrow":   "NOTE: Identify narrow topics.",
        "broad_abst":   "NOTE: Use abstract words to identify broad topics.",
        "narrow_spc":   "NOTE: Use specific words to identify narrow topics.",
        "broad_demo":   "NOTE:\nBroad Topic: processor computer software hardware interface, Narrow Topic: card monitor screen driver vga\nPlease learn from the examples above, and identify broad topics.",   #cited from [Duan et al., 2023] p15, 6th and 240th topics
        "narrow_demo":  "NOTE:\nBroad Topic: processor computer software hardware interface, Narrow Topic: card monitor screen driver vga\nPlease learn from the examples above, and identify narrow topics.",
        "broad_demo_abst":   "NOTE:\nBroad Topic: processor computer software hardware interface, Narrow Topic: card monitor screen driver vga\nPlease learn from the examples above, and use abstract words to identify broad topics.",
        "narrow_demo_spc":  "NOTE:\nBroad Topic: processor computer software hardware interface, Narrow Topic: card monitor screen driver vga\nPlease learn from the examples above, and use specific words to identify narrow topics.",
    }


    for phrase_title in insertion_phrase_dict:
        args.phrase_title = phrase_title
        args.insertion_phrase = insertion_phrase_dict[phrase_title]
        #for trial_index in range(1,args.num_runs+1):
            #args.trial_index = trial_index
            #args.data_file_path = f"datasets/GoogleNewsT/seed1/Part{args.trial_index:02d}/train_texts.txt"
        dataset_name = args.documents_file_path.split(os.sep)[1]
        model_name = f"{args.model}_no_limited_k_{phrase_title}"
        args.outputs_dir_path = generate_outputs_dir_path(args.title, dataset_name, model_name, iof_run=1)
        os.makedirs(args.outputs_dir_path, exist_ok=True)

        print("-"*70)
        print(yaml.dump(vars(args), default_flow_style=False))
        if args.run:
            print("RUN")
            run(model=args.model, 
                num_topics=args.num_topics, 
                num_top_words=args.num_top_words, 
                documents_file_path=args.documents_file_path, 
                prompt_style="reverse",
                insertion_phrase=args.insertion_phrase,
                outputs_dir_path=args.outputs_dir_path)
        if args.eval:
            print("EVALUATE")
            eval(["MeanConc", "MaxConc", "MinConc"], args.outputs_dir_path)
        print("-"*70)

if __name__ == "__main__":
    main()