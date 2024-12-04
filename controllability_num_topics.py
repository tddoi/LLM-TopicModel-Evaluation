import os
from experiment import run, eval, generate_outputs_dir_path
from evaluation import CoverageEvaluator, FactualityEvaluator

def main():
    experiment_title = "2024-06-11-1717-Ctrl-NumTopics"
    model = "gpt-4-turbo-2024-04-09"
    num_top_words = 5
    dataset = "GoogleNewsT"
    documents_file_path = os.path.join("datasets", dataset, "train_texts.txt")
    #documents_file_path = "datasets/GoogleNewsT/seed1/Part01/train_texts.txt"
    prompt_style = "reverse"

    for num_topics in [50]:
        outputs_dir_path = generate_outputs_dir_path(experiment_title, dataset, f"{model}", num_topics)
        run(model, num_topics, num_top_words, documents_file_path, prompt_style, None, outputs_dir_path)

if __name__ == "__main__":
    main()