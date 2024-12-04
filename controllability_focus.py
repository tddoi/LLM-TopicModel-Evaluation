import os
from experiment import run, eval, generate_outputs_dir_path
from evaluation import CoverageEvaluator, FactualityEvaluator

def main():
    experiment_title = "2024-06-11-1717-Ctrl-Focus"
    model = "gpt-4-turbo-2024-04-09"
    num_topics = 5
    num_top_words = 5
    documents_file_path = os.path.join("datasets", "20NG", "train_texts.txt")
    prompt_style = "reverse"

    for iof_run in range(2,6):
        #for focused_topic in ["computer","science","sports","politics"]:
        for focused_topic in ["computer","science","politics"]:
            insertion_phrase = f"Identify topics specifically related to {focused_topic}."
            outputs_dir_path = generate_outputs_dir_path(experiment_title, "20NG", f"{model}_focusing_{focused_topic}", iof_run)
            run(model, num_topics, num_top_words, documents_file_path, prompt_style, insertion_phrase, outputs_dir_path)
            #eval([CoverageEvaluator(os.path.join("datasets", "20NG_talk.politics", "train_texts.txt"))], outputs_dir_path)
            eval([
                    CoverageEvaluator(os.path.join("datasets", "20NG", "train_texts.txt")),
                    CoverageEvaluator(os.path.join("datasets", "20NG_talk.politics", "train_texts.txt")),
                    CoverageEvaluator(os.path.join("datasets", "20NG_comp", "train_texts.txt")),
                    CoverageEvaluator(os.path.join("datasets", "20NG_sci", "train_texts.txt")),
                    CoverageEvaluator(os.path.join("datasets", "20NG_rec.sport", "train_texts.txt")),
                    FactualityEvaluator(os.path.join("datasets", "20NG", "vocab.txt"))
                ],
                outputs_dir_path)
        
        outputs_dir_path = generate_outputs_dir_path(experiment_title, "20NG", f"{model}_default", iof_run)
        run(model, num_topics, num_top_words, documents_file_path, prompt_style, None, outputs_dir_path)
        eval([
                    CoverageEvaluator(os.path.join("datasets", "20NG", "train_texts.txt")),
                    CoverageEvaluator(os.path.join("datasets", "20NG_talk.politics", "train_texts.txt")),
                    CoverageEvaluator(os.path.join("datasets", "20NG_comp", "train_texts.txt")),
                    CoverageEvaluator(os.path.join("datasets", "20NG_sci", "train_texts.txt")),
                    CoverageEvaluator(os.path.join("datasets", "20NG_rec.sport", "train_texts.txt")),
                    FactualityEvaluator(os.path.join("datasets", "20NG", "vocab.txt"))
                ],
                outputs_dir_path)
        
        for oracle_topic in ["comp","sci","talk.politics"]:
            documents_file_path = os.path.join("datasets", f"20NG_{oracle_topic}", "train_texts.txt")
            outputs_dir_path = generate_outputs_dir_path(experiment_title, "20NG", f"{model}_oracle_{oracle_topic}", iof_run)
            run(model, num_topics, num_top_words, documents_file_path, prompt_style, None, outputs_dir_path)
            eval([
                        CoverageEvaluator(os.path.join("datasets", "20NG", "train_texts.txt")),
                        CoverageEvaluator(os.path.join("datasets", "20NG_talk.politics", "train_texts.txt")),
                        CoverageEvaluator(os.path.join("datasets", "20NG_comp", "train_texts.txt")),
                        CoverageEvaluator(os.path.join("datasets", "20NG_sci", "train_texts.txt")),
                        CoverageEvaluator(os.path.join("datasets", "20NG_rec.sport", "train_texts.txt")),
                        FactualityEvaluator(os.path.join("datasets", "20NG", "vocab.txt"))
                    ],
                    outputs_dir_path)

if __name__ == "__main__":
    main()