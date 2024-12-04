import openai
import os
from typing import List, Tuple
import json

class LLMTopicModel:
    def __init__(self, model="gpt-3.5-turbo-0125", num_topics=10, num_top_words=5, prompt_style="description"):
        self.model = model
        self.num_topics = num_topics
        self.num_top_words = num_top_words
        self.prompt_style = prompt_style
        self.client = openai.OpenAI(organization=os.environ["OPENAI_ORG_ID"])
        self.top_words_list = None
        self.num_trials = 10
        self.log_jsonlist = []
        self.has_adequate_output = False

    def set_prompt(self, documents_file_path, insertion_phrase=None):
        with open(documents_file_path) as f:
            documents = f.read().strip()

        if self.prompt_style == "simple":
            prompt_lines = \
            ['Write the results of simulating topic modeling for the following documents, each starting with "#."',
            'Assume you will identify [NUM_TOPICS] topics and use [NUM_TOP_WORDS] top words for each topic.',
            'NOTE: Outputs must always be in the format "Topic k: word word word word word" and nothing else.',
            '"""',
            '[DOCUMENTS]',
            '"""']

        if self.prompt_style == "simulation":
            prompt_lines = \
            ["Write the results of simulating topic modeling for the following documents.",
            "Assume you will identify [NUM_TOPICS] topics and use [NUM_TOP_WORDS] top words for each topic.",
            "Outputs must always be in the format 'Topic k: word word word word word' and nothing else.",
            "[INSERTION_PHRASE]",
            "###",
            "[DOCUMENTS]",
            "###"]
        
        elif self.prompt_style == "description":
            prompt_lines = \
            ["Discover latent [NUM_TOPICS] topics in the following documents.",
            "For each topic, write [NUM_TOP_WORDS] words extracted from input texts to show its meanings.",
            "Outputs must always be in the format 'Topic k: word word word word word' and nothing else.",
            "[INSERTION_PHRASE]",
            "###",
            "[DOCUMENTS]",
            "###"]
        
        elif self.prompt_style == "no_limited_k":
            prompt_lines = \
            ["Discover latent topics in the following documents. The number of topics will be adjusted as needed.", 
            "For each topic, write [NUM_TOP_WORDS] words extracted from input texts to show its meanings.",
            "Outputs must always be in the format 'Topic k: word word word word word' and nothing else.",
            "[INSERTION_PHRASE]",
            "###",
            "[DOCUMENTS]",
            "###"]

        elif self.prompt_style == "reverse":
            prompt_lines = \
            ["###",
            "[DOCUMENTS]",
            "###",
            "Discover latent [NUM_TOPICS] topics in the above documents.",
            "For each topic, write [NUM_TOP_WORDS] words extracted from input texts to show its meanings.",
            "Outputs must always be in the format 'Topic k: word word word word word' and nothing else.",
            "[INSERTION_PHRASE]",
            ]

        prompt_template = "\n".join(prompt_lines)
        if insertion_phrase:
            prompt_template = prompt_template.replace("[INSERTION_PHRASE]", insertion_phrase)
        else:
            prompt_template = prompt_template.replace("[INSERTION_PHRASE]", "")
        self.prompt = prompt_template.replace("[DOCUMENTS]", documents).replace("[NUM_TOPICS]",str(self.num_topics)).replace("[NUM_TOP_WORDS]",str(self.num_top_words))

    def extract_topics(self, output:str) -> Tuple[List[List[str]], str]:
        try:
            lines_startswith_topic = [line for line in output.split("\n") if "Topic" in line]       #Topic を含むを抽出
            top_words_lines = [line.split(": ")[1].rstrip()  for line in lines_startswith_topic]    #": "以後を抽出, 末尾の改行記号を除去
            top_words_lines = [str.join(" ", map(lambda word: word.strip(","), top_words_line.split(" "))) for top_words_line in top_words_lines]  #","区切りだった場合に","を除去する"
            #topics = [top_words_line.split(" ")[:self.num_top_words] for top_words_line in top_words_lines] #トップワード数を揃える, トップワード数が不足している場合もエラー
            topics = [top_words_line.split(" ") for top_words_line in top_words_lines]

            if(self.num_topics):
                if(len(topics) != self.num_topics):
                    print(f"-> Not adequate number of topics")
                    return None, "num_topics"
            if(self.num_top_words):
                for top_words in topics:
                    if len(top_words) != self.num_top_words:
                        print(f"-> Not adequate number of top words")
                        return None, "num_top_words"
        except Exception:   #パースできなかった場合
            return None, "format"

        return topics, "None"

    def run(self):
        for i_trial in range(self.num_trials):
            print("Trial: ", i_trial)
            completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": self.prompt}
            ]
            )
            output = completion.choices[0].message.content
            topics, error_str = self.extract_topics(output)
            self.log_jsonlist.append({"trial":i_trial, "output":output, "error":error_str})
            if topics:
                self.has_adequate_output = True
                self.topics = topics
                break
            
            
        if not self.has_adequate_output:
            self.topics = None
            print("Could not obtain any adequate output")

    def save(self, outputs_dir_path):
        os.makedirs(outputs_dir_path, exist_ok=True)
        prompt_file_path = os.path.join(outputs_dir_path, "prompt.txt")
        with open(prompt_file_path, "w") as f:
            f.write(self.prompt)
        print(f"save the prompt at {prompt_file_path}")

        #output_logs_dir_path = os.path.join(outputs_dir_path, "output_logs")
        #os.makedirs(output_logs_dir_path, exist_ok=True)
        #for i, row_output in enumerate(self.output_logs):
        #    with open(os.path.join(output_logs_dir_path, f"{i}th_row_output.txt"), "w") as f:
        #        f.write(row_output)
        logs_file_path = os.path.join(outputs_dir_path, "log.jsonlist")
        with open(logs_file_path,"w") as f:
            for log in self.log_jsonlist:
                #print(log)
                f.write(json.dumps(log) + "\n")
        print(f"save logs at {prompt_file_path}")

        top_words_file_path = os.path.join(outputs_dir_path, "top_words.txt")
        with open(top_words_file_path, "w") as f:
            if self.has_adequate_output:
                for top_words in self.topics:
                    top_words_line = " ".join(top_words)
                    f.write(top_words_line.strip() + "\n")
            else:
                f.write("NO OUTPUT\n")
        print(f"save topics at {top_words_file_path}")

        