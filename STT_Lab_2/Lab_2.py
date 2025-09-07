import pandas as pd
import csv
from pydriller import Repository
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

import re
import nltk
from transformers import pipeline
from nltk import pos_tag, word_tokenize
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# load summarizer once
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Load models once
similarity_model = SentenceTransformer("all-mpnet-base-v2")
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Imperative verbs for reference
IMPERATIVE_VERBS = [
    "fix", "add", "remove", "update", "refactor", "implement", "change"
]

# Generic placeholders to remove
GENERIC_PHRASES = [
    "fix bug", "bug fix", "error fixed", "issue resolved"
]

def rectify_commit_message(dev_msg, llm_msg):

    dev_lines = [line.strip() for line in dev_msg.split("\n") if line.strip()]
    llm_lines = [line.strip() for line in llm_msg.split("\n") if line.strip()]

    # Combine lines that are unique or more informative
    combined_lines = []
    for d_line in dev_lines:
        # Skip if line is generic
        if any(phrase in d_line.lower() for phrase in GENERIC_PHRASES):
            continue

        # Check similarity to any LLM line
        if llm_lines:
            sims = util.cos_sim(similarity_model.encode([d_line]), similarity_model.encode(llm_lines))
            if sims.max() < 0.8:  # Keep dev line if not redundant
                combined_lines.append(d_line)
        else:
            combined_lines.append(d_line)

    # Add LLM lines not already in combined
    for l_line in llm_lines:
        if l_line not in combined_lines and not any(phrase in l_line.lower() for phrase in GENERIC_PHRASES):
            combined_lines.append(l_line)

    msg = " ".join(combined_lines)

    # Normalize verbs
    replacements = {
        "fixed": "fix", "fixes": "fix", "fixing": "fix",
        "added": "add", "adding": "add",
        "removed": "remove", "removing": "remove",
        "updated": "update", "updating": "update",
        "changed": "change", "changing": "change",
        "implemented": "implement", "implementing": "implement",
        "refactored": "refactor", "refactoring": "refactor"
    }
    msg = msg.lower()
    msg = re.sub(r"^(this commit|commit)\s*", "", msg)
    for k, v in replacements.items():
        msg = re.sub(rf"\b{k}\b", v, msg)

    # Extract first action verb
    tokens = nltk.word_tokenize(msg)
    pos_tags = nltk.pos_tag(tokens)
    verb_found = False
    for i, (word, tag) in enumerate(pos_tags):
        if tag.startswith("VB") and word in IMPERATIVE_VERBS:
            # Move verb to start if not already
            if i > 0:
                tokens.insert(0, tokens.pop(i))
            verb_found = True
            break
    if not verb_found and tokens:
        tokens.insert(0, "update")

    msg = " ".join(tokens)

    summary = msg.split("\n")[0][:50].rstrip()
    return summary



def is_precise_commit_message(msg):
    msg = msg.strip()

    # 1. Too short or too long
    words = msg.split()
    if len(words) < 3 or len(words) > 20:
        return False

    # 2. Action verb at start
    first_word = words[0].lower()
    imperative_verbs = ["fix", "add", "remove", "update", "refactor", "implement", "change"]
    if first_word not in imperative_verbs:
        return False

    # 3. Must contain at least one noun (component/location)
    tokens = word_tokenize(msg)
    pos_tags = pos_tag(tokens)
    nouns = [w for w, t in pos_tags if t.startswith("NN")]
    if len(nouns) == 0:
        return False

    # 4. Reject generic messages
    generic = ["fix bug", "bug fix", "error fixed", "issue resolved"]
    if msg.lower() in generic:
        return False

    return True

def compute_hit_rate(messages):
    hits = sum(is_precise_commit_message(m) for m in messages)
    return hits / len(messages) if messages else 0


def replace_comma(message):
    if message:
        return message.replace(',',' ')
    return message

def is_binary(filename):
    binary_exts = ('.exe', '.dll', '.so', '.bin', '.class', '.pyc',
                   '.jpg', '.jpeg', '.png', '.gif', '.pdf',
                   '.zip', '.tar', '.gz', '.7z')
    return filename.lower().endswith(binary_exts)



def main():
    keywords = ["fixed", "bug", "fixes ", "fix ", " fix", "fixed", " fixes", "crash", "solves", "resolves",
                 "resolves "," issue", "issue", "regression", "fall back", "assertion", "coverity", "reproducible",
                   "stack-wanted", "steps-wanted", "testcase", "failur", "fail", "npe ", " npe", "except", "broken",
                     "differential testing", "error", "hang hang", "test fix", "steps to reproduce", "crash",
                       "assertion", "failure", "leak", "stack trace", "heap overflow", "freez", "problem", "problem",
                         "overflow", "overflow ", "avoid ", " avoid", "workaround ", "workaround", "break", "break", "stop", "stop"]
    
    bug_fix_commit_dict = {"Hash": [], "Message": [], "Hash_Parent": [], "Is_Merge_Commit": [], "Modified_Files": []}
    diff_extraction_dict = {"Hash": [], "Message": [], "Filename": [], "Source_Code_Old": [], "Source_Code_New": [], "Diff": [],
                            "LLM_Inference": [], "Rectified_Message": []}
    
    #Loading the LLM model
    tokenizer = AutoTokenizer.from_pretrained("mamiksik/CommitPredictorT5")
    model = AutoModelForSeq2SeqLM.from_pretrained("mamiksik/CommitPredictorT5")

    max_input_len = 512
    max_output_len = 128
    

    for commit in Repository("https://github.com/1j01/textual-paint").traverse_commits():
        message = commit.msg
        if not any(keyword in message for keyword in keywords):
            continue

        modified_files = commit.modified_files

        # Adds info for the Bug Fixing Commit CSV (first CSV in the assignment)
        bug_fix_commit_dict["Hash"].append(replace_comma(commit.hash))
        bug_fix_commit_dict["Message"].append(replace_comma(message))
        bug_fix_commit_dict["Hash_Parent"].append(commit.parents)
        bug_fix_commit_dict["Is_Merge_Commit"].append(commit.merge)
        bug_fix_commit_dict["Modified_Files"].append(modified_files)

        # Populates data for the diff extraction csv (second CSV in the assignment)
        for file in modified_files:
            if is_binary(file.filename):
                continue

            diff = file.diff

            source_code_old = file.source_code_before
            source_code_new= file.source_code
            if file.source_code_before and len(file.source_code_before) > max_input_len:
                source_code_old= file.source_code_before[:max_input_len]
            if file.source_code and len(file.source_code) > max_input_len:
                source_code_new= file.source_code[:max_input_len]
            if diff and len(diff) > max_input_len:
                diff = diff[:max_input_len]
                

            diff_extraction_dict["Hash"].append(replace_comma(commit.hash))
            diff_extraction_dict["Message"].append(replace_comma(message))
            diff_extraction_dict["Filename"].append(replace_comma(file.filename))
            diff_extraction_dict["Source_Code_Old"].append(replace_comma(source_code_old))
            diff_extraction_dict["Source_Code_New"].append(replace_comma(source_code_new))
            diff_extraction_dict["Diff"].append(replace_comma(diff))

            # LLM Inference
            input_ids = tokenizer(diff, return_tensors="pt", max_length=max_input_len).input_ids
            outputs = model.generate(input_ids, max_length=max_output_len)
            LLM_output = tokenizer.decode(outputs[0], skip_special_tokens= True)

            diff_extraction_dict["LLM_Inference"].append(replace_comma(LLM_output))

            # Rectifier
            rectified_msg = rectify_commit_message(message, LLM_output)

            diff_extraction_dict["Rectified_Message"].append(replace_comma(rectified_msg))


    dev_hit_rate = compute_hit_rate(diff_extraction_dict["Message"])
    llm_hit_rate = compute_hit_rate(diff_extraction_dict["LLM_Inference"])
    rectifier_hit_rate = compute_hit_rate(diff_extraction_dict["Rectified_Message"])
    print(f"The developer hit rate is: {dev_hit_rate}")
    print(f"The LLM hit rate is: {llm_hit_rate}")
    print(f"The Rectifier hit rate is: {rectifier_hit_rate}")
        


    # Extract the data to CSVs
    bug_fix_commit_df = pd.DataFrame(bug_fix_commit_dict)
    diff_extraction_df = pd.DataFrame(diff_extraction_dict)

    bug_fix_commit_df[['Hash', "Message"]] = bug_fix_commit_df[['Hash', "Message"]].apply(lambda col: col.str.replace(r"\r?\n", " ", regex=True))
    diff_extraction_df = diff_extraction_df.apply(lambda col: col.str.replace(r"\r?\n", " ", regex=True))

    bug_fix_commit_df.to_csv(r"C:\Users\Student\Desktop\Siddhath\Bug_Fix_Commit.csv", index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")
    diff_extraction_df.to_csv(r"C:\Users\Student\Desktop\Siddhath\Diff_Extraction.csv", index=False, quoting=csv.QUOTE_ALL, lineterminator="\n")


if __name__ == "__main__":
    main()
