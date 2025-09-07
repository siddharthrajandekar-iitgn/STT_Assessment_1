import pandas as pd
import csv
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze
import code_bert_score
from sacrebleu.metrics import BLEU
import ast
import matplotlib.pyplot as plt
import os

def is_valid_python(code):
    try:
        ast.parse(code)
        return True
    except:
        return False


def main():
    df = pd.read_csv(r"C:\Users\HP\Desktop\CS_202\Labs_1-4\STT_Lab_2\Diff_Extraction.csv", encoding='latin1')
    output_df_dict = {"MI_Change": [], "CC_Change": [], "LOC_Change": [], "Semantic_Similarity": [], "Token_Similarity": [],
                      "Semantic_Class": [], "Token_Class": [], "Classes_Agree": []}
    n = len(df)


    # Total number of commits and files
    total_commits = df["Hash"].nunique()
    total_files = df["Filename"].nunique()

    print(f"Total number of commits: {total_commits}")
    print(f"Total number of files: {total_files}")

    # Average number of modified files per commit
    files_per_commit = df.groupby("Hash")["Filename"].nunique()
    avg_files_per_commit = files_per_commit.mean()

    print(f"Average number of modified files per commit: {avg_files_per_commit:.2f}")

    # Distribution of fix types from LLM Inference
    fix_type_counts = df["LLM_Inference"].value_counts()

    print("\nDistribution of Fix Types:")
    print(fix_type_counts)

    plt.figure(figsize=(8,6))
    bars = plt.bar(range(len(fix_type_counts)), fix_type_counts.values, color="cornflowerblue", edgecolor="white")

    plt.title("Distribution of Fix Types", fontsize=14)
    plt.ylabel("Count")
    plt.xticks([])  # remove x-axis labels entirely
    plt.tight_layout()
    plt.show()

    # Most frequently modified filenames/extensions 
    # Count filenames
    filename_counts = df["Filename"].value_counts().head(10)

    print("\nMost Frequently Modified Files:")
    print(filename_counts)

    # Count extensions
    df["Extension"] = df["Filename"].apply(lambda x: os.path.splitext(str(x))[1])
    ext_counts = df["Extension"].value_counts().head(10)

    print("\nMost Frequently Modified File Extensions:")
    print(ext_counts)

    # Plot extensions
    plt.figure(figsize=(8,5))
    ext_counts.plot(kind="bar", color="lightgreen", edgecolor="black")
    plt.title("Most Frequently Modified File Extensions")
    plt.xlabel("Extension")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    for i in range(n):
        old_code = df["Source_Code_Old"].iloc[i]
        new_code = df["Source_Code_New"].iloc[i]
        old_code = str(old_code) if pd.notna(old_code) else ""
        new_code = str(new_code) if pd.notna(new_code) else ""

        if not is_valid_python(old_code) or not is_valid_python(new_code):
            output_df_dict["MI_Change"].append("")
            output_df_dict["CC_Change"].append('')
            output_df_dict["LOC_Change"].append('')
            output_df_dict["Semantic_Similarity"].append('')
            output_df_dict["Token_Similarity"].append('')
            output_df_dict["Semantic_Class"].append('')
            output_df_dict["Token_Class"].append('')
            output_df_dict["Classes_Agree"].append('')
            continue


        # Cyclomatic Complexity
        cc_before = sum(func.complexity for func in cc_visit(old_code))
        cc_after = sum(func.complexity for func in cc_visit(new_code))

        cc_change = cc_after - cc_before

        # Maintainability Index
        mi_old = mi_visit(old_code, multi=False)
        mi_new = mi_visit(new_code, multi=False)

        mi_change = mi_new - mi_old

        # Lines of Code
        analyzer_old = analyze(old_code)
        analyzer_new = analyze(new_code)
        loc_before = analyzer_old[0]
        loc_after = analyzer_new[0]
        loc_change = loc_after - loc_before


        if not old_code or not new_code:
            token_similarity = 0
            semantic_similarity = 0

        else:      
            # Semantic Similarity
            reference = [[old_code]]
            hypothesis = [new_code]
            bert_score = code_bert_score.score(cands=hypothesis, refs=reference, lang="python")
            semantic_similarity = bert_score[2].item()

            # Token Similarity
            bleu = BLEU()
            token_similarity = bleu.corpus_score(hypothesis, reference).score / 100

        output_df_dict["MI_Change"].append(mi_change)
        output_df_dict["CC_Change"].append(cc_change)
        output_df_dict["LOC_Change"].append(loc_change)
        output_df_dict["Semantic_Similarity"].append(semantic_similarity)
        output_df_dict["Token_Similarity"].append(token_similarity)

        semantic_class = "Minor" if semantic_similarity >= 0.8 else "Major"
        token_class = "Minor" if token_similarity >= 0.75 else "Major"

        output_df_dict["Semantic_Class"].append(semantic_class)
        output_df_dict["Token_Class"].append(token_class)
        class_agree = "Yes" if semantic_class == token_class else "No"
        output_df_dict["Classes_Agree"].append(class_agree)

    output_df = pd.DataFrame(output_df_dict)
    output_df.to_csv(r"C:\Users\HP\Desktop\CS_202\Labs_1-4\STT_Lab_3\Output.csv")


if __name__ =='__main__':
    main()