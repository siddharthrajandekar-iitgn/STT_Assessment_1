import pandas as pd
import matplotlib.pyplot as plt
from pydriller import Repository

def main():
    repos = ["https://github.com/ggml-org/ggml", "https://github.com/1panel-dev/1panel", "https://github.com/2dust/v2rayn"]

    final_dataset_dict = {"old_file_path": [], "new_file_path": [], "commit_SHA": [], "parent_commit_SHA": [], "commit_message": [],
                          "diff_myers": [], "diff_hist": [], "Discrepancy": []}
    
    
    n_src_files = 0
    mis_src_files = 0
    n_test_files = 0
    mis_test_files = 0
    n_readme_files = 0
    mis_readme_files = 0
    n_license_files = 0
    mis_license_files = 0

    for commit in Repository(repos, skip_whitespaces=True, histogram_diff=False).traverse_commits():
        for file in commit.modified_files:
            final_dataset_dict["old_file_path"].append(file.old_path)
            final_dataset_dict["new_file_path"].append(file.new_path)
            final_dataset_dict["commit_SHA"].append(commit.hash)
            if commit.parents:
                final_dataset_dict["parent_commit_SHA"].append(commit.parents[0])
            else:
                final_dataset_dict["parent_commit_SHA"].append("")
            final_dataset_dict["commit_message"].append(commit.msg)
            final_dataset_dict["diff_myers"].append(file.diff)

    idx = 0
    for commit in Repository(repos, skip_whitespaces=True, histogram_diff=True).traverse_commits():
        for file in commit.modified_files:
            final_dataset_dict["diff_hist"].append(file.diff)
            if file.diff == final_dataset_dict["diff_myers"][idx]:
                final_dataset_dict["Discrepancy"].append("No")
            else:
                final_dataset_dict["Discrepancy"].append("Yes")
            
            path = ""
            if file.old_path:
                path = file.old_path
            else:
                path = file.new_path

            if "test" in path:
                if final_dataset_dict["Discrepancy"][-1] == "Yes":
                    mis_test_files += 1
                n_test_files += 1

            elif file.filename == "README.md":
                if final_dataset_dict["Discrepancy"][-1] == "Yes":
                    mis_readme_files += 1
                n_readme_files += 1

            elif file.filename == "LICENSE.txt" or file.filename == "LICENSE":
                if final_dataset_dict["Discrepancy"][-1] == "Yes":
                    mis_license_files += 1
                n_license_files += 1
            
            else:
                if final_dataset_dict["Discrepancy"][-1] == "Yes":
                    mis_src_files += 1
                n_src_files += 1

            idx += 1

    categories = ["Test Files", "README", "License", "Source Files"]
    discrepancies = [mis_test_files, mis_readme_files, mis_license_files, mis_src_files]
    totals = [n_test_files, n_readme_files, n_license_files, n_src_files]

    print("Discrepancies by Category:")
    for cat, mis, total in zip(categories, discrepancies, totals):
        print(f"{cat}: {mis} / {total} files had discrepancies")

    plt.figure(figsize=(7,5))
    bars = plt.bar(categories, discrepancies, color=["#66c2a5","#fc8d62","#8da0cb","#e78ac3"], edgecolor="black")

    for bar, count in zip(bars, discrepancies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, str(count),
                ha='center', va='bottom', fontsize=9)

    plt.title("Number of Discrepancies by File Type")
    plt.ylabel("Discrepancy Count")
    plt.tight_layout()
    plt.show()


    df = pd.DataFrame(final_dataset_dict)
    df.to_csv(r"C:\Users\HP\Desktop\CS_202\Labs_1-4\STT_Lab_4\Discrepency.csv", index=False)

if __name__ == "__main__":
    main()
                