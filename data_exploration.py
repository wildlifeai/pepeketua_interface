import os


def filter_non_jpgs(file_name: str) -> bool:
    return file_name.lower()[-3:] == "jpg"


if __name__ == "__main__":
    folder_files_hash = dict()
    path = "/Users/lioruzan/Downloads/frog_pics"
    for root, folders, other_jpgs in os.walk(path):
        current_jpgs = set(filter(filter_non_jpgs, other_jpgs))
        for other_root, other_jpgs in folder_files_hash.items():
            if len(other_jpgs & current_jpgs) > 0:
                print(root, other_root, list(other_jpgs & current_jpgs))
        folder_files_hash[root] = current_jpgs
