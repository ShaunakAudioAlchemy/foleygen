import os
import csv
import random




clas_dict = {
    "DogBark": 0,
    "Footstep": 1,
    "Gunshot": 2,
    "Keyboard": 3,
    "MovingMotorVehicle": 4,
    "Rain": 5,
    "SneezeCough": 6,
}


def split_dataset_files(dataset_path: str,output_csv_path: str, train_ratio: float = 0.8, test_ratio: float = 0.1, seed: int = None):
    training_files: List[dict] = []
    valid_files: List[dict] = []
    test_files: List[dict] = []

    all_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                class_id = clas_dict.get(os.path.basename(root), None)
                if class_id is not None:
                    all_files.append({"class_id": class_id, "file_path": os.path.join(root, file)})

    if seed is not None:
        random.seed(seed)
    random.shuffle(all_files)

    total_files = len(all_files)
    num_train = int(total_files * train_ratio)
    num_test = int(total_files * test_ratio)
    num_valid = total_files - num_train - num_test

    training_files = all_files[:num_train]
    test_files = all_files[num_train:num_train + num_test]
    valid_files = all_files[num_train + num_test:]

    with open(output_csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filepath", "split"])
        for file_info in training_files:
            writer.writerow([file_info["file_path"], "train"])
        for file_info in test_files:
            writer.writerow([file_info["file_path"], "test"])
        for file_info in valid_files:
            writer.writerow([file_info["file_path"], "validation"])

    return training_files, valid_files, test_files

def print_random_file_details(training_files, valid_files, test_files):
    """
    Prints details of one random audio file from either the training, validation, or test set.
    """
    all_files = [
        {"file": file, "split": "train"} for file in training_files
    ] + [
        {"file": file, "split": "validation"} for file in valid_files
    ] + [
        {"file": file, "split": "test"} for file in test_files
    ]

    random_file = random.choice(all_files)

    print(f"Randomly Selected File:")
    print(f"File Path: {random_file['file']['file_path']}")
    print(f"Class ID: {random_file['file']['class_id']}")
    print(f"Split: {random_file['split']}")

