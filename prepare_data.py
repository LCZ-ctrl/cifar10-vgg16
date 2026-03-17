import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm


# train: 0.9, val: 0.1
def prepare_data(
        raw_train_dir: str = "data/raw/train",
        raw_test_dir: str = "data/raw/test",
        processed_dir: str = "data/processed",
        val_ratio: float = 0.1,
        seed: int = 42
):
    random.seed(seed)
    raw_train_path = Path(raw_train_dir)
    raw_test_path = Path(raw_test_dir)
    processed_path = Path(processed_dir)

    classes = sorted([d for d in os.listdir(raw_train_path) if os.path.isdir(raw_train_path / d)])
    print(f"✅ Found {len(classes)} categories: {classes}")

    for split in ['train', 'val']:
        for category in classes:
            (processed_path / split / category).mkdir(parents=True, exist_ok=True)
    for category in classes:
        (processed_path / 'test' / category).mkdir(parents=True, exist_ok=True)

    tqdm.write("🚀 Splitting training set...")
    for category in classes:
        img_list = [f for f in os.listdir(raw_train_path / category)
                    if f.lower().endswith('.png')]
        val_size = int(len(img_list) * val_ratio)
        random.shuffle(img_list)

        train_imgs = img_list[val_size:]
        val_imgs = img_list[:val_size]

        copy_files(train_imgs, raw_train_path / category,
                   processed_path / 'train' / category, f'Train {category}')
        copy_files(val_imgs, raw_train_path / category,
                   processed_path / 'val' / category, f'Val {category}')

    tqdm.write("🚀 Preparing test set...")
    for category in classes:
        img_list = [f for f in os.listdir(raw_test_path / category)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        copy_files(img_list, raw_test_path / category,
                   processed_path / 'test' / category, f'Test {category}')

    train_total = sum(len(list((processed_path / 'train' / c).glob('*'))) for c in classes)
    val_total = sum(len(list((processed_path / 'val' / c).glob('*'))) for c in classes)
    test_total = sum(len(list((processed_path / 'test' / c).glob('*'))) for c in classes)

    tqdm.write(f"🎉 Data preparation complete!")
    tqdm.write(f"Training set: {train_total} images")
    tqdm.write(f"Validation set: {val_total} images")
    tqdm.write(f"Test set: {test_total} images")


def copy_files(file_list, src_dir, dst_dir, desc='Copying'):
    for file in tqdm(file_list, desc=f"📂 {desc}"):
        shutil.copy2(src_dir / file, dst_dir / file)


if __name__ == "__main__":
    prepare_data()
