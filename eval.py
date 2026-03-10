import torch
from tqdm import tqdm
from dataset import get_test_loader
from model import VGG16
from config import *
from utils import *


def evaluate_test_set():
    model = VGG16(num_classes=NUM_CLASSES).to(DEVICE)

    checkpoint_path = "models/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loader = get_test_loader()
    test_dataset = test_loader.dataset

    test_correct = 0.0
    print(f"🔍 Evaluating on {len(test_dataset)} test images...")

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            test_correct += calculate_correct(outputs, labels)

    avg_test_acc = test_correct / len(test_dataset)
    tqdm.write(f"test acc: {avg_test_acc:.4f}")


if __name__ == "__main__":
    evaluate_test_set()
