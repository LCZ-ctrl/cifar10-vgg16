import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from PIL import Image
from config import *
from model import VGG16
from dataset import get_test_loader, get_train_val_loaders


def predict_random_one():
    model = VGG16(num_classes=NUM_CLASSES).to(DEVICE)

    checkpoint_path = "models/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loader = get_test_loader()
    test_dataset = test_loader.dataset

    idx = random.randint(0, len(test_dataset) - 1)
    img_tensor, true_label = test_dataset[idx]

    img_path = test_dataset.samples[idx][0]
    original_pil = Image.open(img_path).convert('RGB')

    with torch.no_grad():
        inputs = img_tensor.unsqueeze(0).to(DEVICE)
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)

        pred_class = test_dataset.classes[pred.item()]
        true_class = test_dataset.classes[true_label]
        conf_percent = conf.item() * 100

    print(f"idx: {idx}")
    print(f"Label: {true_class}")
    print(f"Prediction: {pred_class}")
    print(f"confidence: {conf_percent:.2f}%")

    if pred.item() == true_label:
        print("✅ Prediction correct!")
    else:
        print(f"❌ Prediction wrong!")

    plt.figure(figsize=(4, 4))
    plt.imshow(original_pil)
    plt.title(f"Label: {true_class} | Prediction: {pred_class} ({conf_percent:.1f}%)",
              fontsize=10, color='green' if pred.item() == true_label else 'red')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    predict_random_one()
# train_loader, val_loader = get_train_val_loaders()
# train_dataset = train_loader.dataset
# idx = random.randint(0, len(train_dataset) - 1)
# img_tensor, true_label = train_dataset[idx]
# img_path = train_dataset.samples[idx][0]
# original_pil = Image.open(img_path).convert('RGB')
# plt.figure(figsize=(4, 4))
# plt.imshow(original_pil)
# plt.axis('off')
# plt.show()
