import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import CNN  # 你的CNN模型


def fgsm_attack(model, loss_fn, data, target, epsilon):
    data.requires_grad = True
    output = model(data)
    loss = loss_fn(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data


def test_with_attack(model, device, test_loader, epsilon):
    loss_fn = nn.CrossEntropyLoss()
    correct_clean = 0
    correct_adv = 0

    model.eval()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred_clean = output.max(1, keepdim=True)[1]
        correct_clean += pred_clean.eq(target.view_as(pred_clean)).sum().item()

        adv_data = fgsm_attack(model, loss_fn, data, target, epsilon)
        output_adv = model(adv_data)
        pred_adv = output_adv.max(1, keepdim=True)[1]
        correct_adv += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    total = len(test_loader.dataset)
    acc_clean = 100. * correct_clean / total
    acc_adv = 100. * correct_adv / total
    return acc_clean, acc_adv


def plot_accuracy_vs_epsilon(model, device, test_loader):
    epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    acc_clean_list = []
    acc_adv_list = []

    for eps in epsilons:
        acc_clean, acc_adv = test_with_attack(model, device, test_loader, eps)
        acc_clean_list.append(acc_clean)
        acc_adv_list.append(acc_adv)
        print(f"Epsilon: {eps:.2f}, Clean Accuracy: {acc_clean:.2f}%, Adversarial Accuracy: {acc_adv:.2f}%")

    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, acc_clean_list, marker='o', label="真实样本准确率")
    plt.plot(epsilons, acc_adv_list, marker='s', label="对抗样本准确率", linestyle="dashed")
    plt.xlabel("扰动强度 (epsilon)")
    plt.ylabel("识别准确率 (%)")
    plt.title("FGSM 攻击下的识别准确率变化")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)  # 适当增大 batch_size 加速计算

    model = CNN().to(device)
    model_path = "./CNN.pth"  # 修改为你的模型路径
    model.load_state_dict(torch.load(model_path, map_location=device))

    plot_accuracy_vs_epsilon(model, device, test_loader)


if __name__ == '__main__':
    main()
