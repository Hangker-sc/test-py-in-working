import os
import sys
import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from ResNet_model import resnet34

features = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4, "dog": 5, "frog": 6, "horse": 7,
                "ship": 8, "truck": 9}
#自定义数据集
class CustomImageDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        path_list = []
        kind_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)]
        for kind_path in kind_paths:
            for img_path in os.listdir(kind_path):
                path_list.append(os.path.join(kind_path, img_path))

        self.image_paths = path_list
        self.labels = self._get_labels()  # 假设标签是某种方式从文件名或目录中获取的

    def _get_labels(self):
        # 这里你需要实现获取标签的逻辑
         # 例如，从文件名中提取类别信息
        labels = []
        for path in self.image_paths:
             # 假设标签是文件名的一部分，你需要根据实际情况调整这部分代码
            label = path.split("\\")[-2]
            labels.append(features[label])
        return labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # 打开图片并转为RGB格式
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),  # 随机裁剪
                                     transforms.RandomHorizontalFlip(),  # 随机翻转
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "test": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

     #数据集载入
    batch_size =100
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    trainset = CustomImageDataset(r".\data\data\train", transform=data_transform["train"])
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=nw)
    testset = CustomImageDataset(r".\data\data\test", transform=data_transform["test"])
    test_loader = DataLoader(testset, batch_size=100, shuffle=True, num_workers=nw)

    #搭建网络
    net = resnet34(num_classes=10)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    LOSS=[]
    ACC=[]
    EPOCHS=[]

    epochs = 10
    save_path = './ResNet.pth'
    best_acc =0.0
    train_steps = len(train_loader)
    #10次迭代训练
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            running_loss+=loss.item()
            loss.backward()
            optimizer.step()

        # validate
        net.eval()  # 在测试时Dropout不起作用
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in test_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / len(test_loader)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        EPOCHS.append(epoch + 1)
        LOSS.append(running_loss / train_steps)
        ACC.append(val_accurate)

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

    #可视化
    plt.plot(EPOCHS, LOSS, marker='o',color="b")  # marker='o' 添加数据点标记
    plt.plot(EPOCHS, ACC, marker='o',color="r")  # marker='o' 添加数据点标记
    # 设置图表坐标轴标签
    plt.xlabel('EPOCHS')
    plt.ylabel('Y-axis')
    # 显示图表
    plt.show()

if __name__ == '__main__':
    main()