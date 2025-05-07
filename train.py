import json
import torch
from torch import nn
from DAMSLNet import DAMSLNet
import numpy as np
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time


# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取图片路径
ROOT_TRAIN = './Apple_train'
ROOT_TEST = './Apple_test'

# Normalize trained on ImageNet dataset
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    # transforms.ColorJitter(brightness=0.2),
    # transforms.ColorJitter(contrast=0.2),
    # transforms.ColorJitter(saturation=0.2),
    # transforms.ColorJitter(hue=0.2),
    transforms.ToTensor(),
    normalize])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    normalize])

# Preprocessing the dataset
train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
test_dataset = ImageFolder(ROOT_TEST, transform=test_transform)

# Print the number of training sets and testing sets
train_num = len(train_dataset)
print('Using {} images for training'.format(train_num))
test_num = len(test_dataset)
print('Using {} images for testing'.format(test_num))

# Use dataloader to load datasets
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Set up training equipment
device = torch.device("cuda:0")
print("using {} device.".format(device))
# Loading the model
model = DAMSLNet(12).to(device)
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameters: %.2fM" % (total / 1e6))

# Define a loss function
loss_fn = nn.CrossEntropyLoss()

# Define an optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
 
    for (x, y) in dataloader:
        image, y = x.to(device), y.to(device)
        output = model(image)
        cur_loss = loss_fn(output, y) 
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(y==pred) / output.shape[0]

        # Backpropagation
        optimizer.zero_grad() 
        cur_loss.sum().backward(retain_graph=True) 
        
        optimizer.step()              
        loss += cur_loss.item()       
        current += cur_acc.item()    
        n = n+1                     

    train_loss = loss / n
    train_acc = current / n
    print('train_loss' + str(train_loss))
    print('train_acc' + str(train_acc))
    return train_loss, train_acc

# 定义验证函数
def test(dataloader, model, loss_fn):
    # 将模型转化为验证模型
    model.eval()
    y_true_label = []
    y_pred_out = []
    y_pred_label = []
    loss, current, n = 0.0, 0.0, 0

    with torch.no_grad():
        for (x, y) in dataloader:
            image, y = x.to(device), y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred_label = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred_label) / output.shape[0] 
            loss += cur_loss.item()                       
            current += cur_acc.item()
            n = n + 1

            y_true_label.extend(y.tolist()) 
            y_pred_label.extend(pred_label.tolist()) 
            y_pred_out.extend(output.tolist()) 

    test_loss = loss / n
    test_acc = current / n
    print('test_loss' + str(test_loss))
    print('test_acc' + str(test_acc))
    return test_loss, test_acc, y_true_label, y_pred_label,y_pred_out

# Define drawing functions
def matplot_loss(train_loss, test_loss, train_acc, test_acc):
    plt.plot(train_loss, label='train_loss')
    plt.plot(test_loss, label='test_loss')
    plt.plot(train_acc, label='train_acc')
    plt.plot(test_acc, label='test_acc')
    # Set the axis range
    plt.ylim((0, 1))
    plt.legend(loc='best')
    # Set the axis name
    plt.ylabel('loss/acc')
    plt.xlabel('epoch')
    plt.title("DAMSLNet")
    # plt.show()
    plt.savefig("DAMSLNet.jpg")
 
# Start training
loss_train = []
acc_train = []
loss_test = []
acc_test = []


epoch = 100
min_acc = 0
for t in range(epoch):
    print(f"epoch{t + 1}\n-----------")
    current_path = os.path.dirname(__file__)
    
    folder = current_path + '/save_FGVC8/DAMSLNet'
    if not os.path.exists(folder):
        os.mkdir('/save_FGVC8/DAMSLNet')
    torch.save(model.state_dict(), folder + '/plant12.pth')
    print("save model success")
    start_time = time.time()
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"epoch{t+1} used {epoch_time:.2f} seconds")
    test_loss, test_acc, y_true_label, y_pred_label,y_pred_out = test(test_dataloader, model, loss_fn)

    loss_train.append(train_loss)          
    acc_train.append(train_acc)
    loss_test.append(test_loss)
    acc_test.append(test_acc)

    # Save the best model weights
    if test_acc >min_acc:
        folder = current_path + '/save_FGVC8/DAMSLNet'
        min_acc = test_acc
        print(f"save best model, 第{t+1}轮")
        torch.save(model.state_dict(), folder + '/best_model.pth')
    # Save the last round of weight files
    if t == epoch-1:
        torch.save(model.state_dict(),  folder + '/last_model.pth')

matplot_loss(loss_train, loss_test, acc_train, acc_test)
print('Done!')


current_path = os.path.dirname(__file__)
file = open(current_path + '/save_FGVC8/DAMSLNet/plant12.json', 'w')
file.write('[')
file.write(str(loss_train))
file.write('\n,\n')
file.write(str(loss_test))
file.write('\n,\n')
file.write(str(acc_train))
file.write('\n,\n')
file.write(str(acc_test))
file.write(']')


if __name__ == '__main__':
    test_loss, test_acc, y_true_label, y_pred_label,y_pred_out = test(test_dataloader, model, loss_fn)
    with open("true_label.json", 'w') as file:
        json.dump(y_true_label, file, indent=4)
    with open("pred_label.json", 'w') as file:
        json.dump(y_pred_label, file, indent=4)
    with open("pred_prob.json", 'w') as file:
        json.dump(y_pred_out, file, indent=4)
    with open("loss_train.json", 'w') as file:
        json.dump(loss_train, file, indent=4)
    with open("acc_train.json", 'w') as file:
        json.dump(acc_train, file, indent=4)
    with open("loss_test.json", 'w') as file:
        json.dump(loss_test, file, indent=4)
    with open("acc_test.json", 'w') as file:
        json.dump(acc_test, file, indent=4)


