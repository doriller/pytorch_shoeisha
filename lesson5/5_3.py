print('***** リスト5.1 データ拡張 *****')

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

cifar10_data = CIFAR10(root="./data",
                       train=False,download=True,
                       transform=transforms.ToTensor())
cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
print("データの数:", len(cifar10_data))

n_image = 25  # 表示する画像の数
cifar10_loader = DataLoader(cifar10_data, batch_size=n_image, shuffle=True)
dataiter = iter(cifar10_loader)  # イテレータ: 要素を順番に取り出せるようにする
#images, labels = dataiter.next()  # 最初のバッチを取り出す
images, labels = next(dataiter) # 最初のバッチを取り出す 上記内容ではエラーになる

plt.figure(figsize=(10,10))  # 画像の表示サイズ
for i in range(n_image):
    ax = plt.subplot(5,5,i+1)
    ax.imshow(images[i].permute(1, 2, 0))  # チャンネルを一番後の次元に
    label = cifar10_classes[labels[i]]
    ax.set_title(label)
    ax.get_xaxis().set_visible(False)  # 軸を非表示に
    ax.get_yaxis().set_visible(False)

# 結果画像を表示なら
#plt.show()

import os
#print('getcwd:      ', os.getcwd())
#print('__file__:    ', __file__)
#print('dirname:     ', os.path.dirname(__file__))
dirname = os.path.dirname(__file__)
filedir = dirname + '/file'
if not os.path.exists(filedir):
    os.makedirs(filedir)

# 結果画像を保存なら
savepath = filedir + '/5-1.jpg'
plt.savefig(savepath)

print('***** リスト5.2 回転とリサイズ *****')

transform = transforms.Compose([transforms.RandomAffine((-45, 45), scale=(0.5, 1.5)),  # 回転とリサイズ
                                transforms.ToTensor()])
cifar10_data = CIFAR10(root="./data",
                       train=False,download=True,
                       transform=transform)

cifar10_loader = DataLoader(cifar10_data, batch_size=n_image, shuffle=True)
dataiter = iter(cifar10_loader)
#images, labels = dataiter.next()
images, labels = next(dataiter) # 最初のバッチを取り出す 上記内容ではエラーになる

plt.figure(figsize=(10,10))  # 画像の表示サイズ
for i in range(n_image):
    ax = plt.subplot(5,5,i+1)
    ax.imshow(images[i].permute(1, 2, 0))
    label = cifar10_classes[labels[i]]
    ax.set_title(label)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#plt.show()
savepath = filedir + '/5-2.jpg'
plt.savefig(savepath)

print('***** リスト5.3 シフト *****')

transform = transforms.Compose([transforms.RandomAffine((0, 0), translate=(0.5, 0.5)),  # 上下左右へのシフト
                                transforms.ToTensor()])
cifar10_data = CIFAR10(root="./data",
                       train=False,download=True,
                       transform=transform)

cifar10_loader = DataLoader(cifar10_data, batch_size=n_image, shuffle=True)
dataiter = iter(cifar10_loader)
#images, labels = dataiter.next()
images, labels = next(dataiter) # 最初のバッチを取り出す 上記内容ではエラーになる

plt.figure(figsize=(10,10))  # 画像の表示サイズ
for i in range(n_image):
    ax = plt.subplot(5,5,i+1)
    ax.imshow(images[i].permute(1, 2, 0))
    label = cifar10_classes[labels[i]]
    ax.set_title(label)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#plt.show()
savepath = filedir + '/5-3.jpg'
plt.savefig(savepath)

print('***** リスト5.4 反転 *****')

transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),  # 左右反転
                                transforms.RandomVerticalFlip(p=0.5),  # 上下反転
                                transforms.ToTensor()])
cifar10_data = CIFAR10(root="./data",
                       train=False,download=True,
                       transform=transform)

cifar10_loader = DataLoader(cifar10_data, batch_size=n_image, shuffle=True)
dataiter = iter(cifar10_loader)
#images, labels = dataiter.next()
images, labels = next(dataiter) # 最初のバッチを取り出す 上記内容ではエラーになる

plt.figure(figsize=(10,10))  # 画像の表示サイズ
for i in range(n_image):
    ax = plt.subplot(5,5,i+1)
    ax.imshow(images[i].permute(1, 2, 0))
    label = cifar10_classes[labels[i]]
    ax.set_title(label)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#plt.show()
savepath = filedir + '/5-4.jpg'
plt.savefig(savepath)

print('***** リスト5.5 一部を消去 *****')

# ErasingはTensorにしか適用できないからtransforms.ToTensor()クラスの後に記述
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.RandomErasing(p=0.5)])  # 一部を消去
cifar10_data = CIFAR10(root="./data",
                       train=False,download=True,
                       transform=transform)

cifar10_loader = DataLoader(cifar10_data, batch_size=n_image, shuffle=True)
dataiter = iter(cifar10_loader)
#images, labels = dataiter.next()
images, labels = next(dataiter) # 最初のバッチを取り出す 上記内容ではエラーになる

plt.figure(figsize=(10,10))  # 画像の表示サイズ
for i in range(n_image):
    ax = plt.subplot(5,5,i+1)
    ax.imshow(images[i].permute(1, 2, 0))
    label = cifar10_classes[labels[i]]
    ax.set_title(label)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#plt.show()
savepath = filedir + '/5-5.jpg'
plt.savefig(savepath)
