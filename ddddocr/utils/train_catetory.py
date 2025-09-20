import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import copy
import torch
import torch.nn as nn
import base64
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
from io import BytesIO


# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义数据增强和预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 确保都是3通道
        transforms.Resize((64, 64)),  # 统一尺寸
        transforms.RandomRotation(5),  # 小幅旋转增强
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # 小幅平移
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class CaptchaSourceClassifier(nn.Module):
    """
    CaptchaSourceClassifier 定义训练模型

    Args:
        nn (_type_): _description_
    """
    def __init__(self, num_classes, use_pretrained=True):
        super(CaptchaSourceClassifier, self).__init__()
        self.class_names = None
        # 使用预训练的ResNet作为特征提取器
        self.feature_extractor = models.resnet18(pretrained=use_pretrained)
        
        # 冻结前面的层（可选）
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # 替换最后的全连接层
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()  # 移除原全连接层
        
        # 添加自定义分类头
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)
    
    def save_model(self, path):
        torch.save({'model_state_dict': self.state_dict(), 'class_names': self.class_names}, path)

    @staticmethod
    def load(path):
        checkpoint = torch.load(path, map_location=device)
        model = CaptchaSourceClassifier(num_classes=len(checkpoint['class_names']))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.class_names = checkpoint['class_names']
        model.to(device)
        model.eval()
        return model
    
    
class ImageClassifier:

    def __init__(self):
        self.model = CaptchaSourceClassifier.load('D:\\Users\\wuquancheng\\Desktop\\image_classifier.pth')
        self.class_names = self.model.class_names

    def predict_image(self, image, device):
        """预测单张验证码图片的来源网站"""
        # 加载图像
        if not isinstance(image, bytes):
            image = Image.open(image).convert('RGB')
        else:
            image_data = base64.b64decode(image)
            image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # 应用转换
        transform = data_transforms['val']
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            _, predicted = torch.max(outputs, 1)
        
        # 获取预测结果和置信度
        predicted_class = self.class_names[predicted.item()]
        confidence = probabilities[predicted.item()].item()
        
        # 获取所有类别的置信度
        all_confidences = {
            self.class_names[i]: f"{probabilities[i].item():.4f}" 
            for i in range(len(self.class_names))
        }
        return predicted_class, confidence, all_confidences
    
    def classify(self, image):
        predicted_class, confidence, all_confidences = self.predict_image(image=image, device=device)
        # print(predicted_class, confidence, all_confidences)
        if confidence > 0.90: 
            return predicted_class
        else:
            return None


def test():
    classifier = ImageClassifier()
    image = "D:\\Users\\wuquancheng\\Desktop\\agrj_bd9ab93d08622651a25611bebf3bfe5a.jpg"
    # with open(image, "rb") as image_file:
    #     # 以二进制模式读取图片文件内容
    #     image_data = image_file.read()
    #     # 对图片数据进行Base64编码
    #     image = base64.b64encode(image_data)
    #     # 将字节串解码为UTF-8字符串
    #     # image = base64_encoded_data.decode('utf-8')
    cat = classifier.classify(image=image)
    print(cat)


if __name__ == '__main__':
    # train()
    test()
