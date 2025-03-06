import torch
import numpy as np
from PIL import Image
import os
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from models.models import MBR_model

# 加载预训练的车辆重识别模型
class_num = 575  # 实际的训练集类别数
n_branches = ["R50", "R50", "BoT", "BoT"]  # 分支数设置
losses = "LBS"
n_groups = 0  # 组数设置
model = MBR_model(class_num, n_branches, n_groups, losses)

# 加载预训练状态字典
state_dict = torch.load(r"D:\MBR4B\logs\Veri776\MBR_4B\HydraAttention\best_mAP.pt")
# 手动过滤不匹配的参数
model_state_dict = model.state_dict()
filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and v.size() == model_state_dict[k].size()}
# 加载过滤后的状态字典
model_state_dict.update(filtered_state_dict)
model.load_state_dict(model_state_dict, strict=False)
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)
    return image

# 提取图像特征
def extract_features(image_path, model, device):
    image = load_and_preprocess_image(image_path).to(device)
    with torch.no_grad():
        preds, features, _, _ = model(image, torch.tensor([0]).to(device), torch.tensor([0]).to(device))
    if isinstance(features, list):
        features = torch.cat(features, dim=1)  # 将列表中的特征拼接起来
    return features.cpu().numpy()

# 读取标签文件，并从文件名中解析标签
def get_image_paths_labels(label_file):
    image_paths = {}
    with open(label_file, 'r') as file:
        for line in file:
            image_id = line.strip()
            if not image_id:
                print(f"Skipping malformed line: {line.strip()}")
                continue
            # 解析文件名前四个字符作为标签，例如 0002
            label = image_id.split('_')[0]
            image_paths[image_id] = label
    return image_paths

# 加载数据库图像路径和标签
database_image_label_path = r"D:\MBR4B\vehicle_reid_itsc2023-main\dataset\VeRi\name_test.txt"  # 标签文件路径
database_image_folder = r"D:\MBR4B\vehicle_reid_itsc2023-main\dataset\VeRi\image_test"
database_image_paths_labels = get_image_paths_labels(database_image_label_path)

# 提取数据库图像的特征和标签，过滤掉与查询图像路径完全相同的样本
def extract_database_features_and_labels(database_image_paths_labels, model, device, query_image_path):
    database_features = []
    database_labels = []
    query_image_name = os.path.basename(query_image_path)  # 获取查询图像的文件名
    for image_id, label in tqdm(database_image_paths_labels.items()):
        if image_id == query_image_name:  # 仅过滤掉与查询图像路径完全相同的样本
            continue
        image_path = os.path.join(database_image_folder, image_id)
        features = extract_features(image_path, model, device)
        database_features.append(features)
        database_labels.append(label)
    return np.vstack(database_features), database_labels

# 加载查询图像并提取特征
query_image_path = r"D:\MBR4B\vehicle_reid_itsc2023-main\dataset\VeRi\image_query\0231_c017_00035840_0.jpg"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
query_features = extract_features(query_image_path, model, device)

# 提取数据库图像的特征和标签，并传入查询图像路径
database_features, database_labels = extract_database_features_and_labels(database_image_paths_labels, model, device, query_image_path)

# 计算相似度
similarities = cosine_similarity(query_features, database_features)
top_k_indices = np.argsort(similarities[0])[::-1][:10]  # 获取前10个最相似图像的索引

# 可视化查询结果
# 可视化查询结果
def visualize_single_row_results(query_image_path, database_image_paths_labels, top_k_indices, labels, image_folder, query_label):
    # 读取和处理查询图像
    query_image = cv2.imread(query_image_path)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    query_image = cv2.resize(query_image, (256, 256))
    # 移除边框设置，这里我们不应用 cv2.copyMakeBorder 来添加边框了
    # query_image = cv2.copyMakeBorder(query_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 255, 0])

    # 创建单行的matplotlib figure来显示图像和对应的文本标签
    fig, axes = plt.subplots(1, 11, figsize=(20, 4))  # 调整为7个子图（包括查询图像）

    # 显示查询图像
    axes[0].imshow(query_image)
    axes[0].axis('off')
    # # 使用text方法在图像下方添加文本，调整文本的大小
    # axes[0].text(0.5, -0.1, "Query", fontsize=8, ha='center', va='center', transform=axes[0].transAxes)

    # 遍历前 10个最相似的图像
    for i, idx in enumerate(top_k_indices, start=1):
        db_image_id = list(database_image_paths_labels.keys())[idx]
        db_image_label = labels[idx]
        db_image_path = os.path.join(image_folder, db_image_id)
        db_image = cv2.imread(db_image_path)
        db_image = cv2.cvtColor(db_image, cv2.COLOR_BGR2RGB)
        db_image = cv2.resize(db_image, (256, 256))

        if db_image_label == query_label:
            color = [0, 255, 0]  # 标记与查询图像标签相同的图像边框为绿色
        else:
            color = [255, 0, 0]  # 反之为红色

        db_image = cv2.copyMakeBorder(db_image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=color)

        # 显示图像
        axes[i].imshow(db_image)
        axes[i].axis('off')
        # # 在图像下方添加文本标签
        # axes[i].text(0.5, -0.1, f"Label: {db_image_label}", fontsize=8, ha='center', va='center', transform=axes[i].transAxes)

    plt.tight_layout()
    plt.show()

# 调用函数
query_label = "0231"  # 查询图像ID
visualize_single_row_results(query_image_path, database_image_paths_labels, top_k_indices, database_labels, database_image_folder, query_label)