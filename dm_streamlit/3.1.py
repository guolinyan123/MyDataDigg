import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern, hog
import cv2
import torch
import warnings

warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载CIFAR-100数据集
from keras.datasets import cifar100

(X_train, y_train), (X_test, y_test) = cifar100.load_data()


# LBP特征提取
def extract_lbp_features(image):
    # print(image.shape)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(257), density=True)
    return lbp_hist


# HOG特征提取
def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hog_features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    return hog_features


print(X_train[0].shape)  # 检查图像数据的形状


# SIFT特征提取（带默认值）
# SIFT特征提取
def extract_sift_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    # 确保描述子是固定长度的
    if descriptors is not None:
        if descriptors.shape[0] < 256:
            # 如果检测到的关键点数量不足，填充零向量
            padded_descriptors = np.zeros((256, descriptors.shape[1]), dtype=np.float32)
            padded_descriptors[:descriptors.shape[0], :] = descriptors
            descriptors = padded_descriptors
        elif descriptors.shape[0] > 256:
            # 如果检测到的关键点数量过多，截断到固定长度
            descriptors = descriptors[:256]
    else:
        # 如果没有检测到关键点，返回全零特征向量
        descriptors = np.zeros((256, 128), dtype=np.float32)

    return descriptors.flatten()


# def extract_sift_features(image):
#     gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     sift = cv2.SIFT_create()
#     keypoints, descriptors = sift.detectAndCompute(gray_image, None)
#     if descriptors is not None and len(keypoints) > 256:
#         return list(descriptors.flatten()[:256])
#     else:
#         return [.5] * 256


# 提取所有图像的特征
def extract_features(images, feature_extractor):
    features = []
    for img in images:
        features.append(feature_extractor(img))
    return np.array(features)


# 特征提取和降维
def process_features(X_train, X_test, feature_extractor):
    train_features = extract_features(X_train, feature_extractor)
    test_features = extract_features(X_test, feature_extractor)

    # 使用PCA进行降维
    pca = PCA(n_components=50)
    print(train_features.shape)
    train_features_pca = pca.fit_transform(train_features)
    test_features_pca = pca.transform(test_features)

    return train_features_pca, test_features_pca


# 分类模型训练和评估
import joblib


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    # 将模型参数持久化存储下来
    joblib.dump(model, model_name + '.pkl')
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# 提取并处理LBP特征
X_train_lbp, X_test_lbp = process_features(X_train, X_test, extract_lbp_features)
# 提取并处理HOG特征
X_train_hog, X_test_hog = process_features(X_train, X_test, extract_hog_features)
# 提取并处理SIFT特征
X_train_sift, X_test_sift = process_features(X_train, X_test, extract_sift_features)

# 使用分类模型进行分类
# 使用分类模型进行分类
models = {
    'GaussianNB': GaussianNB(),  # 朴素贝叶斯模型
    'KNeighborsClassifier': KNeighborsClassifier(),  # KNN模型
    'LogisticRegression': LogisticRegression(max_iter=100)  # 逻辑回归模型
}

# 对每种特征进行分类评估
for feature_name, (X_train_feat, X_test_feat) in zip(['LBP', 'HOG', 'SIFT'],
                                                     [(X_train_lbp, X_test_lbp), (X_train_hog, X_test_hog),
                                                      (X_train_sift, X_test_sift)]):
    print(f"采用 {feature_name} 特征进行分类的结果为:")
    for model_name, model in models.items():
        accuracy = evaluate_model(model_name, model, X_train_feat, y_train.ravel(), X_test_feat, y_test.ravel())
        print(f"{model_name} Accuracy: {accuracy:.4f}")