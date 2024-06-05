import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image
img_url = 'https://img.zcool.cn/community/0156cb59439764a8012193a324fdaa.gif'  # 背景图片的网址
st.markdown('''<style>.css-fg4pbf{background-image:url(''' + img_url + ''');
background-size:100% 100%;background-attachment:fixed;}</style>
''', unsafe_allow_html=True)  # 修改背景样式

st.title('CIFAR-100数据集的分类任务')

# 在侧边栏创建一个选择框，用于选择算法
algorithm = st.sidebar.selectbox('1.算法选择：', ['逻辑回归', 'KNN', '朴素贝叶斯'])

# 根据选择的算法显示相应的信息
if algorithm == '逻辑回归':
    st.write('您选择了逻辑回归算法。')
elif algorithm == 'KNN':
    st.write('您选择了K最近邻算法。')
elif algorithm == '朴素贝叶斯':
    st.write('您选择了朴素贝叶斯算法。')
with open('classes.txt', 'r') as f:
    cat = f.readlines()

minmaxscaler = st.sidebar.radio('2.是否持久化模型：', ['是', '否'])

# 在侧边栏创建一个文本区域用于上传图像
uploaded_file = st.sidebar.file_uploader("3.上传图像：", type=["png", "jpg", "jpeg"])

# 如果有图像被上传，则显示图像
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='上传的图像', use_column_width=True)
else:
    st.sidebar.write("请上传图像文件")
figure2 = st.sidebar.text('4.准确率展示：')

# 定义一个函数来处理按钮点击事件
def show_accuracy_image():
    st.image("img.png", caption='准确率展示', use_column_width=True)
    st.image("1.png", caption='损失函数', use_column_width=True)
# 在侧边栏创建一个按钮
st.sidebar.button('显示准确率图', on_click=show_accuracy_image)
# 加载预训练的模型参数
import pickle

# 使用pickle加载朴素贝叶斯模型
with open('朴素贝叶斯模型.pkl', 'rb') as file:
    model_1 = pickle.load(file)

# 如果需要的话，也可以用同样的方式加载其他模型，比如KNN模型
with open('KNN模型.pkl', 'rb') as file:
    model_3 = pickle.load(file)


# 定义模型字典
models = {
    '朴素贝叶斯': model_1,
    # '实逻辑回归': model_2,
    ' KNN模型 ': model_3,
}


# 创建包含数据的字典
data = {
    'LBP': [0.072, 0.049, 0.0578],
    'HOG': [0.2001, 0.1770, 0.1954],
    'SIFT': [0.0573, 0.0307, 0.0604]
}

# 设置算法名称
algorithms = ['朴素贝叶斯', 'KNN', '逻辑回归']

# 设置特征名称
features = list(data.keys())

# 设置柱状图宽度
bar_width = 0.2

# 绘制柱状图
for i, feature in enumerate(features):
    plt.bar([j + i * bar_width for j in range(len(algorithms))], data[feature], width=bar_width, label=feature)

# 设置x轴刻度
plt.xticks([i + bar_width for i in range(len(algorithms))], algorithms)

# 设置图表标题、x轴和y轴标签
plt.title('不同特征在不同算法下的准确率')
plt.xlabel('算法')
plt.ylabel('准确率')

# 显示图例
plt.legend()

# 显示图表
plt.show()
import streamlit as st

# 假设这是你的模型评估结果数据结构，实际情况中这些值可能来自模型训练和评估过程
results = {
    'LBP': {'GaussianNB': 0.073, 'KNeighborsClassifier': 0.049, 'LogisticRegression': 0.0578},
    'HOG': {'GaussianNB': 0.1988, 'KNeighborsClassifier': 0.1785, 'LogisticRegression': 0.1937},
    'SIFT': {'GaussianNB': 0.055, 'KNeighborsClassifier': 0.0307, 'LogisticRegression': 0.061}
}

# 创建一个下拉菜单供用户选择特征
selected_feature = st.selectbox("选择特征类型", list(results.keys()))

# 当用户做出选择后，展示该特征对应的模型结果
if selected_feature:
    st.write(f"### {selected_feature} 特征的分类结果:")
    for model, accuracy in results[selected_feature].items():
        st.write(f"{model}: 准确率 {accuracy * 100:.2f}%")
        # 这里仅展示信息，实际保存模型的过程应该在模型训练阶段完成，不建议直接在Streamlit应用中执行模型保存操作

# 添加一个按钮，用于展示表1的数据
if st.button("显示表1: 实验数据"):
    st.write("### 表1: 不同特征提取算法和分类器的实验数据(%)")
    st.markdown(
        """
        | 特征提取算法 | 朴素贝叶斯 | KNN | 逻辑回归 |
        |-------------|-----------|-----|----------|
        | LBP         | 0.073     | 0.049| 0.0578   |
        | HOG         | 0.1988    | 0.1785| 0.1937   |
        | SIFT        | 0.055     | 0.0307| 0.061    |
        """
    )
