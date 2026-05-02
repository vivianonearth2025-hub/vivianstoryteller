import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. Streamlit UI 标题
st.title("🧑‍🦳 Age Classification using ViT")

# 2. 核心优化：使用缓存加载模型
# 加上这个装饰器后，模型只会在应用第一次启动时加载一次，之后直接调用，极大提升速度并节省内存！
@st.cache_resource
def load_age_classifier():
    return pipeline("image-classification", model="nateraw/vit-age-classifier")

# 3. 加载模型
age_classifier = load_age_classifier()

# 4. 让用户动态上传图片，而不是写死本地文件名
uploaded_file = st.file_uploader("Select an Image to guess the age...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 5. 读取上传的图片
    image = Image.open(uploaded_file).convert("RGB")
    
    # 6. 在网页上展示图片
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Processing prediction...")
    
    # 7. 进行年龄预测 (直接传入 PIL Image 对象)
    age_predictions = age_classifier(image)
    
    # 8. 排序并展示结果
    age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)
    
    st.subheader("Predicted Age Range:")
    # 提取置信度最高的结果，并将其格式化显示在网页上
    top_label = age_predictions[0]['label']
    top_score = age_predictions[0]['score']
    st.success(f"**{top_label}** (Confidence: {top_score:.2%})")

# Display results
print("Predicted Age Range:")
print(f"Age range: {age_predictions[0]['label']}")
