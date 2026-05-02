import streamlit as st
from transformers import pipeline
from PIL import Image

# 1. Streamlit UI 标题
st.title("🧑‍🦳 Age Classification using ViT")

# 2. 核心优化：使用缓存加载模型
@st.cache_resource
def load_age_classifier():
    return pipeline("image-classification", model="nateraw/vit-age-classifier")

# 3. 加载模型
age_classifier = load_age_classifier()

# 4. 接收用户上传图片
uploaded_file = st.file_uploader("Select an Image to guess the age...", type=["jpg", "jpeg", "png"])

# 5. 只有在图片上传后，才执行以下所有逻辑
if uploaded_file is not None:
    # 读取图片
    image = Image.open(uploaded_file).convert("RGB")
    
    # 在网页上展示图片
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    st.write("Processing prediction...")
    
    # 进行年龄预测
    age_predictions = age_classifier(image)
    
    # 排序
    age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)
    
    # 展示结果（这部分已经在 if 里面了，所以绝对安全）
    st.subheader("Predicted Age Range:")
    top_label = age_predictions[0]['label']
    top_score = age_predictions[0]['score']
    st.success(f"**{top_label}** (Confidence: {top_score:.2%})")

# 注意：这个文件的最下面不应该再有任何其他没缩进的代码了！
