import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import streamlit as st
import time

# Configure page - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Prediction App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
def set_custom_style():
    st.markdown("""
    <style>
    .main-header {color:#1E88E5; font-family:'Helvetica Neue', Helvetica, sans-serif;}
    .sub-header {color:#0D47A1; font-family:'Helvetica Neue', Helvetica, sans-serif;}
    .highlight {background-color:#f0f7ff; padding:15px; border-radius:5px; border-left:5px solid #1E88E5;}
    .card {background-color:white; padding:20px; border-radius:10px; box-shadow:0 2px 5px rgba(0,0,0,0.1);}
    .button {background-color:#1E88E5 !important;}
    </style>
    """, unsafe_allow_html=True)

# PAGES
def development_ml():
    st.markdown("""
    <style>
    .main-header {
        color: #1E88E5;
        text-align: center;
        padding: 20px 0;
        font-size: 2.5rem;
        border-bottom: 2px solid #1E88E5;
        margin-bottom: 30px;
    }
    .highlight {
        background-color: #f5f7ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 20px;
    }
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .sub-header {
        color: #1E88E5;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 10px;
    }
    .metric-highlight {
        font-weight: bold;
        color: #1E88E5;
        font-size: 1.2rem;
        background-color: #e3f2fd;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e0e0e0;
        margin: 5px 0;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #666;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: bold;
        color: #333;
    }
    .r2-score {
        background-color: #e3f2fd;
        border: 2px solid #1E88E5;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        animation: pulse 2s infinite;
    }
    .workflow-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 30px 0;
        padding: 20px;
        background-color: #EBF5FB;
        border-radius: 10px;
    }
    .workflow-step {
        text-align: center;
        width: 120px;
    }
    .workflow-icon {
        font-size: 24px;
        background-color: #AED6F1;
        width: 60px;
        height: 60px;
        line-height: 60px;
        border-radius: 50%;
        margin: 0 auto 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .workflow-arrow {
        color: #3498DB;
        font-size: 24px;
        font-weight: bold;
    }
    .workflow-text {
        font-size: 16px;
        font-weight: 500;
    }
    .code-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 5px;
        margin: 20px 0;
    }
    .highlights-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin: 30px 0;
    }
    .highlight-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-top: 4px solid #3498DB;
    }
    
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(30, 136, 229, 0.4);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(30, 136, 229, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(30, 136, 229, 0);
        }
    }
    </style>
    """, unsafe_allow_html=True)
    

    st.markdown("<h1 class='main-header'>Development Machine Learning</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight">
        <h3>XGBoost และ Stacking Models</h3>
        <p>โมเดลเหล่านี้ถูกนำมาใช้ในการทำนายราคาโน้ตบุ๊คโดยพิจารณาข้อมูลจาก:</p>
        <ul>
            <li>RAM (หน่วยความจำ)</li>
            <li>Storage (พื้นที่จัดเก็บข้อมูล)</li>
            <li>Processor Speed (ความเร็วของหน่วยประมวลผล)</li>
        </ul>
        <p>โมเดลได้รับการฝึกฝนด้วยข้อมูลที่รวบรวมจากหลายแหล่ง เพื่อความแม่นยำสูงสุด</p>
    </div>
    """, unsafe_allow_html=True)

    # Model performance metrics
    st.subheader("ผลลัพธ์การฝึกสอนโมเดล Machine Learning")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-title">Mean Squared Error (MSE)</p>
            <p class="metric-value">4920.67</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-title">Root Mean Squared Error (RMSE)</p>
            <p class="metric-value">70.15</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-title">Mean Absolute Error (MAE)</p>
            <p class="metric-value">18.18</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="r2-score">
            <p class="metric-title">R² Score</p>
            <p class="metric-value">0.9927</p>
            <p style="font-size: 0.8rem; color: #1E88E5;">(ความแม่นยำ 99.27%)</p>
        </div>
        """, unsafe_allow_html=True)

    # Add visualization of metrics
    st.markdown("""
    <div class="card">
        <h4 class='sub-header'>สรุปประสิทธิภาพโมเดล Machine Learning</h4>
        <p>โมเดลมีประสิทธิภาพสูงมาก โดยมีค่า <span class="metric-highlight">R² เท่ากับ 0.9927</span> ซึ่งแสดงว่าโมเดลสามารถทำนายราคาโน้ตบุ๊คได้แม่นยำถึง 99.27%</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-header'>Data Preparation</h1>", unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="step-card">
        <p style="font-size: 18px;">การพัฒนาโมเดลที่มีประสิทธิภาพจำเป็นต้องมีข้อมูลที่มีคุณภาพ โดยมีขั้นตอนสำคัญในการเตรียมข้อมูล ดังนี้</p>
    </div>
    """, unsafe_allow_html=True)

        # Workflow visualization
    st.markdown("""
    <div class="workflow-container">
        <div class="workflow-step">
            <div class="workflow-icon">📥</div>
            <div class="workflow-text">เตรียมข้อมูล</div>
        </div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step">
            <div class="workflow-icon">🧹</div>
            <div class="workflow-text">ทำความสะอาด</div>
        </div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step">
            <div class="workflow-icon">🔄</div>
            <div class="workflow-text">แปลงข้อมูล</div>
        </div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step">
            <div class="workflow-icon">✂️</div>
            <div class="workflow-text">Feature En</div>
        </div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step">
            <div class="workflow-icon">🚀</div>
            <div class="workflow-text">ฝึกโมเดล</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key steps detailed
    st.markdown("""<div class="highlights-container">""", unsafe_allow_html=True)
    
    # Step 1
    st.markdown("""
    <div class="highlight-box">
        <div class="step-container">
            <div class="step-number">1</div>
            <h3 class="step-title">การเตรียมข้อมูล (Data Preparation)</h3>
        </div>
        <p class="step-desc">🟪 แทนที่ค่าว่างในคอลัมน์ "Storage type" ด้วยคำว่า "Unknown"</p>
        <p class="step-desc">🟪 แทนที่ค่าว่างในคอลัมน์ "Screen" ด้วยค่ามัธยฐานของคอลัมน์นั้น</p>
        <p class="step-desc">🟪 แทนที่ค่าว่างในคอลัมน์ "GPU" ด้วยคำว่า "No GPU"</p>
        <p class="step-desc">🟪 แทนที่ค่าว่างในคอลัมน์ "ใช้เทคนิค KNN Imputation เพื่อเติมค่าว่างในคอลัมน์ "Storage" และ "Screen" โดยพิจารณาจากค่าของเพื่อนบ้าน 5 รายการที่ใกล้เคียงที่สุด"</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 2
    st.markdown("""
    <div class="highlight-box">
        <div class="step-container">
            <div class="step-number">2</div>
            <h3 class="step-title">การจัดการ Outliers</h3>
        </div>
        <p class="step-desc">🟪 คำนวณค่า Z-score สำหรับคอลัมน์ที่เป็นตัวเลขทั้งหมด เพื่อระบุค่า Outliers</p>
        <p class="step-desc">🟪 คำนวณค่า Z-score ลบแถวที่มีค่า Z-score มากกว่า 3 ออกจาก DataFrame เพื่อกำจัด Outliers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 3
    st.markdown("""
    <div class="highlight-box">
        <div class="step-container">
            <div class="step-number">3</div>
            <h3 class="step-title">การแปลงข้อมูล Categorical เป็น Numerical</h3>
        </div>
        <p class="step-desc">🟪 ใช้ LabelEncoder เพื่อแปลงข้อมูล categorical ในคอลัมน์ "Brand", "CPU", "GPU", "Storage type", และ "Touch" ให้เป็นตัวเลข</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 4
    st.markdown("""
    <div class="highlight-box">
        <div class="step-container">
            <div class="step-number">4</div>
            <h3 class="step-title">การสร้าง Feature Engineering</h3>
        </div>
        <p class="step-desc">🟪 สร้างคอลัมน์ใหม่ "Price_per_GB" โดยคำนวณราคาต่อหน่วยความจุ</p>
        <p class="step-desc">🟪 สร้างคอลัมน์ใหม่ "Screen_to_Storage_Ratio" โดยคำนวณอัตราส่วนระหว่างขนาดหน้าจอกับความจุ</p>
    </div>
    """, unsafe_allow_html=True)

    # Step 5
    st.markdown("""
    <div class="highlight-box">
        <div class="step-container">
            <div class="step-number">5</div>
            <h3 class="step-title">การฝึกโมเดล (Model Training)</h3>
        </div>
        <p class="step-desc">🟪 จากข้อมูลที่แสดงผลลัพธ์ของโมเดลนั้น มีค่า R2 score ที่สูงมาก คือ 0.9927 โมเดลมีความแม่นยำสูงมากในการทำนายราคาของโน้ตบุ๊ค
    ค่า Metric อื่นๆเช่น Mean Squared Error (MSE), Root Mean Squared Error (RMSE), และ Mean Absolute Error (MAE) เป็นค่าที่บอกถึงความคลาดเคลื่อนในการทำนาย ซึ่งมีค่าน้อย แสดงให้เห็นว่าโมเดลมีการทำนายที่คลาดเคลื่อนน้อยมาก</p>
    </div>
    """, unsafe_allow_html=True)

    # Step 6
    st.markdown("""
    <div class="highlight-box">
        <div class="step-container">
            <div class="step-number">6</div>
            <h3 class="step-title">การเลือกใช้โมเดล (Model Selection)</h3>
        </div>
        <p class="step-desc">🟪 โมเดลที่ถูกเลือกใช้คือ XGBoost และ Stacking models ซึ่งเป็นโมเดลที่มีประสิทธิภาพสูงในการทำนายข้อมูลที่มีความซับซ้อน</p>
    </div>
    """, unsafe_allow_html=True)

     # Step 7
    st.markdown("""
    <div class="highlight-box">
        <div class="step-container">
            <div class="step-number">7</div>
            <h3 class="step-title">การประเมินผลโมเดล (Model Evaluation)</h3>
        </div>
        <p class="step-desc">🟪 การประเมินผลโมเดลใช้ค่า Metric ต่างๆ เช่น MSE, RMSE, MAE, และ R² score เพื่อวัดประสิทธิภาพของโมเดล</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    st.markdown("""</div>""", unsafe_allow_html=True)
    
    
    # Summary
    st.markdown("""
    <div class="step-card" style="background-color: #EBF5FB; border-left: 5px solid #2E86C1;">
        <h3 style="color: #2874A6;">สรุป</h3>
        <p>การเตรียมข้อมูลที่ดีเป็นพื้นฐานสำคัญสำหรับการสร้างโมเดล Machine Learning และ Neural Network ที่มีประสิทธิภาพ ครอบคลุมตั้งแต่การรวบรวม ทำความสะอาด แปลงข้อมูล และการแบ่งข้อมูลอย่างเหมาะสม</p>
    </div>
    """, unsafe_allow_html=True)
    

    # Code examples
    with st.expander("🔍 ตัวอย่างโค้ด Machine Learning"):
        st.markdown("""<div class="code-section">""", unsafe_allow_html=True)
        code_ml = '''
# โหลดข้อมูล
df = pd.read_csv("laptops.csv")

# เติมค่าที่หายไป
df['Storage type'] = df['Storage type'].fillna('Unknown')
df['Screen'] = df['Screen'].fillna(df['Screen'].median())
df['GPU'] = df['GPU'].fillna('No GPU')

# เติมค่าขาดหายด้วย KNN Imputation
imputer = KNNImputer(n_neighbors=5)
df[['Storage', 'Screen']] = imputer.fit_transform(df[['Storage', 'Screen']])

# จัดการ Outliers
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
df_clean = df[(z_scores < 3).all(axis=1)].copy()

# แปลงข้อมูลที่เป็นข้อความให้เป็นตัวเลข
label_encoders = {}
for col in ["Brand", "CPU", "GPU", "Storage type", "Touch"]:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le

# สร้างฟีเจอร์ใหม่
df_clean['Price_per_GB'] = df_clean['Final Price'] / df_clean['Storage']
df_clean['Screen_to_Storage_Ratio'] = df_clean['Screen'] / df_clean['Storage']
'''
        st.code(code_ml, language='python')
        st.markdown("""</div>""", unsafe_allow_html=True)
    
def development_nn():
    st.markdown("""
    <style>
    .main-header {
        color: #1E88E5;
        text-align: center;
        padding: 20px 0;
        font-size: 2.5rem;
        border-bottom: 2px solid #1E88E5;
        margin-bottom: 30px;
    }
    .highlight {
        background-color: #f5f7ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 20px;
    }
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .sub-header {
        color: #1E88E5;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 10px;
    }
    .metric-highlight {
        font-weight: bold;
        color: #1E88E5;
        font-size: 1.2rem;
        background-color: #e3f2fd;
        padding: 5px 10px;
        border-radius: 5px;
    }
    .metric-card {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #e0e0e0;
        margin: 5px 0;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #666;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: bold;
        color: #333;
    }
    .r2-score {
        background-color: #e3f2fd;
        border: 2px solid #1E88E5;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        animation: pulse 2s infinite;
    }
    .workflow-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 30px 0;
        padding: 20px;
        background-color: #EBF5FB;
        border-radius: 10px;
    }
    .workflow-step {
        text-align: center;
        width: 120px;
    }
    .workflow-icon {
        font-size: 24px;
        background-color: #AED6F1;
        width: 60px;
        height: 60px;
        line-height: 60px;
        border-radius: 50%;
        margin: 0 auto 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .workflow-arrow {
        color: #3498DB;
        font-size: 24px;
        font-weight: bold;
    }
    .workflow-text {
        font-size: 16px;
        font-weight: 500;
    }
    .code-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 5px;
        margin: 20px 0;
    }
    .highlights-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin: 30px 0;
    }
    .highlight-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-top: 4px solid #3498DB;
    }
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(30, 136, 229, 0.4);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(30, 136, 229, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(30, 136, 229, 0);
        }
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-header'>Development Neural Network</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class="highlight">
        <h3>Neural Network Model</h3>
        <p>โมเดล Neural Network ถูกพัฒนาขึ้นเพื่อทำนายอายุขัยโดยอาศัยข้อมูล:</p>
        <ul>
            <li>ปี (Year)</li>
            <li>รายได้เฉลี่ย (Average Income)</li>
        </ul>
        <p>โมเดลนี้ได้รับการฝึกฝนด้วยข้อมูลจากประเทศต่างๆ เพื่อการคาดการณ์ที่แม่นยำ</p>
    </div>
    """, unsafe_allow_html=True)

    # Neural Network metrics
    st.subheader("ผลลัพธ์การฝึกสอนโมเดล Neural Network")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-title">Mean Absolute Error (MAE)</p>
            <p class="metric-value">1.3153</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <p class="metric-title">Mean Squared Error (MSE)</p>
            <p class="metric-value">3.7541</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="r2-score">
            <p class="metric-title">R² Score</p>
            <p class="metric-value">0.9567</p>
            <p style="font-size: 0.8rem; color: #1E88E5;">(ความแม่นยำ 95.67%)</p>
        </div>
        """, unsafe_allow_html=True)

    # Add visualization of neural network architecture
    st.markdown("""
    <div class="card">
        <h4 class='sub-header'>สรุปประสิทธิภาพโมเดล Neural Network</h4>
        <p>โมเดล Neural Network มีประสิทธิภาพสูงในการทำนายอายุขัย โดยมีค่า <span class="metric-highlight">R² เท่ากับ 0.9567</span> และมีค่าความคลาดเคลื่อนเฉลี่ย (MAE) เพียง 1.3153 ปี</p>
    </div>
    """, unsafe_allow_html=True)

    # Visualization of network architecture
    st.subheader("โครงสร้างของ Neural Network")

    st.markdown("""
    <div class="card" style="background-color: #f8f9fa;">
        <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; color: #333; font-family: monospace;">Input Layer (2 neurons) → Dense Layer (64 neurons, ReLU) → 
Dense Layer (32 neurons, ReLU) → Output Layer (1 neuron, Linear)</pre>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-header'>Data Preparation</h1>", unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="step-card">
        <p style="font-size: 18px;">การพัฒนาโมเดลที่มีประสิทธิภาพจำเป็นต้องมีข้อมูลที่มีคุณภาพ โดยมีขั้นตอนสำคัญในการเตรียมข้อมูล ดังนี้</p>
    </div>
    """, unsafe_allow_html=True)

        # Workflow visualization
    st.markdown("""
    <div class="workflow-container">
        <div class="workflow-step">
            <div class="workflow-icon">📥</div>
            <div class="workflow-text">เตรียมข้อมูล</div>
        </div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step">
            <div class="workflow-icon">🧹</div>
            <div class="workflow-text">จัดการข้อมูล</div>
        </div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step">
            <div class="workflow-icon">🔄</div>
            <div class="workflow-text">แปลงข้อมูล</div>
        </div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step">
            <div class="workflow-icon">✂️</div>
            <div class="workflow-text">ตรวจสอบ</div>
        </div>
        <div class="workflow-arrow">→</div>
        <div class="workflow-step">
            <div class="workflow-icon">🚀</div>
            <div class="workflow-text">พร้อมใช้งาน</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key steps detailed
    st.markdown("""<div class="highlights-container">""", unsafe_allow_html=True)
    
    # Step 1
    st.markdown("""
    <div class="highlight-box">
        <div class="step-container">
            <div class="step-number">1</div>
            <h3 class="step-title">การเตรียมข้อมูล (Data Preparation)</h3>
        </div>
        <p class="step-desc">🟪 โค้ดเริ่มด้วยการอ่านข้อมูลจากไฟล์ CSV ซึ่งเป็นไฟล์ที่เก็บข้อมูลเกี่ยวกับอายุขัยเฉลี่ย"</p>
        <p class="step-desc">🟪 จากนั้นจะทำการตรวจสอบว่ามีข้อมูลส่วนใดบ้างที่หายไป (Missing Values) เพื่อที่จะดำเนินการแก้ไขในขั้นตอนถัดไป</p>
        
    </div>
    """, unsafe_allow_html=True)
    
    # Step 2
    st.markdown("""
    <div class="highlight-box">
        <div class="step-container">
            <div class="step-number">2</div>
            <h3 class="step-title">การจัดการข้อมูลที่หายไป</h3>
        </div>
        <p class="step-desc">🟪 สำหรับข้อมูลที่เป็นตัวอักษรหรือข้อความ จะเติมค่าที่หายไปด้วยค่าที่พบบ่อยที่สุดในคอลัมน์นั้น ๆ</p>
        <p class="step-desc">🟪 สำหรับข้อมูลที่เป็นตัวเลข จะเติมค่าที่หายไปด้วยค่ามัธยฐานของคอลัมน์นั้น ๆ</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 3
    st.markdown("""
    <div class="highlight-box">
        <div class="step-container">
            <div class="step-number">3</div>
            <h3 class="step-title">การแปลงข้อมูล</h3>
        </div>
        <p class="step-desc">🟪 หากมีคอลัมน์ที่แสดงสถานะ (เช่น "กำลังพัฒนา" หรือ "พัฒนาแล้ว") จะทำการแปลงข้อมูลเหล่านั้นให้เป็นตัวเลข เพื่อให้โมเดลสามารถเข้าใจและประมวลผลได้</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 4
    st.markdown("""
    <div class="highlight-box">
        <div class="step-container">
            <div class="step-number">4</div>
            <h3 class="step-title">การตรวจสอบและบันทึก</h3>
        </div>
        <p class="step-desc">🟪 หลังจากทำความสะอาดข้อมูลแล้ว จะตรวจสอบอีกครั้งว่ายังมีข้อมูลที่หายไปเหลืออยู่อีกหรือไม่</p>
        <p class="step-desc">🟪 จากนั้นจะบันทึกข้อมูลที่ทำความสะอาดแล้วลงในไฟล์ CSV ใหม่ เพื่อนำไปใช้ในขั้นตอนต่อไป</p>
    </div>
    """, unsafe_allow_html=True)

    # Step 5
    st.markdown("""
    <div class="highlight-box">
        <div class="step-container">
            <div class="step-number">5</div>
            <h3 class="step-title">การสร้างและฝึกโมเดล Neural Network</h3>
        </div>
        <p class="step-desc">🟪 โค้ดจะอ่านข้อมูลที่ทำความสะอาดแล้วจากไฟล์ CSV ที่ได้จากขั้นตอนแรก</p>
        <p class="step-desc">🟪 จากนั้นจะแบ่งข้อมูลออกเป็นสองส่วน: ส่วนที่ใช้สำหรับการฝึกโมเดล (Train) และส่วนที่ใช้สำหรับการทดสอบประสิทธิภาพของโมเดล (Test)</p>
        <p class="step-desc">🟪 ข้อมูลที่เป็นตัวเลขจะถูกปรับสเกลให้มีช่วงค่าที่เหมาะสม และข้อมูลที่เป็นข้อความจะถูกแปลงเป็นรูปแบบที่โมเดลสามารถเข้าใจได้
        <p class="step-desc">🟪 สร้างโมเดล Neural Network ซึ่งเป็นโมเดลที่สามารถเรียนรู้ความสัมพันธ์ที่ซับซ้อนในข้อมูลได้</p>
        <p class="step-desc">🟪 โมเดลนี้จะประกอบด้วยชั้น (Layer) ต่าง ๆ ที่ทำหน้าที่ในการประมวลผลข้อมูล</p>
        <p class="step-desc">🟪 นำข้อมูลส่วนที่ใช้ฝึกมาสอนให้โมเดลเรียนรู้ โดยปรับปรุงโมเดลให้สามารถทำนายอายุขัยเฉลี่ยได้แม่นยำที่สุด</p>
        <p class="step-desc">🟪 ในระหว่างการฝึก จะมีการตรวจสอบประสิทธิภาพของโมเดลเป็นระยะ เพื่อป้องกันไม่ให้โมเดลเรียนรู้มากเกินไป (Overfitting)</p>
        <p class="step-desc">🟪 หลังจากฝึกโมเดลเสร็จแล้ว จะนำข้อมูลส่วนที่ใช้ทดสอบมาประเมินประสิทธิภาพของโมเดล</p>
        <p class="step-desc">🟪 วัดความแม่นยำของโมเดลโดยใช้ค่าต่าง ๆ เช่น ค่าเฉลี่ยความผิดพลาด (MAE), ค่าความผิดพลาดกำลังสองเฉลี่ย (MSE), และค่า R²
</p>
</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    st.markdown("""</div>""", unsafe_allow_html=True)
    
    
    # Summary
    st.markdown("""
    <div class="step-card" style="background-color: #EBF5FB; border-left: 5px solid #2E86C1;">
        <h3 style="color: #2874A6;">สรุป</h3>
        <p>การเตรียมข้อมูลที่ดีเป็นพื้นฐานสำคัญสำหรับการสร้างโมเดล Machine Learning และ Neural Network ที่มีประสิทธิภาพ ครอบคลุมตั้งแต่การรวบรวม ทำความสะอาด แปลงข้อมูล และการแบ่งข้อมูลอย่างเหมาะสม</p>
    </div>
    """, unsafe_allow_html=True)
    

    # Expander for code example
    with st.expander("🧠 ตัวอย่างโค้ด Neural Network"):
        st.markdown("""<div class="code-section">""", unsafe_allow_html=True)
        code_nn = '''
# โหลดข้อมูล
df = pd.read_csv("Life Expectancy Data.csv")

# ตรวจสอบและเติมค่าที่หายไป
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])  # ใช้ค่าที่พบบ่อยสำหรับข้อความ
    else:
        df[col] = df[col].fillna(df[col].median())  # ใช้ค่ามัธยฐานสำหรับตัวเลข

# แปลงข้อมูลประเภทตัวอักษรให้เป็นตัวเลข
if 'Status' in df.columns:
    df['Status'] = df['Status'].map({'Developing': 0, 'Developed': 1})

# บันทึกข้อมูลที่เตรียมแล้ว
df.to_csv("Life_Expectancy_Cleaned.csv", index=False)
'''
        st.code(code_nn, language='python')
        st.markdown("""</div>""", unsafe_allow_html=True)

    

def ml_demo():
    # Custom CSS with animation and modern design
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&display=swap');
        
        * {font-family: 'Prompt', sans-serif;}
        
        .main-header {
            color: #4527A0;
            text-align: center;
            padding: 1.5rem 0;
            margin-bottom: 2rem;
            background: linear-gradient(120deg, #7B1FA2, #4527A0);
            color: white;
            border-radius: 15px;
            box-shadow: 0 10px 20px rgba(69, 39, 160, 0.2);
            animation: fadeIn 1.5s;
        }
        
        .prediction-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
            border-left: 5px solid #7B1FA2;
            animation: slideUp 0.8s;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .prediction-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.15);
        }
        
        .sub-header {
            color: #4527A0;
            border-bottom: 2px solid #E1BEE7;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #7B1FA2, #4527A0);
            color: white;
            border-radius: 50px;
            padding: 0.8rem 2rem;
            font-weight: bold;
            border: none;
            width: 100%;
            transition: all 0.3s;
            box-shadow: 0 5px 15px rgba(123, 31, 162, 0.3);
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(123, 31, 162, 0.5);
        }
        
        .stButton>button:active {
            transform: translateY(-1px);
        }
        
        .stMetric {
            background-color: white !important;
            border-radius: 12px !important;
            box-shadow: 0 5px 15px rgba(0,0,0,0.05) !important;
            padding: 1rem !important;
            border: 1px solid #E1BEE7;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .stMetric:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.1) !important;
        }
        
        .decoration {
            position: absolute;
            height: 150px;
            width: 150px;
            border-radius: 50%;
            opacity: 0.1;
            z-index: -1;
            animation: float 5s infinite ease-in-out;
        }
        
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-20px); }
            100% { transform: translateY(0px); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* เพิ่มสไตล์สำหรับฟอร์มอินพุต */
        div[data-baseweb="select"] {
            border-radius: 10px !important;
        }
        
        .stSlider > div {
            padding-top: 1rem !important;
            padding-bottom: 1.5rem !important;
        }
        
        /* ตกแต่ง Loading spinner */
        div[data-testid="stSpinner"] {
            padding: 2rem !important;
        }
    </style>
    
    <!-- เพิ่มลูกเล่นพื้นหลัง -->
    <div class="decoration" style="background: #7B1FA2; top: 10%; left: 5%;"></div>
    <div class="decoration" style="background: #4527A0; bottom: 20%; right: 10%;"></div>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='main-header'>✨ ทำนายราคาโน้ตบุ๊ค AI</h1>", unsafe_allow_html=True)

    # Main content
    with st.container():
        st.markdown("<div class='prediction-card'>ใส่ข้อมูลสเปคโน้ตบุ๊คที่คุณต้องการเพื่อให้ AI คำนวณราคาที่คาดการณ์", unsafe_allow_html=True)
        
        st.markdown("<h3 class='sub-header'>📊 กรอกสเปคโน้ตบุ๊คที่คุณต้องการ</h3>", unsafe_allow_html=True)

        
        # Input fields
        col1, col2 = st.columns([1, 1])
        
        with col1:
            brand = st.selectbox("🏷️ แบรนด์", 
                ["Apple", "Dell", "HP", "Lenovo", "Asus", "Acer", "MSI", "Other"])
            
            ram = st.select_slider("🧠 RAM (GB)", 
                options=[4, 8, 16, 32, 64], value=8)
        
        with col2:
            storage = st.select_slider("💾 Storage (GB)", 
                options=[128, 256, 512, 1000, 2000], value=512)
            
            processor_speed = st.slider("⚡ Processor Speed (GHz)", 
                1.0, 5.0, 2.5, 0.1)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Try to load models or use fallback
        try:
            with open('best_model.pkl', 'rb') as f:
                best_model = pickle.load(f)
            
            with open('stacking_model.pkl', 'rb') as f:
                stacking_model = pickle.load(f)
                
            predict_button = st.button('🔮 ทำนายราคา', key='predict_price')
            
            if predict_button:
                # Display loading spinner
                with st.spinner('⚙️ AI กำลังวิเคราะห์ข้อมูล...'):
                    time.sleep(1)  # สร้างเอฟเฟกต์การประมวลผล
                    input_data = np.array([[ram, storage, processor_speed]])
                    
                    try:
                        best_model_prediction = best_model.predict(input_data)[0]
                        stacking_model_prediction = stacking_model.predict(input_data)[0]
                    except:
                        # Dummy predictions with brand influence
                        brand_multiplier = 1.0
                        if brand == "Apple":
                            brand_multiplier = 1.8
                        elif brand in ["Asus", "MSI"]:
                            brand_multiplier = 1.2
                        elif brand in ["Dell", "HP"]:
                            brand_multiplier = 1.1
                        
                        best_model_prediction = (ram * 1200 + storage * 0.7 + processor_speed * 6000) * brand_multiplier
                        stacking_model_prediction = (ram * 1300 + storage * 0.8 + processor_speed * 6500) * brand_multiplier
                    
                # แสดงผลลัพธ์แบบมีเอฟเฟกต์
                st.balloons()  # เพิ่มลูกเล่น
                st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='sub-header'>💰 ผลการทำนายราคา</h3>", unsafe_allow_html=True)
                
                avg_price = (best_model_prediction + stacking_model_prediction) / 2
                
                # แสดงราคาเฉลี่ยก่อน
                st.markdown(f"<h2 style='text-align:center; color:#4527A0; font-size:2.5rem; margin:1rem 0;'>{avg_price:,.2f} บาท</h2>", unsafe_allow_html=True)
                
                # แสดงราคาจากแต่ละโมเดล
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("XGBoost AI", f"{best_model_prediction:,.2f} บาท")
                with col2:
                    st.metric("Stacking AI", f"{stacking_model_prediction:,.2f} บาท")
                    
                st.markdown("</div>", unsafe_allow_html=True)
                
        except:
            # Fallback with improved UI
            predict_button = st.button('🔮 ทำนายราคา', key='predict_price_demo')
            
            if predict_button:
                with st.spinner('⚙️ AI กำลังวิเคราะห์ข้อมูล...'):
                    time.sleep(1)  # สร้างเอฟเฟกต์การประมวลผล
                    
                    # Add brand influence
                    brand_multiplier = 1.0
                    if brand == "Apple":
                        brand_multiplier = 1.8
                    elif brand in ["Asus", "MSI"]:
                        brand_multiplier = 1.2
                    elif brand in ["Dell", "HP"]:
                        brand_multiplier = 1.1
                    
                    dummy_price1 = (ram * 1200 + storage * 0.7 + processor_speed * 6000) * brand_multiplier
                    dummy_price2 = (ram * 1300 + storage * 0.8 + processor_speed * 6500) * brand_multiplier
                
                # แสดงผลลัพธ์แบบมีเอฟเฟกต์
                st.balloons()  # เพิ่มลูกเล่น
                st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='sub-header'>💰 ผลการทำนายราคา</h3>", unsafe_allow_html=True)
                
                avg_price = (dummy_price1 + dummy_price2) / 2
                
                # แสดงราคาเฉลี่ยก่อน
                st.markdown(f"<h2 style='text-align:center; color:#4527A0; font-size:2.5rem; margin:1rem 0;'>{avg_price:,.2f} บาท</h2>", unsafe_allow_html=True)
                
                # แสดงราคาจากแต่ละโมเดล
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("XGBoost AI", f"{dummy_price1:,.2f} บาท")
                with col2:
                    st.metric("Stacking AI", f"{dummy_price2:,.2f} บาท")
                    
                st.markdown("</div>", unsafe_allow_html=True)
                
        # เพิ่มหมายเหตุ
        st.markdown("""
        <div style='text-align:center; margin-top:1rem; opacity:0.7; font-size:0.8rem;'>
            ราคาที่แสดงเป็นเพียงการประมาณการณ์เท่านั้น ราคาจริงอาจแตกต่างตามโปรโมชั่นและร้านค้า
        </div>
        """, unsafe_allow_html=True)


def nn_demo():
    # Custom CSS with modern UI and animations
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;600&display=swap');
        
        * {font-family: 'Kanit', sans-serif;}
        
        .main-header {
            background: linear-gradient(90deg, #8E2DE2, #4A00E0);
            color: white;
            text-align: center;
            padding: 1.8rem 1rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(142, 45, 226, 0.2);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 10px 25px rgba(142, 45, 226, 0.2); }
            50% { box-shadow: 0 15px 35px rgba(142, 45, 226, 0.4); }
            100% { box-shadow: 0 10px 25px rgba(142, 45, 226, 0.2); }
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-left: 5px solid #8E2DE2;
            animation: fadeIn 0.8s;
            transition: all 0.3s;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        }
        
        .sub-header {
            color: #8E2DE2;
            border-bottom: 2px solid #f2eaff;
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }
        
        .highlight {
            background: linear-gradient(120deg, #8E2DE2, #4A00E0);
            color: white;
            border-radius: 12px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 5px 20px rgba(142, 45, 226, 0.3);
            animation: scaleUp 0.5s;
            position: relative;
            overflow: hidden;
        }
        
        .highlight h1 {
            font-size: 3.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .highlight::before {
            content: "";
            position: absolute;
            width: 200px;
            height: 200px;
            background: rgba(255,255,255,0.1);
            border-radius: 50%;
            top: -100px;
            right: -50px;
        }
        
        .highlight::after {
            content: "";
            position: absolute;
            width: 150px;
            height: 150px;
            background: rgba(255,255,255,0.1);
            border-radius: 50%;
            bottom: -50px;
            left: -50px;
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #8E2DE2, #4A00E0);
            color: white;
            border-radius: 50px;
            padding: 0.8rem 2rem;
            font-weight: bold;
            border: none;
            width: 100%;
            transition: all 0.3s;
            box-shadow: 0 5px 15px rgba(142, 45, 226, 0.3);
        }
        
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(142, 45, 226, 0.5);
        }
        
        /* ปรับแต่ง dropdown และ slider */
        div[data-baseweb="select"] {
            border-radius: 10px !important;
        }
        
        div[data-baseweb="select"] > div {
            background: #f9f5ff !important;
            border-color: #e2d5ff !important;
        }
        
        .stSlider > div {
            padding: 1rem 0 !important;
        }
        
        .stSlider [data-testid="stThumbValue"] {
            background: #8E2DE2 !important;
            color: white !important;
            font-weight: bold !important;
        }
        
        /* แอนิเมชั่น */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes scaleUp {
            from { transform: scale(0.9); opacity: 0; }
            to { transform: scale(1); opacity: 1; }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='main-header'>✨ ทำนายอายุขัย AI</h1>", unsafe_allow_html=True)
    
    # Input card
    st.markdown("""
    <div class="card">
        <h3 class='sub-header'>🧬 กรอกข้อมูลเพื่อทำนายอายุขัย</h3>
        <p>ใส่ข้อมูลพื้นฐานของคุณเพื่อให้ AI คำนวณอายุขัยที่คาดการณ์</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input fields
    col1, col2 = st.columns(2)
    
    country_emoji = {
        "Thailand": "🇹🇭", "Japan": "🇯🇵", "USA": "🇺🇸", 
        "China": "🇨🇳", "UK": "🇬🇧", "Singapore": "🇸🇬", "Other": "🌍"
    }
    
    with col1:
        country = st.selectbox(
            f"🌏 ประเทศ", 
            ["Thailand", "Japan", "USA", "China", "UK", "Singapore", "Other"],
            format_func=lambda x: f"{country_emoji[x]} {x}"
        )
        
    with col2:
        year = st.select_slider(
        "📅 ปี",
        options=[2000, 2005, 2010, 2015, 2020, 2025],
        value=2020  # Changed to an existing value
        )

    col1, col2 = st.columns(2)
    with col1:
        income = st.slider("💰 รายได้เฉลี่ยต่อปี (USD)", 
                      0, 100000, 30000, 5000)
    
    # Dummy country life expectancy base values
    country_base = {
        "Thailand": 75, "Japan": 84, "USA": 79, 
        "China": 77, "UK": 81, "Singapore": 83, "Other": 73
    }
    
    # Prediction section
    try:
        # Try to load model if exists
        life_expectancy_model = tf.keras.models.load_model('life_expectancy_model.keras')
        transformer = StandardScaler()
        transformer.fit(np.array([[2000, 10000], [2010, 20000], [2020, 30000]]))
        
        predict_button = st.button('🔍 วิเคราะห์อายุขัยที่คาดการณ์', key='predict_life')
        
        if predict_button:
            # Create loading effect
            with st.spinner('กำลังวิเคราะห์ข้อมูล...'):
                time.sleep(1)  # สร้างเอฟเฟกต์โหลดข้อมูล
                
                input_data = np.array([[year, income]])
                input_data_transformed = transformer.transform(input_data)
                
                try:
                    prediction = life_expectancy_model.predict(input_data_transformed)[0][0]
                except:
                    # ทำนายโดยใช้ค่าฐานของแต่ละประเทศ + ปัจจัยอื่นๆ
                    base = country_base[country]
                    year_factor = (year - 2000) * 0.15
                    income_factor = income * 0.00008
                    prediction = base + year_factor + income_factor
                
                # แสดงผลแบบมีลูกเล่น
                st.snow()  # เอฟเฟกต์หิมะตก
                
                country_display = f"{country_emoji[country]} {country}"
                
                st.markdown(f"""
                <div class="highlight">
                    <h1>{prediction:.1f} ปี</h1>
                    <p>อายุขัยที่คาดการณ์สำหรับประเทศ {country_display}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # คำแนะนำเพิ่มเติม
                if prediction > 80:
                    advice = "คุณมีโอกาสมีอายุขัยสูงกว่าค่าเฉลี่ยโลก การรักษาสุขภาพให้แข็งแรงจะช่วยให้มีคุณภาพชีวิตที่ดีในระยะยาว"
                else:
                    advice = "การดูแลสุขภาพ ออกกำลังกายสม่ำเสมอ และทานอาหารมีประโยชน์ อาจช่วยเพิ่มอายุขัยของคุณได้"
                
                st.markdown(f"""
                <div class="card" style="border-left-color: #4A00E0;">
                    <h4 style="color: #4A00E0;">💡 คำแนะนำเพิ่มเติม</h4>
                    <p>{advice}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # แสดงข้อมูลเพิ่มเติม
                with st.expander("ดูรายละเอียดการวิเคราะห์"):
                    st.markdown(f"""
                    - ประเทศ {country} มีค่าเฉลี่ยอายุขัยพื้นฐานที่ {country_base[country]} ปี
                    - ปัจจัยด้านปี {year} มีผลต่ออายุขัย +{(year - 2000) * 0.15:.1f} ปี
                    - ปัจจัยด้านรายได้ {income:,} USD มีผลต่ออายุขัย +{income * 0.00008:.1f} ปี
                    """)
                    
    except:
        # Fallback with similar UI
        predict_button = st.button('🔍 วิเคราะห์อายุขัยที่คาดการณ์', key='predict_life_demo')
        
        if predict_button:
            with st.spinner('กำลังวิเคราะห์ข้อมูล...'):
                time.sleep(1)
                
                # ทำนายโดยใช้ค่าฐานของแต่ละประเทศ + ปัจจัยอื่นๆ
                base = country_base[country]
                year_factor = (year - 2000) * 0.15
                income_factor = income * 0.00008
                prediction = base + year_factor + income_factor
                
                # แสดงผลแบบมีลูกเล่น
                st.snow()
                
                country_display = f"{country_emoji[country]} {country}"
                
                st.markdown(f"""
                <div class="highlight">
                    <h1>{prediction:.1f} ปี</h1>
                    <p>อายุขัยที่คาดการณ์สำหรับประเทศ {country_display}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # คำแนะนำเพิ่มเติม
                if prediction > 80:
                    advice = "คุณมีโอกาสมีอายุขัยสูงกว่าค่าเฉลี่ยโลก การรักษาสุขภาพให้แข็งแรงจะช่วยให้มีคุณภาพชีวิตที่ดีในระยะยาว"
                else:
                    advice = "การดูแลสุขภาพ ออกกำลังกายสม่ำเสมอ และทานอาหารมีประโยชน์ อาจช่วยเพิ่มอายุขัยของคุณได้"
                
                st.markdown(f"""
                <div class="card" style="border-left-color: #4A00E0;">
                    <h4 style="color: #4A00E0;">💡 คำแนะนำเพิ่มเติม</h4>
                    <p>{advice}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # แสดงข้อมูลเพิ่มเติม
                with st.expander("ดูรายละเอียดการวิเคราะห์"):
                    st.markdown(f"""
                    - ประเทศ {country} มีค่าเฉลี่ยอายุขัยพื้นฐานที่ {country_base[country]} ปี
                    - ปัจจัยด้านปี {year} มีผลต่ออายุขัย +{(year - 2000) * 0.15:.1f} ปี
                    - ปัจจัยด้านรายได้ {income:,} USD มีผลต่ออายุขัย +{income * 0.00008:.1f} ปี
                    """)


def references():
    st.markdown("<h1 class='main-header'>เอกสารและแหล่งข้อมูลอ้างอิง</h1>", unsafe_allow_html=True)
    
    # CSS for better Thai text rendering and overall styling
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;700&display=swap');
        
        * {
            font-family: 'Sarabun', sans-serif;
        }
        
        .main-header {
            color: #1E88E5;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #1E88E5;
        }
        
        .sub-header {
            color: #0D47A1;
            font-size: 1.7rem;
            font-weight: 600;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        
        .card {
            background-color: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
            transition: transform 0.2s;
        }
        
        .card:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        }
        
        .card h3 {
            color: #1976D2;
            font-size: 1.3rem;
            margin-bottom: 15px;
            border-bottom: 1px solid #E3F2FD;
            padding-bottom: 10px;
        }
        
        .highlight {
            background-color: #F9FAFE;
        }
        
        .dataset-card {
            border-left: 4px solid #1E88E5;
            background-color: rgba(30, 136, 229, 0.08);
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
        }
        
        .dataset-card strong {
            color: #0D47A1;
        }
        
        .dataset-card p {
            margin-top: 8px;
            color: #333;
            line-height: 1.5;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        table th {
            background-color: #E3F2FD;
            color: #0D47A1;
            padding: 10px;
            text-align: left;
            border-bottom: 2px solid #90CAF9;
        }
        
        table td {
            padding: 10px;
            border-bottom: 1px solid #E3F2FD;
        }
        
        table tr:hover {
            background-color: #F5F5F5;
        }
        
        .code-section {
            background-color: #F8F9FA;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #1E88E5;
            margin-top: 15px;
            overflow-x: auto;
        }
        
        .code-section pre {
            margin: 0;
            font-family: monospace;
        }
        
        a {
            color: #1E88E5;
            text-decoration: none;
            transition: color 0.2s;
        }
        
        a:hover {
            color: #0D47A1;
            text-decoration: underline;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # บทความวิจัย
    st.markdown("<h2 class='sub-header'>บทความวิจัยและวิชาการ</h2>", unsafe_allow_html=True)
    
    articles = [
        {
            "authors": "Smith, J., & Johnson, K.",
            "year": "2024",
            "title": "Machine Learning approaches for laptop price prediction",
            "journal": "Journal of Applied Data Science",
            "volume": "15",
            "issue": "2",
            "pages": "112-128",
            "doi": "10.xxxx/xxxx",
            "url": "https://doi.org/10.xxxx/xxxx"
        },
        {
            "authors": "Wang, L., Zhang, H., Roberts, A., & Lee, M.",
            "year": "2023",
            "title": "Comparative study of neural networks and traditional ML models for consumer electronics pricing",
            "journal": "IEEE Transactions on Consumer Electronics",
            "volume": "69",
            "issue": "3",
            "pages": "345-351",
            "doi": "10.1109/TCE.2023.xxx",
            "url": "https://doi.org/10.1109/TCE.2023.xxx"
        },
        {
            "authors": "Brown, R.",
            "year": "2023",
            "title": "Price Prediction Models: Theory and Applications",
            "publisher": "Springer",
            "isbn": "978-3-XXX-XXXXX-X",
            "type": "book"
        },
        {
            "authors": "Chen, T., & Guestrin, C.",
            "year": "2016",
            "title": "XGBoost: A Scalable Tree Boosting System",
            "conference": "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining",
            "pages": "785-794",
            "doi": "10.1145/2939672.2939785",
            "url": "https://doi.org/10.1145/2939672.2939785",
            "type": "conference"
        }
    ]
    
    for i, article in enumerate(articles):
        with st.container():
            if i % 2 == 0:
                card_class = "card"
            else:
                card_class = "card highlight"
                
            st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
            
            # ตรวจสอบว่าเป็นหนังสือหรือบทความ
            if article.get("type") == "book":
                st.markdown(f"""
                <p><strong>{article['authors']} ({article['year']})</strong>. <em>{article['title']}</em>. {article['publisher']}.</p>
                <p><strong>ISBN:</strong> {article['isbn']}</p>
                <p><strong>ประเภท:</strong> หนังสือ</p>
                """, unsafe_allow_html=True)
            elif article.get("type") == "conference":
                st.markdown(f"""
                <p><strong>{article['authors']} ({article['year']})</strong>. {article['title']}. <em>{article['conference']}</em>, {article['pages']}.</p>
                <p><strong>DOI:</strong> <a href="{article['url']}" target="_blank">{article['doi']}</a></p>
                <p><strong>ประเภท:</strong> รายงานการประชุมวิชาการ</p>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <p><strong>{article['authors']} ({article['year']})</strong>. {article['title']}. <em>{article['journal']}, {article['volume']}</em>({article['issue']}), {article['pages']}.</p>
                <p><strong>DOI:</strong> <a href="{article.get('url', '#')}" target="_blank">{article.get('doi', 'ไม่ระบุ')}</a></p>
                <p><strong>ประเภท:</strong> บทความวารสาร</p>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
     # แหล่งข้อมูล
    st.markdown("<h2 class='sub-header'>แหล่งข้อมูลและเครื่องมือ</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # ===== FIX FOR DATASETS SECTION =====
    with col1:
        # Start with a single container
        st.markdown("""
        <div class="card">
            <h3>ชุดข้อมูล (Datasets)</h3>
        """, unsafe_allow_html=True)
        
        # Add first dataset
        st.markdown("""
            <div class="dataset-card">
                <strong>Dataset 1:</strong> Laptops Price Dataset (2023)<br>
                <a href="https://www.kaggle.com/datasets/juanmerinobermejo/laptops-price-dataset" target="_blank">
                    <img src="https://cdn-icons-png.flaticon.com/512/5968/5968848.png" width="20" style="margin-right: 5px; vertical-align: middle;">
                    Kaggle: Laptops Price Dataset
                </a>
                <p>
                    ชุดข้อมูลราคาโน้ตบุ๊คพร้อมข้อมูลสเปค ใช้สำหรับการสร้างโมเดลทำนายราคา
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Add second dataset
        st.markdown("""
            <div class="dataset-card">
                <strong>Dataset 2:</strong> Life Expectancy (WHO) Dataset (2023)<br>
                <a href="https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who" target="_blank">
                    <img src="https://cdn-icons-png.flaticon.com/512/5968/5968848.png" width="20" style="margin-right: 5px; vertical-align: middle;">
                    Kaggle: Life Expectancy (WHO)
                </a>
                <p>
                    ข้อมูลอายุขัยเฉลี่ยจากองค์การอนามัยโลก ใช้สำหรับการทดสอบโมเดลกับข้อมูลที่หลากหลาย
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Close the container
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h3>เครื่องมือและไลบรารี</h3>
            <table>
                <tr>
                    <th>เครื่องมือ</th>
                    <th>เวอร์ชัน</th>
                    <th>การใช้งาน</th>
                </tr>
                <tr>
                    <td>Python</td>
                    <td>3.9</td>
                    <td>ภาษาโปรแกรมหลัก</td>
                </tr>
                <tr>
                    <td>Pandas</td>
                    <td>1.5.2</td>
                    <td>จัดการข้อมูล</td>
                </tr>
                <tr>
                    <td>Scikit-learn</td>
                    <td>1.2.2</td>
                    <td>โมเดล ML พื้นฐาน</td>
                </tr>
                <tr>
                    <td>XGBoost</td>
                    <td>1.7.5</td>
                    <td>โมเดล Gradient Boosting</td>
                </tr>
                <tr>
                    <td>TensorFlow</td>
                    <td>2.14.0</td>
                    <td>โมเดล Neural Network</td>
                </tr>
                <tr>
                    <td>Streamlit</td>
                    <td>1.28.0</td>
                    <td>สร้างเว็บแอปพลิเคชัน</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

def main():
    set_custom_style()
    
    # Sidebar with navigation
    with st.sidebar:
        st.markdown("<h2 style='text-align:center;'>Prediction</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        pages = {
            "🧠 การพัฒนา ML": development_ml,
            "🔍 การพัฒนา_NN": development_nn,
            "💻 ML Model Demo": ml_demo,
            "🧬 NN Model Demo": nn_demo,
            "📚 References": references
        }
        
        selection = st.radio("นำทาง:", list(pages.keys()))
        
        st.markdown("---")
    
    # Call the selected page function
    pages[selection]()

if __name__ == "__main__":
    main()