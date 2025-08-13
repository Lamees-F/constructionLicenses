import streamlit as st
import pandas as pd
import joblib
from datetime import timedelta
import base64
from PIL import Image
import io

model_daily = joblib.load("xgb_permits_daily_model.pkl")
encoders = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="أمانة منطقة الجوف", layout="wide")

st.markdown("""
    <style>
        html, body, [class*="css"]  {
            direction: rtl;
            text-align: right;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stButton > button {
            width: 100%;
            background-color: #0099cc;
            color: white;
            font-weight: bold;
        }
        .info-box {
            background-color: #f0f0f5;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid #ccc;
            padding-right: 25px; 

        }
    </style>
""", unsafe_allow_html=True)



def get_base64_of_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

logo_base64 = get_base64_of_image("big_logo.png")

st.markdown(f"""
    <div style="display: flex; flex-direction: row-reverse; align-items: center; justify-content: space-between; margin-bottom: 20px;">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="height: 80px; margin-left: 20px;">
        <h1 style="margin: 0; font-size: 28px;"> توقع عدد طلبات الرخص الإنشائية في منطقة الجوف</h1>
    </div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("""<div class='info-box'>
    <h4> نبذة عن المشروع</h4>
    هذا التطبيق يتيح لك توقع عدد طلبات الرخص الإنشائية التي قد تصدرها أمانة منطقة الجوف بناءً على بيانات السنوات السابقة.
    <br><br>
    <strong>طريقة الاستخدام:</strong>
    <ul>
    <li>اختر تاريخ بداية التوقع.</li>
    <li>حدد الفترة (يومي، أسبوعي، شهري).</li>
    <li>حدد الخيارات الأخرى.</li>
    <li>اضغط على "توقع" للحصول على النتيجة.</li>
    </ul>
    <strong> مصدر البيانات:</strong> سجلات الرخص الإنشائية من أمانة منطقة الجوف.
    </div>
    """, unsafe_allow_html=True)

with st.form("prediction_form"):
    st.subheader(" بيانات التوقع")

    col1, col2, col3 = st.columns(3)

    with col1:
        start_date = st.date_input(" اختر تاريخ بداية التوقع")
    with col2:
        period = st.selectbox(" اختر فترة التوقع", ['يومي', 'أسبوعي', 'شهري'])
    with col3:
        municipality = st.selectbox(" اختر البلدية", encoders['البلدية'].classes_)

    col4, col5 = st.columns(2)

    with col4:
        request_type = st.selectbox(" الغرض", encoders['الغرض'].classes_)
    with col5:
        ownership_type = st.selectbox(" نوع سند الملكية", encoders['نوع سند الملكية'].classes_)

    submitted = st.form_submit_button(" توقع")

try:
    municipality_code = encoders['البلدية'].transform([municipality])[0]
    request_type = encoders['الغرض'].transform([request_type])[0]
    ownership_code = encoders['نوع سند الملكية'].transform([ownership_type])[0]
except Exception as e:
    st.error(f"خطأ في التشفير: {e}")
    st.stop()

def create_input_df(date):
    year = date.year
    month = date.month
    day = date.day
    week = date.isocalendar()[1]
    day_of_week = date.weekday() 

    data = {
        'البلدية': [municipality_code],
        'الغرض': [request_type],
        'نوع سند الملكية': [ownership_code],
        'سنة الطلب': [year],
        'شهر الطلب': [month],
        'أسبوع الطلب': [week],
        'يوم الطلب': [day],
        'يوم الأسبوع': [day_of_week]
    }
    return pd.DataFrame(data)

if submitted:
    if period == 'يومي':
        input_df = create_input_df(start_date)
        pred = model_daily.predict(input_df)[0]
        if pred < 0:
            st.warning("⚠️ لا توجد طلبات متوقعة لهذا اليوم.")
        else:
            st.success(f"🔹 التوقع ليوم {start_date.strftime('%Y-%m-%d')} هو **{round(pred)}** طلب.")
    
    else:
        if period == 'أسبوعي':
            days_to_predict = 7
            label = f"الأسبوع من {start_date.strftime('%Y-%m-%d')} إلى {(start_date + timedelta(days=6)).strftime('%Y-%m-%d')}"
        else:  
            next_month = start_date.replace(day=28) + timedelta(days=4)
            last_day = next_month - timedelta(days=next_month.day)
            days_to_predict = (last_day - start_date).days + 1
            label = f"الشهر من {start_date.strftime('%Y-%m-%d')} إلى {last_day.strftime('%Y-%m-%d')}"

        total_pred = 0
        daily_preds = []

        for i in range(days_to_predict):
            current_date = start_date + timedelta(days=i)
            input_df = create_input_df(current_date)
            pred = model_daily.predict(input_df)[0]
            daily_preds.append((current_date.strftime('%Y-%m-%d'), pred))
            total_pred += pred

        if total_pred < 0:
            st.warning("⚠️ لا توجد طلبات متوقعة لهذه الفترة.")
        else:
            st.success(f"🔹 إجمالي التوقع للفترة {label} هو **{round(total_pred)}** طلب.")

            with st.expander("📆 التفاصيل اليومية"):
                for date_str, val in daily_preds:
                    if val < 0:
                        st.write(f"{date_str}: لا توجد طلبات")
                    else:
                        st.write(f"{date_str}: {round(val)} طلب")
