import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


st.title("초코볼의 무게 예측 :smile:")
st.write("초코볼의 무게를 측정하여 입력해봅시다.")

col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)

with col1:
    data2 = st.number_input(label="2", min_value=0.00, key=1)

with col2:
    data3 = st.number_input(label="3", min_value=0.00, key=2)
    
with col3:
    data4 = st.number_input(label="4", min_value=0.00, key=3)
    
with col4:
    data5 = st.number_input(label="5", min_value=0.00, key=4)
    
with col5:
    data7 = st.number_input(label="7", min_value=0.00, key=5)
    
with col6:
    data8 = st.number_input(label="8", min_value=0.00, key=6)
    
with col7:
    data9 = st.number_input(label="9", min_value=0.00, key=7)
    
with col8:
    data10 = st.number_input(label="10", min_value=0.00, key=8)
    
with col9:
    data11 = st.number_input(label="11",min_value=0.00, key=9)
    
with col10:
    data12 = st.number_input(label="12",min_value=0.00, key=10)

x = [2,3,4,5,7,8,9,10, 11, 12]
y = [data2, data3, data4, data5, data7, data8, data9, data10, data11, data12]
x_n = np.array(x).reshape(-1, 1)
y_n = np.array(y)
        
def errors(x, y, a, b):
    error=0
    for i in range(len(x)):
        predicted_y = a * x[i] + b
        error += (y[i] - predicted_y)**2
    return error

    
fig, ax = plt.subplots()
ax.scatter(x, y)
plt.ylim(0, 50)
ax.grid(True)

# 제목과 축 레이블 추가
ax.set_title('prediction weight of m&ms')
ax.set_xlabel('number')
ax.set_ylabel('weight')
ax.legend()

with st.sidebar:
    line = st.checkbox("선 그리기", key=21)

    if line:
        
        slope = st.slider("기울기", min_value=.0, max_value=5.0, value=1.0, step=0.1, key=11)
        bias = st.slider("y절편", min_value=5.0, max_value=15.0, value=10.0, step=0.1, key=12)
        error = round(errors(x, y, slope, bias),2)
        st.caption(f"오차:{error}")
        y_human = np.array(x_n*slope+bias)
        ax.plot(x_n, y_human, color='blue', label='인간이 그린 선')
        # 회귀 방정식을 그래프에 추가
        equation_text1 = f'Human prediction y = {slope:.2f}x + {bias:.2f}'
        ax.text(11, y_human[-2]+2, equation_text1, fontsize=8, color='black')
        
    line_r = st.checkbox("인공지능 선 그리기", key=22)
        
    if line_r:
        model = LinearRegression()
        model.fit(x_n, y_n)

        a = round(model.coef_[0],2)
        b = round(model.intercept_,2)
        
        # 예측값 계산
        
        y_pred = model.predict(x_n)

        # 결과 시각화
        ax.plot(x_n, y_pred, color='red', label='Fitted line')
        equation_text2 = f'AI prediction y = {a:.2f}x + {b:.2f}'
        ax.text(11, y_pred[-2]-2, equation_text2, fontsize=8, color='black')
        st.caption(f"y={a}x+{b}")
        error = round(errors(x, y, a, b),2)
        st.caption(f"오차:{error}")


st.pyplot(fig)
