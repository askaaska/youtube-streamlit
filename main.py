import streamlit as st
import numpy as np
import pandas as pd
#from PIL import Image
import time

st.title("streamlit・超入門")

df = pd.DataFrame({
    '1列目':[1,2,3,4],
    '2列目':[10,20,30,40]
})

#いろいろな表の書き方
st.write(df)
st.dataframe(df.style.highlight_max(axis = 0),width=1000,height=1000)#タテヨコ指定できる
st.table(df.style.highlight_max(axis = 0))

""""
# 章
## 節
### 項

```python
import streamlit as st
import numpy as np
import pandas as pd
```

"""

df = pd.DataFrame(
    np.random.rand(20,3),
    columns = ['a','b','c']
)

st.line_chart(df)
st.bar_chart(df)

df = pd.DataFrame(
    np.random.rand(100, 2) / [50, 50] + [35.69,139.70],
    columns = ['lat','lon']
)

st.map(df)

#if st.checkbox('Show Image'):
#    st.write('Display Image')
#    img = Image.open('sample.jpg')
#    st.image(img,caption = 'TestIMAGE',use_column_width=True)

option = st.selectbox(
    'あなたが好きな数字を教えてください',
    list(range(1,11))
)

'あなたの好きな数字は',option,'です'

left_column, right_column = st.columns(2)
button = left_column.button('右カラムに文字を表示')
if button:
    right_column.write('ここは右カラム')

expander = st.expander('問い合わせ1')
expander.write('問い合せの回答1')
expander2 = st.expander('問い合わせ2')
expander2.write('問い合せの回答2')
expander3 = st.expander('問い合わせ3')
expander3.write('問い合せの回答3')


text = st.text_input('あなたの趣味をおしえてください')
'あなたの趣味：',text

condition = st.slider('あなたの今の調子は？', 0, 100, 50)
'コンディション：',condition

# text2 = st.sidebar.text_input('あなたの趣味をおしえてください')
# condition2 = st.sidebar.slider('あなたの今の調子は？', 0, 100, 50)
# 'あなたの趣味：',text2
# 'コンディション：',condition2

st.title("プログレスバーの表示")
'Start!!'

latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    latest_iteration.text(f'Interation {i+1}')
    bar.progress(i+1)
    time.sleep(0.1)
'Done!!'
