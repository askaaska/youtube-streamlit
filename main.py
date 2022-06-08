import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.decomposition import PCA
import datetime
import streamlit as st
import base64
import plotly.graph_objects as go
import plotly.express as px

#ボタンの色の設定
button_css = f"""
<style>
  div.stButton > button:first-child  {{
    font-weight  : bold                ;/* 文字：太字                   */
    border       :  2px solid #f36     ;/* 枠線：ピンク色で5ピクセルの実線 */
    border-radius: 10px 10px 10px 10px ;/* 枠線：半径10ピクセルの角丸     */
    background   : #ffc             ;/* 背景色：淡い黄色(3桁のカラーコード参照https://fromkato.com/color/ff0000#hsl)*/
  }}
</style>
"""
# csv出力のリンク作成関数
def csv_link(a,b,c):
    csv = a.to_csv(index=False)
    b64 = base64.b64encode(csv.encode('utf-8-sig')).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download={c}>Download Link</a>'
    st.markdown(b + f"CSVファイルのダウンロード(utf-8 BOM):  {href}", unsafe_allow_html=True)
#サイドバーの作成
st.title("主成分分析 アプリケーション")
st.sidebar.header('ご利用ありがとうございます😊')
st.sidebar.text('実行手順')
st.sidebar.text('STEP0 : csvファイルを作成')
input_template = pd.read_csv("input_template.csv")
st.sidebar.write('<span style="color:gray"><font size="1"> [テンプレート]をご参照ください ',unsafe_allow_html=True)
csv_link(input_template,"[テンプレート] ","input_template.csv")
st.sidebar.write('<span style="color:red;background:yellow"><font size="1">＜ 1列目に目的変数、以降は説明変数を入力してください ＞',unsafe_allow_html=True)

st.sidebar.text('STEP1 : csvファイルをアップロード')
uploaded_file = st.sidebar.file_uploader("", type='csv')
st.markdown(button_css, unsafe_allow_html=True)
st.sidebar.write('<span style="color:red;background:yellow"><font size="1">＜ アップロード可能な最大ファイル容量は200MBです ＞',unsafe_allow_html=True)
st.sidebar.text('STEP2 : 実行ボタンをクリック')

if st.sidebar.button('実行'):
    #現在の時刻
    today = datetime.date.today()
    todaydetail = datetime.datetime.today()
    today_time = str(today) +'_' +str(todaydetail.hour) + '_' + str(todaydetail.minute) + '_'

    #入力データ読込
    df = pd.read_csv(uploaded_file)
    #相関係数ヒートマップ作成
    st.write("◇次元圧縮の効果がありそうか判断するために相関係数を可視化します。",)
    dfcorr = df.corr()
    
        # Correlation Matrix in Content #説明変数に相関があるか（次元圧縮が有効そうか）確認
    fig_corr = go.Figure([go.Heatmap(z=dfcorr.values,
                                    x=dfcorr.index.values,
                                    y=dfcorr.columns.values,
                                    zmin = -1,
                                    zmax = 1)])
    fig_corr.update_layout(height=1000,
                          width=1000,)
    st.plotly_chart(fig_corr)
    st.write('<center><span style="color:navy">＜ 相関係数ヒートマップ図 ＞',unsafe_allow_html=True)
    st.write('<center><span style="color:gray"><font size="2"> ※相関係数が[1]または[-1]に近い組合せの因子は圧縮できる可能性が高いです  ',unsafe_allow_html=True)
    #csv保存リンク作成
    csv_link(dfcorr.corr(),"[相関係数マトリクス] ",today_time + "corr_mat.csv")
 
    #'・・・標準化処理実行中・・・'
    dfs = df.iloc[:, 1:].apply(lambda x: (x-x.mean())/x.std(), axis=0).fillna(0)
    dfs = dfs.fillna(0)#nanをゼロで置換
    #'・・・主成分分析実行中・・・'
    pca = PCA()
    feature = pca.fit(dfs)
    # データを主成分空間に写像
    feature = pca.transform(dfs)
    # 第一主成分と第二主成分でプロットする
    img2 = plt.figure(figsize=(6, 6))
    plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=list(df.iloc[:, 0]))
    plt.grid()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    pca_features = pd.DataFrame(feature)
    st.text(" \n")#改行
    st.write("◇主成分を算出します。")
    st.write('<center><span style="color:navy"> ＜ 主成分表 ＞ ',unsafe_allow_html=True)
    col_name = list()
    for i in range(len(pca_features.columns)):
      col_name.append("PC"+ str(i+1))
    pca_features.columns = col_name
    st.dataframe(pca_features)#表の表示
    csv_link(pca_features,"[主成分] ",today_time + "pca_features.csv")
    st.write("◇第一主成分 - 第二主成分の散布図を描写します。")
    st.pyplot(img2)#画像表示
    st.write('<center><span style="color:navy"> ＜ 第一主成分 - 第二主成分の散布図 ＞ ',unsafe_allow_html=True)
    filename = today_time  + '第一主成分 - 第二主成分の散布図.png'
    plt.close()
    st.write("◇寄与率を算出します。")
    # 寄与率
    kiyoritu_list = pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    kiyoritu_list = kiyoritu_list.rename(columns={kiyoritu_list.columns[0]: 'contribution_rate'})
    st.write('<center><span style="color:navy"> ＜ 寄与率図 ＞ ',unsafe_allow_html=True)
    # Graph (Pie Chart in Sidebar)
    df_target = kiyoritu_list
    fig_target = go.Figure(data=[go.Pie(labels=df_target.index,
                                        values=list(df_target['contribution_rate']),
                                        hole=.3)])
    st.plotly_chart(fig_target)
    csv_link(pca_features,"[寄与率] ",today_time + "contribution_rate.csv")

    # 累積寄与率を図示する   
    st.write("◇累積寄与率を算出します。") 
    img4 = plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
    plt.grid()
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(img4)#画像表示
    st.write('<center><span style="color:navy"> ＜ 累積寄与率図 ＞ ',unsafe_allow_html=True)
    plt.close()

    # PCA の固有値
    st.write("◇固有値を算出します。")
    koyuchi = pd.DataFrame(pca.explained_variance_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    koyuchi = koyuchi.rename(columns={koyuchi.columns[0]: 'eigenvalue'})
    st.write('<center><span style="color:navy"> ＜ 固有値表 ＞ ',unsafe_allow_html=True)
    index_koyuchi = pd.DataFrame(koyuchi.index)
    koyuchi = koyuchi.reset_index(drop=True)
    index_koyuchi.columns = ['principal_components']
    koyuchi_plotdata = pd.concat([index_koyuchi,koyuchi],axis=1)
    st.write(px.bar(koyuchi_plotdata, x='principal_components', y='eigenvalue'))
    csv_link(koyuchi,"[固有値] ",today_time + "eigenvalue.csv")

    # PCA の固有ベクトル
    st.write("◇固有ベクトルを算出します。")
    koyubekutoru = pd.DataFrame(pca.components_, columns=df.columns[1:], index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
    st.write('<center><span style="color:navy"> ＜ 固有ベクトル表 ＞ ',unsafe_allow_html=True)
    st.dataframe(koyubekutoru)
    csv_link(koyubekutoru,"[固有ベクトル] ",today_time + "unique_vectors.csv")
    # 第一主成分と第二主成分における観測変数の寄与度をプロットする
    st.write("◇第一主成分 - 第二主成分の固有ベクトルを描写します。")
    img5 = plt.figure(figsize=(6, 6))
    for x, y, name in zip(pca.components_[0], pca.components_[1], df.columns[1:]):
        plt.text(x, y, name)
    plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
    plt.grid()
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    st.pyplot(img5)#画像表示
    st.write('<center><span style="color:navy"> ＜ 第一主成分 - 第二主成分の固有ベクトル図 ＞ ',unsafe_allow_html=True)
    plt.close()
    st.write("◇主成分と初期変数の関係を分析します。")
    #固有ベクトルを絶対値に変換
    koyubekutoru_abs = koyubekutoru.abs()
    koyubekutoru_abs = koyubekutoru_abs
    koyubekutoru_sum = koyubekutoru_abs.sum(axis=1)
    # st.dataframe(koyubekutoru_abs)
    # st.dataframe(koyubekutoru_sum)
    koyubekutoru_kind = pd.DataFrame()
    # st.write(str(len(koyubekutoru_sum)))
    for i in range(len(koyubekutoru_sum)):
        koyubekutoru_current = (koyubekutoru_abs.iloc[i,:].div(koyubekutoru_sum.iloc[i]))*100
        koyubekutoru_kind = pd.concat([koyubekutoru_kind , koyubekutoru_current],axis=1)
    koyubekutoru_kind = koyubekutoru_kind.T
    # for i in range(len(koyubekutoru_sum)):
        # debug = koyubekutoru_kind.iloc[i,0] + koyubekutoru_kind.iloc[i,1] + koyubekutoru_kind.iloc[i,2] + koyubekutoru_kind.iloc[i,3]+ koyubekutoru_kind.iloc[i,4]+ koyubekutoru_kind.iloc[i,5] + koyubekutoru_kind.iloc[i,6] + koyubekutoru_kind.iloc[i,7]+ koyubekutoru_kind.iloc[i,8] + koyubekutoru_kind.iloc[i,9]
        # st.write(str((debug)))
    koyubekutoru_kind = koyubekutoru_kind.T
    # st.dataframe(koyubekutoru_kind)
    import plotly.figure_factory as ff
    z = np.array(koyubekutoru_kind.values)
    fig_kind = go.Figure([go.Heatmap(z=koyubekutoru_kind.values,
                                    x=koyubekutoru_kind.index.values,
                                    y=koyubekutoru_kind.columns.values,)],)#hoverinfo='z'
    fig_kind.update_layout(height=1000,
                          width=1000,)
    st.plotly_chart(fig_kind)
    st.write('<center><span style="color:navy"> ＜ 主成分への初期変数の寄与度ヒートマップ図 ＞ ',unsafe_allow_html=True)
    koyubekutoru_kind = koyubekutoru_kind.T
    csv_link(koyubekutoru_kind,"[主成分への初期変数の寄与度] ","input_template.csv")
    

st.sidebar.write('<span style="color:red;background:yellow"><font size="1">＜ ファイルをアップロードしてから実行してください ＞',unsafe_allow_html=True)