#func: any df,calc distance with certain sample, scale, plot
import pandas as pd
import numpy as np
def mydistance(df, column_name, sample_index_col,origin_sample, distance_type):
    from sklearn.preprocessing import StandardScaler
    """
    choose distance type and calculate the distance between each row and the specific row, then add the distance to the original data and will change the columns belong to column_name to scaled data.
    """
    if distance_type == 'Euclidean':
        from sklearn.metrics.pairwise import euclidean_distances as distance
    elif distance_type == 'Manhattan':
        from sklearn.metrics.pairwise import manhattan_distances as distance
    elif distance_type == 'Cosine':
        from sklearn.metrics.pairwise import cosine_distances as distance

    
    scaler = StandardScaler()
    arr_scaled = scaler.fit_transform(df[column_name])
    # 将标准化后的数据转换回DataFrame
    df_scaled = pd.DataFrame(arr_scaled, columns=column_name, index=df.index)
    
    # 将未标准化的列添加到新的DataFrame中
    columns=df.columns
    for i in columns:
        if i not in column_name:
            df_scaled[i]=df[i]
      
    distance_result=distance(df_scaled[column_name],df_scaled[df_scaled[sample_index_col]==origin_sample][column_name])
    df_scaled[f'{distance_type}'] =distance_result 

    return df_scaled
    

def plot_distance(df, distance_type:str,origin_sample=None,x_col=None,y_col=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
        
    plt.rcParams['font.family'] = 'Arial'
    # 设置字体大小
    plt.rcParams['font.size'] = 12  # 设置字体大小
   
    # 示例数据
    categories = df[f'{x_col}'].values
    values = df[f'{y_col}'][:-1].values

    # 定义渐变颜色（浅蓝到深蓝）
    # cmap = LinearSegmentedColormap.from_list("custom_blue", ["#00008b","#576db9"])  # 深蓝到浅蓝渐变
    cmap = LinearSegmentedColormap.from_list("custom_blue", ["#ed2f47","#f08693"])  # 深红到浅红渐变

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制水平柱状图（每个柱子填充水平方向渐变色）
    for i, (category, value) in enumerate(zip(categories, values)):
        # 创建水平方向的渐变数组
        gradient = np.linspace(0, 1, 500).reshape(1, -1)  # 水平方向渐变,若要垂直方向渐变，改为gradient = np.linspace(0, 1, 500).reshape(-1, 1)
        ax.imshow(
            gradient,
            extent=(0, value, i - 0.4, i + 0.4),  # 设置柱子的范围，水平方向为(0, value),i-0.4到i+0.4表示柱子的高度(即宽度)
            origin='lower', # 原点在左下角，也可以设置为upper，原点在左上角
            aspect='auto', # 设置纵横比
            cmap=cmap
        )

    # 设置y轴标签和刻度
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=12)
    ax.set_xlim(0, max(values) + 1)  # x轴范围
    ax.set_ylim(-0.5, len(categories) - 0.5)  # 防止柱子贴边

    # 添加标题和标签
    #ax.set_title('Horizontal Bar Chart with Horizontal Gradient Colors', fontsize=16)
    ax.set_xlabel('Values', fontsize=14)
    ax.set_ylabel('Categories', fontsize=14)

    # 显示图形
    plt.tight_layout()
    plt.xlabel(f'{distance_type} distance with {origin_sample}', fontsize=14)
    plt.ylabel('Solvent', fontsize=14)
    plt.savefig(f'{distance_type} distance with {origin_sample}.svg')
    plt.show()
    

if __name__ == '__main__':
    all_solvent=pd.read_csv('cluster/all_solvent_drop.csv')
    df=all_solvent[all_solvent['green']==1]
    df=pd.concat([all_solvent[all_solvent['abbr']=='DMF'],df])
    df=df.iloc[::-1]
    column_name=['LogP', 'MW', 'HBD', 'HBA', 'TPSA', 'RB', 'C', 'O', 'N','P','S','F','B']
    df_scaled=mydistance(df,column_name, 'abbr', 'DMF', 'Euclidean')
    plot_distance(df_scaled, 'Euclidean','DMF','abbr','Euclidean')

    
