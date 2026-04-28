#func: any df,select column, scale, cluster, plot
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import silhouette_score, silhouette_samples

def mycluster(df, column_name,cluster_method,n_clusters=3):
    """
    choose cluster method and cluster the data, then add the cluster result to the original data and will change the columns belong to column_name to scaled data.
    
    """
    if cluster_method == 'kmeans':
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=n_clusters)
    elif cluster_method == 'dbscan':
        from sklearn.cluster import DBSCAN
        model = DBSCAN(eps=0.1, min_samples=5) # eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. mini
    elif cluster_method == 'hdbscan':
        import hdbscan
        model = hdbscan.HDBSCAN(min_cluster_size=5) # min_cluster_size: The minimum size of clusters; single linkage splits that contain fewer points than this will be considered points "falling out" of a cluster rather than a cluster splitting off from a larger one. 即每个簇的最小样本数

    elif cluster_method == 'gmm':
        from sklearn.mixture import GaussianMixture
        model = GaussianMixture(n_components=3)
    elif cluster_method == 'birch':
        from sklearn.cluster import Birch
        model = Birch(n_clusters=n_clusters)
    elif cluster_method == 'agglomerative':
        from sklearn.cluster import AgglomerativeClustering
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif cluster_method == 'spectral':
        from sklearn.cluster import SpectralClustering
        model = SpectralClustering(n_clusters=n_clusters)
    else:
        print('cluster_method not in [kmeans,dbscan,gmm,birch,agglomerative,spectral]')
        return
    
    # 标准化数据
    scaler = StandardScaler()
    arr_scaled = scaler.fit_transform(df[column_name])
    # 将标准化后的数据转换回DataFrame
    df_scaled = pd.DataFrame(arr_scaled, columns=column_name, index=df.index)
    # 训练模型
    model.fit(df_scaled)
    # 将聚类结果添加到原始数据中
    
    df[column_name]=df_scaled[column_name]
    df['cluster'] = model.labels_
    labels = model.labels_
    # Calculate silhouette score only if there is more than one cluster
    if len(np.unique(labels)) > 1 and -1 not in labels:  # Ensure there is more than one cluster and no noise
        sil_score = silhouette_score(df_scaled, labels)
    else:
        print("Silhouette score cannot be calculated (e.g., only one cluster detected or noise).")

    return df, sil_score

def plot_cluster(df,column_name:list, cluster_method:str, dim_red=True,x_column=None,y_column=None,dim_red_method='t-SNE',show_specific_sample=False,cluster_col=None,color_col=None,show_sample_col=None,samples_name=None,sample_name_dict=None,silhouette_score=None):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Ellipse

    if dim_red:
        if dim_red_method == 't-SNE':
            # Perform t-SNE to reduce dimensions to 2
            tsne = TSNE(n_components=2, random_state=42)
            tsne_result = tsne.fit_transform(df[column_name])
            # Create a new DataFrame with t-SNE results
            df_dim = pd.DataFrame(tsne_result, columns=['Dim1', 'Dim2'])

        elif dim_red_method == 'pca':
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df[column_name])
            # Create a new DataFrame with PCA results
            df_dim = pd.DataFrame(pca_result, columns=['Dim1', 'Dim2'])

        df_dim[cluster_col] = df[cluster_col]
        df_dim[show_sample_col] = df[show_sample_col]  # Add solvent names for annotation
        df_dim[color_col] = df[color_col]
        #df_dim['silhouette'] = df['silhouette']
        df_dim.to_csv('cluster/df_dim.csv',index=False)
            # 提取焦点 A 和 B
        A = df_dim[df_dim[show_sample_col] == 'DMF'][['Dim1', 'Dim2']].values[0]
        B = df_dim[df_dim[show_sample_col] == 'DMSO'][['Dim1', 'Dim2']].values[0]

        # 计算椭圆的长轴和短轴
        distance_AB = np.linalg.norm(A - B)  # A 和 B 之间的距离
        D=distance_AB*1.10
        if D <= distance_AB:
            print("The distance between the points is larger than D, can't form an ellipse.")
            return
        
        # 椭圆的长轴 a 和短轴 b
        a = D / 2  # 长轴
        b = np.sqrt(a**2 - (distance_AB / 2)**2)  # 短轴

        # 计算椭圆的中心 (C点)
        center = (A + B) / 2  # 椭圆中心是 A 和 B 的中点
        
        # 计算椭圆的旋转角度
        angle = np.arctan2(B[1] - A[1], B[0] - A[0]) * 180 / np.pi  # 计算与x轴的角度

        # 使用 Ellipse 绘制椭圆
        ellipse = Ellipse(xy=center, width=2*a, height=2*b, angle=angle, edgecolor='red', fc='none', lw=1)

        # Add silhouette values for annotation
        
        
        
        # Visualize the results
        plt.rcParams['font.family'] = 'Arial'
    # 设置字体大小
        plt.rcParams['font.size'] = 12  # 设置字体大小
        plt.figure(figsize=(10, 6))
        # n_colors=len(df_dim[color_col].unique())
        # custom_palette=sns.color_palette("hls",n_colors ) # Custom colors
        custom_palette=['#f2051d','#0ecf12']

        # Plot the scatterplot
        sns.scatterplot(x='Dim1', y='Dim2', hue=color_col,  style=cluster_col,data=df_dim, palette=custom_palette, s=100)
        if show_specific_sample:
            for i in range(len(df_dim)):
                solvent_name = df_dim['solvent'].iloc[i]
                if solvent_name in samples_name:  # Check if the solvent is one of the selected

                    plt.text(df_dim['Dim1'].iloc[i], df_dim['Dim2'].iloc[i],
                         f"{sample_name_dict[solvent_name]}",
                         fontsize=10, ha='center', va='bottom', color='black', weight='bold')
                    
                    # plt.text(df_dim['Dim1'].iloc[i], df_dim['Dim2'].iloc[i], sample_name_dict[solvent_name], fontsize=16, ha='center', va='bottom', color='black', weight='bold')

        # Add silhouette values for annotation

        plt.gca().add_patch(ellipse)
        # Final plot settings
        # plt.title(f'{dim_red_method} visualization of descriptor vectors with {cluster_method} cluster results', fontsize=16)
        # 设置轴刻度标签的字体大小
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.xticks(fontsize=16, fontweight='bold')  # x轴刻度数字
        plt.yticks(fontsize=16, fontweight='bold')  # y轴刻度数字
        # 设置轴线的宽度
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['right'].set_linewidth(2)
        plt.legend(fontsize=20)
        if silhouette_score is not None:
            plt.text(0.95, 0.95, f"Silhouette: {silhouette_score:.3f}",
                     fontsize=14, color='black', ha='right', va='top', transform=plt.gca().transAxes)
        plt.xlabel(f'Component 1', fontsize=18,fontweight='bold')
        plt.ylabel(f'Component 2', fontsize=18,fontweight='bold')
        plt.savefig(f'{cluster_method}_{dim_red_method}.svg')
        plt.show()
        
    else:
        plt.rcParams['font.family'] = 'Arial'
    # 设置字体大小
        plt.rcParams['font.size'] = 12  # 设置字体大小
        plt.figure(figsize=(10, 6))
        n_colors=len(df[color_col].unique())
        custom_palette=sns.color_palette("hls", n_colors) # Custom colors
        # Plot the scatterplot
        sns.scatterplot(x='Dim1', y='Dim2', hue=color_col, style=cluster_col,data=df_dim, palette=custom_palette, s=100)
        
        if show_specific_sample:
            for i in range(len(df)):
                solvent_name = df[show_sample_col].iloc[i]
                if solvent_name in samples_name:  # Check if the solvent is one of the selected
                    plt.text(df[x_column].iloc[i], df[y_column].iloc[i], sample_name_dict[solvent_name], fontsize=12, 
                            ha='center', va='bottom', color='black', weight='bold')

        # Final plot settings
        plt.title(f' {cluster_method} clustering (silhouette score:{silhouette_score})', fontsize=16)
        plt.xlabel(f'{x_column}', fontsize=14,fontweight='bold')
        plt.ylabel(f'{y_column}', fontsize=14,fontweight='bold')
        plt.legend()
        plt.savefig(f'{cluster_method}_{dim_red_method}.svg')
        plt.show()


        

if __name__ == '__main__':
    all_solvent=pd.read_csv('cluster/all_solvent_drop.csv')
    column_name=['LogP', 'MW', 'HBD', 'HBA', 'TPSA', 'RB', 'C', 'O', 'N','P','S','F','B']
    all_solvent_clustered,score=mycluster(all_solvent,column_name,cluster_method='kmeans',n_clusters=4)

    # samples_name=["DMF","DMSO","ACN","THF","AN","2MEeTHF","isopropanol","GBL","GVL","ethyl acetate","NMP"]
    # samples_name_dict={"DMF":"DMF","DMSO":"DMSO","THF":"THF","AN":"AN","2MEeTHF":"2-MeTHF","isopropanol":"IPA","GBL":"GBL","GVL":"GVL","ethyl acetate":"EA",}
    samples_name=['DMF','DMSO','GVL','THF','2MeTHF','1-P(1-pentanol)']
    samples_name_dict={'DMF':'DMF','DMSO':'DMSO','GVL':'GVL','THF':'THF','2MeTHF':"2-MeTHF",'1-P(1-pentanol)':'1-P'}
    # samples_name=all_solvent_clustered['solvent'].unique()
    # samples_name_dict={i:i for i in samples_name}
    plot_cluster(all_solvent_clustered,column_name,cluster_method='hdbscan', dim_red=True,x_column=None,y_column=None,dim_red_method='t-SNE',show_specific_sample=True,cluster_col='cluster',color_col='green',show_sample_col='solvent',samples_name=samples_name,sample_name_dict=samples_name_dict,silhouette_score=score)
    
