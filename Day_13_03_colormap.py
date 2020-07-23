import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import pandas as pd
def colormap_1():
    x=np.random.rand(100)
    y=np.random.rand(100)
    t=np.arange(100)

    # plt.plot(x,y,'ro')
    # plt.scatter(x,y)
    plt.scatter(x,y,c=t)
    plt.show()

def colormap_2():
    x=np.arange(100)
    #                             #      R         G         B    투명도
    # print(cm.viridis(0))        # (0.267004, 0.004874, 0.329415, 1.0)
    # print(cm.viridis(255))      # (0.993248, 0.906157, 0.143936, 1.0)

    # plt.scatter(x, x)
    # plt.scatter(x, x, c=x, cmap='jet')

    # 반대쪽 대각선 그리기
    # plt.scatter(x, x[::-1], c=x, cmap='jet')
    # plt.scatter(x[::-1], x, c=x, cmap='jet')

    # 색상 반대로 출력하기
    plt.scatter(x, x, c=-x, cmap='jet')
    plt.scatter(x, x, c=x[::-1], cmap='jet')
    plt.scatter(x, x, c=x, cmap='jet_r')
    plt.show()

def colormap_3():
    print(plt.colormaps())
    '''
    ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap',
     'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r',
     'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r',
     'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples',
     'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r',
     'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia',
     'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot',
     'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r',
     'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix',
     'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat',
     'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r',
     'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r',
     'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r',
     'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic',
     'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b',
     'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r',
     'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']
     '''
    print(len(plt.colormaps())) # 166
    x=np.arange(100)
    plt.figure(1)
    plt.scatter(x,x,c=x, cmap='spring')
    # plt.show()

    plt.figure(2)
    plt.scatter(x,x,c=x, cmap='rainbow')
    plt.colorbar()
    plt.show()

def colormap_4():
    # plt.imshow(np.random.rand(10,10))
    plt.imshow(np.random.rand(10,10), cmap='Accent')
    plt.tight_layout()
    plt.show()

def colormap_5():
    jet=cm.get_cmap('jet')
    print(jet(-5))
    print(jet(0))
    print(jet(127))
    print(jet(128))
    print(jet(255))
    print(jet(256))

    print(jet(0.1))
    print(jet(0.5))
    print(jet(0.7))

    print(jet(128/255))
    print(jet(255/255))
    print()

    print(jet([0,255]))
    print(jet(range(0,256,32)))
    print(jet(np.linspace(0.2,0.7,3)))
    # print(np.arange(0,1,0.1))
    # print(np.linspace(0,1,11))
def colormap_6():
    flight = sns.load_dataset('flights')
    print(type(flight))
    print(flight)

    df=flight.pivot('month', 'year', 'passengers')
    print(df, end='\n\n')
    #
    # plt.pcolor(df.values)
    # plt.title('flights heatmap')
    # # plt.xticks(range(12), df.index)
    # plt.xticks(0.5+np.arange(0,12,2), df.columns[::2])
    # plt.yticks(0.5+np.arange(12), df.index)
    # plt.colorbar()
    # plt.show()
    #
    sns.heatmap(df,annot=True, fmt='d')
    plt.show()
# colormap_1()
# colormap_2()
# colormap_3()
# colormap_4()
# colormap_5()
colormap_6()