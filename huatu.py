import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.serif'] = 'Arial'
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['text.usetex'] = True
import matplotlib.font_manager as font_manager

"""
a = [32.62,32.90,33.26,33.03,32.93]#table5 psnr
b = [0.9390,0.9405,0.9425,0.9426,0.9413]#table5 ssim
labels=[0.1,0.3,0.5,0.7,0.9] #table5
xlabel = r'\textbf{loss weighting coefficient} $\lambda$'
savepath = r'table5.png'
"""


a=[32.52,33.26,33.13,33.10]#table6 psnr
b=[0.9383,0.9425,0.9430,0.9432]#table6 ssim
labels = [2,5,10,15]#table6,7
xlabel = r'\textbf{patch size} $p$'
savepath = r'table6.png'


"""
a=[32.81,33.26,33.21,33.15]#table7 psnr
b=[0.9401,0.9425,0.9416,0.9420]#table7 ssim
labels = [2,5,10,15]#table6,7
xlabel = r'\textbf{window size} $g$'
savepath = r'table7.png'
"""

"""
a = [32.59,32.96,33.26,33.20,32.95]#table8 psnr
b = [0.9396,0.9409,0.9425,0.9413,0.9422]#table8 ssim
labels = [1,2,4,6,8]#table8
xlabel = r'\textbf{number of heads} $M$'
savepath = r'table8.png'
"""

plt.rcParams['axes.labelsize'] = 25 # xy轴label的size
plt.rcParams['xtick.labelsize'] = 20 # x轴ticks的size
plt.rcParams['ytick.labelsize'] = 20 # y轴ticks的size
width = 0.3 # 柱形的
x1_list = []
x2_list = []
for i in range(len(a)):
    x1_list.append(i)
    x2_list.append(i+width)
# 创建图层
font_axes = {#'family': 'Arial',
                 'weight': 'bold',
                 'size': 24,
              'color'  : 'black'
                 }
font_axesxx = {#'family': 'Times New Roman',
                 'weight': 'bold',
                 'size': 25,
              'color'  : 'black'
                 }
font = font_manager.FontProperties(#family='Arial',
                                    weight='bold',
                                   style='normal', 
                                   size=20)
fig, ax1 = plt.subplots()
ax1.set_ylabel(r'\textbf{PSNR}',font_axes)
ax1.set_xlabel(xlabel,font_axes)
plt.grid(ls='-.',which='both')
fig.set_size_inches(8,6)
ax1.set_ylim(32.500, 33.400)
#ax1.bar(x1_list,a, width=width, label=r"\textbf{PSNR}",color='c', align='edge')
#ax1.bar(x2_list,b, width=width, label=r"\textbf{SSIM}",color='g', align='edge', tick_label=labels)
#ax1.plot(x1_list,a,'s-',label=r"\textbf{PSNR}",color='c', align='edge')
ax6 = ax1.twinx()
ax6.set_ylim(0.9380, 0.9440)
ax6.set_ylabel(r'\textbf{SSIM}',font_axes)
#b1 = ax1.bar(x1_list,a, width=width, label=r"\textbf{PSNR}",color='c', align='edge')
#b2 = ax6.bar(x2_list,b, width=width, label=r"\textbf{SSIM}",color='g', align='edge', tick_label=labels)
p1,=ax1.plot(labels,a,'s-',label=r"\textbf{PSNR}",color='c')
p2,=ax6.plot(labels,b,'o-',label=r"\textbf{SSIM}",color='g')
#plt.legend(handles=[b1,b2],loc="upper left",prop=font,ncol=2)

leg=plt.legend(handles=[p1,p2],loc="upper left",prop=font,ncol=2)
ax1.yaxis.get_label().set_color(p1.get_color())
leg.texts[0].set_color(p1.get_color())

ax6.yaxis.get_label().set_color(p2.get_color())
leg.texts[1].set_color(p2.get_color())

plt.savefig(savepath,transparent=True,pad_inches=0.1,bbox_inches = 'tight',format='png')
#plt.show() 
