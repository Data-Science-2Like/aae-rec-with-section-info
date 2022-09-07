import matplotlib.pyplot as plt
import numpy as np

def plot_config(x,y1,y2,title,metric):
    fig = plt.figure()

    plt.ylim(0,max(max(y1),max(y2)))
    plt.plot(x,y1,label='No Section')
    plt.plot(x,y2,label='Section-Info')
    plt.ylabel(metric)
    plt.xlabel('drop factor')
    plt.legend()
    plt.title(title)
    plt.savefig(title + '.png')






x = [0.2,0.5,0.8]


t5_mrr = [0.06434,0.13253,0.18032]

t5_r2000 = [0.28878,0.28142,0.27664]

s5_mrr = [0.066033,0.13544,0.18421]

s5_r2000 = [0.28310,0.27654,0.27201]

plot_config(x,t5_mrr,s5_mrr,'Threshold-5-MRR','MRR')
plot_config(x,t5_r2000,s5_r2000,'Threshold-5-Recall','Recall@2k')

t10_mrr = [0.064829,0.137475,0.1811]

t10_r2000 = [0.359323,0.34914,0.34319]

s10_mrr = [0.065506,0.14172,0.19264]

s10_r2000 = [0.35040,0.34111,0.335640]

plot_config(x,t10_mrr,s10_mrr,'Threshold-10-MRR','MRR')
plot_config(x,t10_r2000,s10_r2000,'Threshold-10-Recall','Recall@2k')