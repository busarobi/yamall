import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os, sys
########################################################################################################################
########################################################################################################################
########################################################################################################################

methods = ['SGD_VW', 'Pistol', 'SOLO', 'SVRG', 'SVRG_FR', 'free_rex']
data_type = 'default'
lr = 0.1
result_for_plot = {}
for m in methods:
    results = []
    ml = sys.maxsize
    for rep in range(0, 10):
        fname = ('./output/%s_r_%.4d_%s_lr_%0.1f.txt' % (m, rep, data_type, lr))
        if (not os.path.exists(fname)):
            print( "Does not exists: %s" % fname )
            continue

        print('Reading %s' % fname)
        # result[m] = np.array(np.loadtxt(fname))
        arr = np.genfromtxt(fname, delimiter=' ')
        ml = min((arr.shape[0], ml))
        results.append(arr)
    # averaging


    key = "%s (lr %0.1f)" % (m, lr)

    # test error
    arr = np.zeros((ml, 4))
    for d in results:
        arr += d[0:ml, :]

    arr /= len(results)

    result_for_plot[key] = arr


colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

leg = []
# list all data in history
#print(history_sgd.history.keys())
i = 0
for m in result_for_plot.keys():
    mat = result_for_plot[m]
    x = mat[:,0]
    y_train = mat[:, 1]
    y_test = mat[:, 2]
    plt.loglog(x,y_train, color=colors[i])
    plt.loglog(x,y_test, '--',color=colors[i])

    leg.append('train (%s)' % m)
    leg.append('test (%s)' % m)

    i += 1

# eloss=0.609341
# xlim = plt.gca().get_xlim()
# plt.plot(xlim, [eloss,eloss], '-')

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(leg, loc='upper right')
plt.grid(True)

pp = PdfPages(( './comp_lr_%0.1f.pdf' % lr ) )
plt.savefig(pp, format='pdf')
pp.close()