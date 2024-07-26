import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import os

cur_dir = os.path.dirname(__file__)





with open(f"{cur_dir}/results/predictions_t10000_s10_f70_n11.pkl", 'rb') as fp:
    ccn_preds = pickle.load(fp)

with open(f"{cur_dir}/results/reg_predictions_t10000.pkl", 'rb') as fp:
    reg_preds = pickle.load(fp)

conf_ccn = confusion_matrix(ccn_preds['target'], ccn_preds['sample'])
conf_reg = confusion_matrix(reg_preds['target'], reg_preds['sample'])

loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals


fig, axs = plt.subplots(1,2, figsize=(20,15))
axs[0].matshow(conf_ccn)
axs[0].xaxis.set_major_locator(loc)
axs[0].yaxis.set_major_locator(loc)
axs[0].set_title("Correlation Classwise")
axs[1].matshow(conf_reg)
axs[1].xaxis.set_major_locator(loc)
axs[1].yaxis.set_major_locator(loc)
axs[1].set_title("Logistic Regression")
plt.tight_layout()
plt.show()