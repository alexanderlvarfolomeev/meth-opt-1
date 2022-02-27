import numpy as np

import onedimsearch
from gradient import fund_best_schedule
from graphic import F

if __name__ == '__main__':
    fund_best_schedule(F(), onedimsearch.GoldSplit(), 1000, 1e-7, np.array([15, 15]))
