import numpy as np

import onedimsearch
from gradient import fund_best_schedule
from graphic import F, F2

if __name__ == '__main__':
    fund_best_schedule(F2(), onedimsearch.GoldSplit(), 1000, 1e-3, np.array([15, 15]))
