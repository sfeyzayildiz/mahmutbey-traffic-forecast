'''
Conformalized quantile regression (CQR) for the prediction intervals.

The models predict quantiles, but on new data their intervals can be too narrow.
CQR widens (or narrows) an interval by a constant learned on a calibration set,
so that its coverage on new data is close to the nominal level.

Reference: Romano, Patterson and Candès (2019), "Conformalized Quantile Regression", NeurIPS.
'''

import numpy as np


def cqr_offset(y, lower, upper, coverage):
    '''Offset to add to both ends of an interval so that it reaches the target coverage.'''
    scores=np.maximum(lower-y, y-upper)
    n=len(y)
    level=min(1.0, np.ceil((n+1)*coverage)/n)
    return float(np.quantile(scores, level, method='higher'))


def calibrate(cal, test, pairs=(('q05', 'q95', 0.90), ('q25', 'q75', 0.50))):
    '''Return a copy of test with calibrated interval columns, plus the offsets used.'''
    out=test.copy()
    offsets={}
    for lo, hi, cov in pairs:
        off=cqr_offset(cal['y'].values, cal[lo].values, cal[hi].values, cov)
        out[lo]=test[lo]-off
        out[hi]=test[hi]+off
        offsets[f'{lo}-{hi}']=off
    return out, offsets
