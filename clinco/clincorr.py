import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
print sys.path[0]
import argparse
import collections

import numpy as np
from scipy.stats import linregress, f_oneway
import pandas as pa
from scipy.stats import chi2_contingency

from pca import read_clinical

def is_categorical(column):
    if np.issubdtype(column.dtype, np.floating):
        return False
    if np.issubdtype(column.dtype, np.int):
        return len(np.unique(column)) < 6

    if column.dtype == object:
        return True
    1/0

# bcol is the group.
def _group_anova(acol, bcol):
    agroups = []
    for bgroup in np.unique(bcol):
        agroups.append(acol[bcol == bgroup])
    agroups = [ag for ag in agroups if len(ag) > 0]
    f, p_value = f_oneway(*agroups)
    return {'anova_groups': len(agroups), 'p': p_value}

def compare(cola, colb):
    d = dict.fromkeys("correlation p n anova_groups".split(), "na")
    acat, bcat = is_categorical(cola), is_categorical(colb)

    a = cola[cola.notnull() & colb.notnull()]
    b = colb[cola.notnull() & colb.notnull()]
    d['n'] = len(a)
    if d['n'] == 0: return d

    # both numerical
    if acat == bcat == False:
        slope, intercept, r_value, p_value, std_err = linregress(a, b)
        d['correlation'] = r_value
        d['p'] = p_value

    # group a values by b and do anova.
    elif acat == False and bcat == True:
        d.update(_group_anova(a, b))

    elif acat == False and bcat == True:
        d.update(_group_anova(b, a))

    # both categorical...
    else:
        dsub = pa.DataFrame({a.name:a, b.name: b})
        summ = np.array(dsub.groupby(by=[a.name, b.name]).size().unstack(b.name))
        summ[np.isnan(summ)] = 0
        chi2, p, dof, ex = chi2_contingency(summ)    
        d['anova_groups'] = 'chi-sq'
        d['p'] = p

    return d

def run(clin):
    corr = collections.defaultdict(dict)
    for ic in clin.columns:
        for jc in clin.columns:
            if ic == jc: continue
            if (jc, ic) in corr: continue
            d = compare(clin[ic], clin[jc])
            corr[ic, jc] = d
            if d['p'] != 'na' and d['p'] < 0.01:
                print ic, jc, d

def main():
    p = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", dest="clinical", help="clinical data file.")

    args = p.parse_args()
    if (None in (args.clinical, )):
        sys.exit(not p.print_help())

    clin = read_clinical(args.clinical)
    run(clin)


if __name__ == "__main__":
    import doctest
    if doctest.testmod(optionflags=doctest.ELLIPSIS |\
                                   doctest.NORMALIZE_WHITESPACE).failed == 0:
        main()
