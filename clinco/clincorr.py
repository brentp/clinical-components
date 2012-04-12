import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from collections import Counter
import argparse

import numpy as np
from scipy.stats import linregress, f_oneway
import pandas as pa
from scipy.stats import chi2_contingency

from pca import read_clinical

def is_categorical(column):

    try:
        if np.allclose(np.array(column).astype(int), column):
            column = column.astype(int)
    except (ValueError, TypeError):
        pass

    if np.issubdtype(column.dtype, np.floating):
        return False
    if np.issubdtype(column.dtype, np.int):
        return len(np.unique(column)) < 4

    if column.dtype == object:
        return True

    raise Exception("%s not handled" % column.dtype)

# bcol is the group.
def _group_anova(acol, bcol):
    agroups = []
    for bgroup in np.unique(bcol):
        agroups.append(acol[bcol == bgroup])
    agroups = [ag for ag in agroups if len(ag) > 0]
    if Counter([len(ag) for ag in agroups]).most_common(1)[0][0] == 1:
        # if many bin-sizes of 1, won't get valid results.
        p_value = 1.0
    else:
        f, p_value = f_oneway(*agroups)
    return {'anova_groups': len(agroups), 'p': p_value}

def compare(cola, colb):
    d = dict.fromkeys("correlation p n anova_groups atype btype".split(), "na")

    a = cola[cola.notnull() & colb.notnull()]
    b = colb[cola.notnull() & colb.notnull()]

    acat, bcat = is_categorical(a), is_categorical(b)
    l = ['numeric', 'categorical']
    d['atype'] = l[int(acat)]
    d['btype'] = l[int(bcat)]


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

    elif acat == True and bcat == False:
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


def run(clin, col_name=None, cutoff=0.05):
    tmpl = "%(acol)s\t%(bcol)s\t%(n)s\t%(correlation)s\t%(anova_groups)s\t%(p)s\t%(atype)s\t%(btype)s"
    print "#" + tmpl.replace("%(", "").replace(")s", "")
    for i, acol in enumerate(clin.columns):
        if col_name and acol != col_name: continue
        for j, bcol in enumerate(clin.columns[0 if col_name else (i + 1):]):
            if acol == bcol: continue
            d = compare(clin[acol], clin[bcol])
            # {'p': 0.0012043784307836521, 'anova_groups': 'na', 'n': 467, 'correlation':
            # -0.14938818779760124}
            if d['p'] != 'na' and d['p'] <= cutoff:
                d['p'] = "%.4g" % d['p']
                if 'na' != d['correlation']:
                    d['correlation'] = "%.3f" % d['correlation']
                d['acol'], d['bcol'] = acol, bcol
                print tmpl % d

def main():
    p = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", dest="clinical", help="clinical data file.")
    p.add_argument("--column",
            help="do correlations of all variables with this column")

    args = p.parse_args()
    if (None in (args.clinical, )):
        sys.exit(not p.print_help())

    clin = read_clinical(args.clinical)
    run(clin, args.column)


if __name__ == "__main__":
    import doctest
    if doctest.testmod(optionflags=doctest.ELLIPSIS |\
                                   doctest.NORMALIZE_WHITESPACE).failed == 0:
        main()
