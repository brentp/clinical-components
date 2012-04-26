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
    """
    try to guess if a column is numeric or categorical
    e.g. if the column only has values 0 and 1, assume
    categorical
    """

    try:
        if np.allclose(np.array(column).astype(int), column):
            column = column.astype(int)
    except (ValueError, TypeError):
        pass

    if np.issubdtype(column.dtype, np.floating):
        return False
    if np.issubdtype(column.dtype, np.int):
        if column.name.startswith("num"):
            return False
        # contains all integer values from min to max and min is 0 or 1 then
        # categorical
        if np.min(column) in (0, 1) \
            and len(np.setdiff1d(np.arange(min(column), max(column) + 1), column)) < 2:
            return True
        return len(np.unique(column)) < 3

    if column.dtype == object:
        return True

    raise Exception("%s not handled" % column.dtype)

def x_most_frequent(counts, x):
    return Counter(counts).most_common(1)[0][0] == x

# bcol is the group.
def _group_anova(acol, bcol):
    agroups = []
    for bgroup in np.unique(bcol):
        agroups.append(acol[bcol == bgroup])
    agroups = [ag for ag in agroups if len(ag) > 0]
    if x_most_frequent([len(ag) for ag in agroups], 1):
        # if many bin-sizes of 1, won't get valid results.
        p_value = 1.0
    else:
        f, p_value = f_oneway(*agroups)
    return {'anova_groups': len(agroups), 'p': p_value}

def compare(cola, colb):
    d = dict.fromkeys("correlation p n anova_groups atype btype".split(), "na")

    try:
        acat, bcat = is_categorical(cola[cola.notnull()]), is_categorical(colb[colb.notnull()])
    except ValueError:
        return d
    a = cola[cola.notnull() & colb.notnull()]
    b = colb[cola.notnull() & colb.notnull()]

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
        d['anova_groups'] = 'chi-sq:' + ",".join(map(str, summ.shape))
        d['p'] = p
        if x_most_frequent(summ.flat, 0):
            d['p'] = 1

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
    p.add_argument("--na", help="addition comma-separated tokens to count as na",
            default="")
    p.add_argument("--cutoff", help="only show assocations with a p-value below \
                    cutoff. default: %(default)s", type=float, default=0.05)
    p.add_argument("--column",
            help="do correlations of all variables with this column")
    p.add_argument("clinical", help="clinical data file.")

    args = p.parse_args()
    if (None in (args.clinical, )):
        sys.exit(not p.print_help())

    clin = read_clinical(args.clinical, args.na.split(","))
    run(clin, args.column, args.cutoff)


if __name__ == "__main__":
    import doctest
    if doctest.testmod(optionflags=doctest.ELLIPSIS |\
                                   doctest.NORMALIZE_WHITESPACE).failed == 0:
        main()
