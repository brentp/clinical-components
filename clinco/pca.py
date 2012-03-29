import sys
import argparse
from itertools import cycle, izip
import matplotlib
from toolshed import reader
import numpy as np
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA, RandomizedPCA, ProbabilisticPCA, \
        KernelPCA

from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.lda import LDA

from scipy.stats import linregress, f_oneway

CLASSES = {
    "PCA": PCA,
    "LDA": LDA,
    "RandomizedPCA": RandomizedPCA,
    "IsoMap": Isomap,
    "LLE": LocallyLinearEmbedding,
    "KPCA": KernelPCA,
    "PPCA": ProbabilisticPCA,
}

def shannon(explained_variance_ratio):
    evr = np.asarray(explained_variance_ratio)
    L = len(evr)
    evr = evr[evr > 0]
    return -1.0 / np.log(L) * (evr * np.log(evr)).sum()

def readX(fX, n=1):
    """
    n == 1 means to skip first column because it's the ID
    """
    fhX = reader(fX, header=False)
    X_headers = fhX.next()

    ids, X = [], []
    nan = float('nan')
    nan = 10
    for toks in fhX:
        ids.append(toks[0])
        try:
            vals = map(float, toks[n:])
        except ValueError:
            vals = [float(t) if not t in ("NA", "na") else nan for t in
                                                                toks[n:]]
        X.append(np.array(vals))
    return X_headers, np.array(ids), np.array(X)

import dateutil
def date_parse(adate):
    try:
        return dateutil.parser.parse(adate)
    except ValueError:
        return np.nan

def read_clinical(fclinical):
    """
    read the clinical data and try to guess the types
    """
    import pandas as pa
    header = open(fclinical).readline().rstrip("\r\n").split("\t")
    conv = dict((col, date_parse) for col in header if "date" in col or \
                            col == "passed_qc")
    try:
        return pa.read_table(fclinical, converters=conv)
    except:
        return pa.read_table(fclinical)

def _clinical_to_ys(clinical1):
    classes = [x for x in sorted(np.unique(np.array(clinical1)))] # if not np.isnan(x)]
    return classes, np.array([classes.index(c) if c in classes else np.nan for c in clinical1]) # if not np.isnan(c)])

def run(fX, fclinical, header_keys, fig_name, klass, label_key=None):

    clinical = read_clinical(fclinical)
    X_headers, X_ids, X = readX(fX)

    ci = map(str, list(clinical[clinical.columns[0]]))
    X_out = [xi for xi in X_ids if not xi in ci]
    if X_out:
        X = X[[i for i, xi in enumerate(X_ids) if xi in ci]]

        print >>sys.stderr, "removing %i rows in X but not in clinical" \
                % (len(X_out))
    X_ids = np.array([xi for xi in X_ids if xi in ci])
    clinical = clinical.irow([ci.index(xi) for xi in X_ids if not xi in X_out])
    #print clinical
    assert all(X_id == str(c_id) for X_id, c_id in
               zip(X_ids, clinical[clinical.columns[0]])), "IDS don't match!"
    if False: # example filtering
        #p = np.array([c["diagmin"] == "IPF" for c in
        p = np.array([c["diagmaj"] == "control" or c["diagmin"] == "IPF" for c in
          (clinical.irow(i) for i in range(len(clinical)))])
        clinical = clinical[p]
        X = X[p]
        X_ids = X_ids[p]

    assert all(k in clinical for k in header_keys)
    assert label_key is None or label_key in clinical

    assert len(header_keys) == 1, "not implemented"
    header_key = header_keys[0]
    yclasses, y = _clinical_to_ys(clinical[header_key])

    assert X.shape[0] == y.shape[0], (X.shape, y.shape)

    if klass.__name__ == "KernelPCA":
        clf = klass(20, kernel="linear", gamma=3./X.shape[0]).fit(X) #3./X.shape[0]).fit(X)
    elif not ("PCA" in klass.__name__ or "LDA" in klass.__name__):
        clf = klass(6, out_dim=10).fit(X, y)
    else:
        clf = klass(20).fit(X)
    X_r = clf.fit_transform(X)

    #U, S, V = clf._fit(X)
    #print U.shape, S.shape, V.shape
    # Which to use?
    components = X_r
    #print clf.components_.T.shape
    #components = V.T
    #print components.shape, clinical.shape


    for i, (color, yclass) in list(enumerate(zip(cycle("rgbckym"), yclasses))):
        try:
            if np.isnan(yclass): continue
        except NotImplementedError:
            pass
        if isinstance(yclass, np.floating):
            p = y == i
            color = 1. / (len(yclasses) + 1) * (i + 1)
            color = (color, color, color)
        else:
            p = clinical[header_key] == yclass

        xs = components[p, 0]
        ys = components[p, 1]
        y2s = components[p, 2]


        plt.subplot(2, 1, 1)
        plt.scatter(xs, ys, c=color, edgecolor=color, s=12, label=yclass)
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        #plt.scatter(xs, ys, c=color, s=6, label=yclass)
        if label_key is not None and i == 1:
            labels = clinical[label_key][p]
            for xx, yy, label in izip(xs, ys, labels):
                plt.text(xx, yy, label, color=color, fontsize=6, multialignment='right')

        plt.subplot(2, 2, 3)
        plt.scatter(xs, y2s, c=color, edgecolor=color, s=12, label=yclass)

    plt.subplot(2, 1, 1)
    plt.title(header_key)
    leg = plt.legend(scatterpoints=1, loc='lower right')
    leg.get_frame().set_alpha(0.5)
    leg_txt = plt.gca().get_legend().get_texts()
    plt.subplot(2, 2, 3)
    plt.xlabel('component 2')
    plt.ylabel('component 3')
    plt.setp(leg_txt, fontsize=8)

    print_correlations(components, clinical)


    if hasattr(clf, "explained_variance_ratio_"):
        ax = plt.subplot(2, 2, 4)

        print >>sys.stderr, map(float, clf.explained_variance_ratio_[:9])
        idx = min(50,  len(clf.explained_variance_ratio_))
        plt.bar(np.arange(idx) + 0.5, 100. * clf.explained_variance_ratio_[:idx],
                color=(0.5, 0.5, 0.5),)
        plt.xlim(0, idx + 1)
        plt.xlabel('# components')
        plt.ylabel('% variance explained')
        plt.text(0.5, 0.9, "entropy: %.2f" % shannon(clf.explained_variance_ratio_),
                transform=ax.transAxes)
    #plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.26)

    plt.savefig(fig_name)

    plt.figure()
    plt.imshow(components, aspect='auto', interpolation='none')
    plt.savefig('i.png')


def print_correlations(components, clinical):
    print "component_vector\tclinical_variable\tn\tR\tanova_groups\tp_value"
    for i in range(min(5, components.shape[1])):
        j = i + 1
        for column in clinical.columns[1:]:
            clin = clinical[column]
            x = np.array(clin[clin.notnull()])
            y = components[clin.notnull(), i]
            #if not x.dtype in (np.float64, np.float, np.float32, np.int, np.datetime_):

            if x.dtype == object and not hasattr(x[0], "date"):
                xu = sorted(np.unique(x))
                if len(xu) > 20: continue # too many classes for anova
                a = [[] for _ in xu]
                # if it's a string class, do anova to get the p-value
                for xx, yy in zip(x, y):
                    idx = xu.index(xx)
                    a[idx].append(yy)
                f, p_value = f_oneway(*[aa for aa in a if len(aa) > 0])
                R = "na"
                aov = "%i-groups" % len(a)

                #x = np.array([xu.index(xi) for xi in x], dtype='float')
            else:
                if hasattr(x[0], "date"):
                    x = np.array(x, dtype=np.datetime64).astype(int)
                    if i == 0:
                        print >>open('data.txt', 'w'), "\n".join("%i\t%.3f" % (xx, yy) for xx, yy in zip(x, y))
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                R = "%.3f" % r_value
                aov = "na"

            if p_value > 0.1 or np.isnan(p_value): continue
            p_value = "%.3g" % p_value
            n = len(y)
            print "%(j)i\t%(column)s\t%(n)i\t%(R)s\t%(aov)s\t%(p_value)s" % \
                locals()



def main():
    p = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-X", dest="X", help="numerical data file first column is ID"
              " first row is header")
    p.add_argument("-c", dest="clinical", help="clinical data file. first "
              "column matches that in X")
    p.add_argument("-k", dest="key", help="column header(s) to separate classes")
    p.add_argument("-l", dest="label",
        help="optional column header(s) to label points")
    p.add_argument("-m", dest="method", choices=CLASSES.keys(),
                 help="method to use for transformation.",
                 default="RandomizedPCA")
    p.add_argument("-f", dest="fig_name", help="path to save figure")

    args = p.parse_args()
    if (None in (args.X, args.clinical, args.key, args.fig_name)):
        sys.exit(not p.print_help())

    run(args.X, args.clinical, args.key.rstrip().split(","), args.fig_name,
        CLASSES[args.method], args.label)

if __name__ == "__main__":
    import doctest
    if doctest.testmod(optionflags=doctest.ELLIPSIS |\
                                   doctest.NORMALIZE_WHITESPACE).failed == 0:
        main()