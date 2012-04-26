import sys
import argparse
from itertools import cycle, izip
import matplotlib
from toolshed import reader
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')

from dateutil.parser import parse as date_parse

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

def readX(fX, transpose, n=1, nan_value=0):
    """
    n == 1 means to skip first column because it's the ID
    returns ids, probe_names, X
    """
    fhX = reader(fX, header=False)
    X_probes = fhX.next()[1:]

    ids, X = [], []
    #nan = float('nan')
    for toks in fhX:
        ids.append(toks[0])
        try:
            vals = map(float, toks[n:])
        except ValueError:
            vals = [float(t) if not t in ("NA", "na", "") else nan_value
                                       for t in toks[n:]]
        X.append(np.array(vals))
    X = np.array(X)
    if transpose:
        return X_probes, np.array(ids), X
        #return np.array(ids), X_probes, X.T
    else:
        return np.array(ids), X_probes, X.T

def try_date_parse(adate):
    try:
        return date_parse(adate)
    except ValueError:
        return np.nan

def read_clinical(fclinical, na_values=None):
    """
    read the clinical data and try to guess the types
    """
    import pandas as pa
    if na_values is None:
        na_values = []

    header = open(fclinical).readline().rstrip("\r\n").split("\t")
    conv = dict((x, try_date_parse) for x in header if "date" in x or
            x == "passed_qc")
    na_values.extend(("NA", "na", "NaN", "nan"))
    try:
        return pa.read_table(fclinical, converters=conv, parse_dates=True,
                                na_values=na_values)
    except:
        return pa.read_table(fclinical, parse_dates=True, na_values=na_values)

def _clinical_to_ys(clinical1):
    classes = [x for x in sorted(np.unique(np.array(clinical1)))] # if not np.isnan(x)]
    return classes, np.array([classes.index(c) if c in classes else np.nan for c in clinical1]) # if not np.isnan(c)])

def run(fX, fclinical, header_keys, fig_name, klass, nan_value=0,
        label_key=None, transpose=False):

    clinical = read_clinical(fclinical)
    X_ids, X_probes, X = readX(fX, transpose, nan_value=nan_value)
    assert X.shape[1] == len(X_ids), (X.shape, len(X_ids), len(X_probes))

    ci = map(str, list(clinical[clinical.columns[0]]))
    X_out = [xi for xi in X_ids if not xi in ci]
    if X_out:
        X = X[:, [i for i, xi in enumerate(X_ids) if xi in ci]]
        print >>sys.stderr, "removing %i rows in X but not in clinical" \
                % (len(X_out))

    print >>sys.stderr, X.shape, "after removed"
    X = X.T
    X_ids = np.array([xi for xi in X_ids if xi in ci])
    clinical = clinical.irow([ci.index(xi) for xi in X_ids if not xi in X_out])
    print >>sys.stderr, clinical.shape, "clinical after removed"

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

    #assert X.shape[1] == y.shape[0], (X.shape, y.shape)

    if klass.__name__ == "KernelPCA":
        clf = klass(20, kernel="linear", gamma=3./X.shape[0]).fit(X) #3./X.shape[0]).fit(X)
    elif not ("PCA" in klass.__name__ or "LDA" in klass.__name__):
        clf = klass(6, out_dim=10).fit(X, y)
    else:
        clf = klass(20).fit(X)
    X_r = clf.transform(X)
    print >>sys.stderr, "X_r", X_r.shape

    components = X_r
    assert len(clf.explained_variance_ratio_ == clinical.shape[0])
    #if components.shape[0] != clinical.shape[0]:
    #    components = clf.components_.T

    #print components.shape , X.shape[0] , clinical.shape[0]
    assert components.shape[0] == clinical.shape[0], (components.shape,
            clinical.shape)

    ax = plt.subplot(111, projection='3d')
    proxies = []
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
        zs = components[p, 2]
        assert xs.shape[0] <= len(clinical)

        proxies.append(plt.Circle((0, 0), 0.001, fc=color))

        ax.scatter(xs, ys, zs, c=color, edgecolor=color,
                s=12, label=str(yclass))

        #plt.scatter(xs, ys, c=color, s=6, label=yclass)
        if label_key is not None and i == 0:
            labels = clinical[label_key][p]
            for xx, yy, label in izip(xs, ys, labels):
                plt.text(xx, yy, label, color=color, fontsize=6, multialignment='right')

    exr = clf.explained_variance_ratio_
    labels = [("(%.1f" % (100. * e)) + "%)" for e in exr[:3]]
    plt.xlabel('component 1 ' + labels[0])
    plt.ylabel('component 2 ' + labels[1])
    ax.set_zlabel('component 3 ' + labels[2])

    plt.title(header_key)
    leg = plt.legend(proxies, yclasses, scatterpoints=1, loc='upper left')

    print_correlations(components, clinical, clf.explained_variance_ratio_)

    plt.savefig(fig_name)


def print_correlations(components, clinical, evr):
    n_tests = min(10, components.shape[1]) * (len(clinical.columns) - 1)
    print >>sys.stderr, "n-tests:", n_tests

    print "pc_num\tvariance_explained\tclinical_variable\tn\tR\tanova_groups\tp_value\tbonf_p"
    for i in range(min(10, components.shape[1])):
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
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                R = "%.3f" % r_value
                aov = "na"

            if p_value > 0.1 or np.isnan(p_value): continue
            adj_p = "%.3g" % min(1, (p_value * n_tests))
            p_value = "%.3g" % p_value
            n = len(y)
            evr_j = "%.3f" % evr[i]
            print "%(j)i\t%(evr_j)s\t%(column)s\t%(n)i\t%(R)s\t%(aov)s\t%(p_value)s\t%(adj_p)s" % \
                locals()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                   formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-X", dest="X", help="numerical data file first column is ID"
              " first row is header")
    p.add_argument("-t", dest="transpose", action="store_true", default=False,
        help="data is in rows==probes, columns==samples")
    p.add_argument("-c", dest="clinical", help="clinical data file. first "
              "column matches that in X")
    p.add_argument("-k", dest="key", help="column header(s) to separate classes")
    p.add_argument("-l", dest="label",
        help="optional column header(s) to label points")
    p.add_argument("-m", dest="method", choices=CLASSES.keys(),
                 help="method to use for transformation.",
                 default="RandomizedPCA")
    p.add_argument("-f", dest="fig_name", help="path to save figure")
    p.add_argument("--na", help="value to use instead of nan",
            default=0.0, type=float)

    args = p.parse_args()
    if (None in (args.X, args.clinical, args.key, args.fig_name)):
        sys.exit(not p.print_help())

    run(args.X, args.clinical, args.key.rstrip().split(","), args.fig_name,
        CLASSES[args.method], args.na, args.label, args.transpose)

if __name__ == "__main__":
    import doctest
    if doctest.testmod(optionflags=doctest.ELLIPSIS |\
                                   doctest.NORMALIZE_WHITESPACE).failed == 0:
        main()
