clinco
======

Work In Progress.

Summarize the clinical (or lab) components of your dataset.
Get an overview of the associations and correlations in your
dataset.

What is clinco?
---------------

clinco is designed to give you a comprehensive view of your
dataset so that you can design your analyses. It let's you
address questions like:

1. Does my data have batch effects?
2. Is age correlated with expression (or methylation or ...)?
3. Does atopy vary by asthma status (correlations within clinical variables)
4. Which clinical variables are associated with the first X
   principal components?


Usage
-----

Everything is dispatched from the single command-line executable:

```Shell
clinco
```

Data Format
-----------

clinical data format
++++++++++++++++++++

Clinco expects that your clinical-information is in tab-delimited
format with a header
indicating the clinical variable in each column one row for each
sample. The first column must be some unique ID if the clinical data is to
be join to some numerical data such as expression or methylation.

An example with 3 samples would be:

```
#individual_id	age	sex	date_processed	location
individual_01	22	M	14-01-2012	Denver
individual_02	29	F	22-11-2011	Boston
individual_03	20	F	12-03-2012	Denver
```

numerical data format
+++++++++++++++++++++

The numerical data is expected to have n_samples * (n_features + 1).
Where each feature is a probe or a measurement. The first column is
expected to be an ID that maps that sample back to the clinical data.
An example matching the above clinical data is:

```
#individual_id	probe1	probe2	probe3	probe4
individual_01	0.11	0.91	0.93	0.14
individual_01	0.01	0.71	0.72	0.01
individual_01	0.14	0.99	0.99	0.16
```


