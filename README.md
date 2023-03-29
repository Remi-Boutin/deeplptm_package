# The Deep Latent Topic Model package

This code is provided with the **Deep-LPTM** paper. 
The package can be used directly in command line or in a python script as described below.


## Command line
This section provides an example of usage of the package in the command line. 
Note that the folder at the *data_path* location is assumed to hold the binary adjacency matrix in
``adjacency.csv``, and the texts in ``texts.csv`` as a matrix T, such that $$T[i,j]$$ = doc sent from node $i$ to node $j$ 


<code>
python main.py -K 3 -Q 4 --data_path <i>data_path</i> --save_results <i>True</i>
--save_path <i>results_path</i>  --init_type <i>dissimilarity</i> --max_iter 10 --tol 0.001  --initialise_etm <i>True</i>
</code>
Other arguments can be provided.


## Function
To use the package in a script, the following arguments are required:
- ``adj``: (array) binary adjacency matrix of the directed graph
- ``W``  : (list of str) texts corresponding  to the edges of the graph
- ``Q``  : (int) number of clusters
- ``K``  : (int) number of topics

The following function fits Deep-LPTM and provides the results in a dictionary:

```python
from deeplptm_package import deeplptm 
results = deeplptm(adj, W, Q, K)
 ``` 

Remark: all the arguments that can be provided to ``deeplptm()`` are provided in
the documentation. To access to this information, use the command ```help(deeplptm)``` in a Python terminal,
after loading the package.

If you are interested in handling the text preprocessing yourselves, as well as getting your hands on the model, 
you can follow the tutorial notebook at ``doc/Tutorial_deep-LPTM.ipynb``.


