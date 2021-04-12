## OpenWGL: Open-World Graph Learning

This repository contains the author's implementation Tensorflow in  for our ICDM 2020 paper "OpenWGL: Open-World Graph Learning".


## Dependencies

- Python (>=3.6)
- Tensorflow/Tensorflow-gpu: (>= 1.14.0 or >= 2.0.0b1)
- numpy (>=1.17.4)
- scipy (>= 1.1.0)
- tf_geometric (>=1.0)

## Datasets
The data folder can contain many graph datasets, and we provide the default Cora dataset.

## Implementation

Here we provide the implementation of OpenWGL, along with the default dataset (Cora). The repository is organised as follows:

 - `data/` contains the necessary dataset files (more datasets can be found in [tf_geometric](https://github.com/CrawlScript/tf_geometric/tree/master/tf_geometric/datasets));
 - `opgl/` contains the implementation of the VGAE and the basic utils;

 Finally, `OpenWGL_demo.py` puts all of the above together and can be used to execute a full training run on the datasets.

## Process
 - Place the datasets in `data/`
 - Change the `dataset` in `OpenWGL_demo.py` .
 - Training/Testing:
 ```bash
 python OpenWGL_demo.py
 ```
- Or 

```bash
 python OpenWGL_demo.py --dataset_name cora --unseen_num 1
```

# Citation

```
@inproceedings{wu2020OPGL
author={Man Wu and Shirui Pan and Xingquan Zhu},
title={OpenWGL: Open-World Graph Learning},
booktitle={20th {IEEE} International Conference on Data Mining, ICDM},
year={2020}
}
```
## Some of the code was forked from the following repository

[tf_geometric](https://github.com/CrawlScript/tf_geometric)