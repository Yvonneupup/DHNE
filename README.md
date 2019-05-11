# DHNE
This repository contains implementation of DHNE : Network Representation Learning Method for Dynamic Heterogeneous Network.

DHNE combines the historical information into current information in the netwrok to learn the representations of nodes in dynamic heterogeneous networks . 

## Requirement
*  python 3.4 (or later)
*  networkx 1.11
*  gensim 2.3.0

## To run the DHNE algorithm:
Please use *--dataset <dataset-name>* argument, where *dataset-name* can be one of the following: "Dblp", "Aminer".

```
cd code
python DHNE.py --dataset Aminer
```

The output will be saved in */Aminer/aminer_result* folder

## Data
We experiment on two real-world datasets: DBLP, Aminer datasets
*  Folder "Dblp/dblp_dataset" contains DBLP dataset graphs. There are 19 graphs from 2000 to 2018 .
*  Folder "Aminer/aminer_dataset" contains Aminer dataset graphs. There are 16 graphs from 1990 to 2005.
