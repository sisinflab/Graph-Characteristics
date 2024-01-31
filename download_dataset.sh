#!/bin/bash


mkdir data 

mkdir ./data/yelp2018/ ./data/yelp2018/raw
mkdir ./data/amazon-book ./data/amazon-book/raw
mkdir ./data/gowalla ./data/gowalla/raw

curl -o ./data/yelp2018/raw/train.csv https://raw.githubusercontent.com/tanatosuu/svd_gcn/main/datasets/yelp/train_sparse.csv
curl -o ./data/yelp2018/raw/test.csv https://raw.githubusercontent.com/tanatosuu/svd_gcn/main/datasets/yelp/test_sparse.csv

curl -o ./data/amazon-book/raw/train.txt https://raw.githubusercontent.com/kuandeng/LightGCN/master/Data/gowalla/train.txt
curl -o ./data/amazon-book/raw/test.txt https://raw.githubusercontent.com/kuandeng/LightGCN/master/Data/gowalla/test.txt

curl -o ./data/gowalla/raw/train.txt https://raw.githubusercontent.com/huangtinglin/Knowledge_Graph_based_Intent_Network/main/data/amazon-book/train.txt
curl -o ./data/gowalla/raw/test.txt https://raw.githubusercontent.com/huangtinglin/Knowledge_Graph_based_Intent_Network/main/data/amazon-book/test.txt


