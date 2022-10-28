第一次先用  
```
sh get_mrr.sh
```
进行预处理。
注意，如果增加了新的打分
- get_mrr.py 更改 100行处的num_models=x
- get_mrr.py 增加 默认权重个数
- get_mrr.sh 修改 存储位置
- get_mrr.sh 增加 model位置
- config_v2.yaml 修改 存储位置
- config_v2.yaml 增加 model位置
- searchspace_v2.json 增加 搜索权重
- 


之后  
```
pip install nni==2.8
sh nni.sh
```
打开超链接即可看到运行情况。

新增base搜索
为了搜索base需要在初始化时统计每个节点的出现次数，生成valid_can_count_matrix.npy。
base的搜索空间目前采用normal