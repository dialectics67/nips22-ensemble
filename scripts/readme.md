第一次先用  
```
sh get_mrr.sh
```
进行预处理。
注意，如果增加了新的打分
- get_mrr.py 更改100行处的num_models=x
- get_mrr.py 增加默认权重个数
- get_mrr.sh 修改存储位置
- get_mrr.sh 增加model位置
- config_v2.yaml 修改存储位置
- config_v2.yaml 增加model位置
- searchspace_v2.json 增加搜索权重
- 


之后  
```
pip install nni==2.8
sh nni.sh
```
打开超链接即可看到运行情况。