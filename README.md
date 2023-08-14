# SequenceTaggingNER

通过序列标注方法抽取实体，可以采用BIOES或BIO形式进行标注

解码方式可以选择softmax或者CRF


## 指标计算

对各个类别的实体分别统计TP，FP及FN数量，计算precision，recall与f1

对总体计算 micro precision、micro recall、micro f1，macro precision、macro recall、macro f1与weighted precision、weighted recall、weighted f1

## 训练

将预训练模型的路径修改为本地路径，其他参数可以使用默认值也可以自行修改

python src/train.py --pretrained_model_path local_path