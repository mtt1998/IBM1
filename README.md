# IBM1
implement ibm model 1 using Python

# Model
IBM1模型在建模翻译模型p(f|e)的时候引入了对齐的隐变量变量a,并且假设设所有可能的对齐是均匀分布的。\
 ![image](/IMG/eq1.png)\

采用EM算法进行参数学习：\
E步：\
 ![image](/IMG/e-step.png)
 ![image](/IMG/e-step2.png)\
 M步:\
 把后验概率当做"伪计数"，进行参数的更新
 
# code
model.train: 使用EM算法训练\
model.align: 给定翻译句子对,输出概率最大的对齐
