系统下载地址：
	链接：https://pan.baidu.com/s/1Q2kCdSBxlcygZ48stxHpPA 
	提取码：zm2y 

1. 使用前需安装python3、tensorflow2.0

2. 一共四个入口，对应四个功能：
	main.py		用于分词
	train.py		用于训练并保存模型
	train_embedding.py	用于训练字向量
	evaluate.py	用于给测试集分词并评估分词结果

3. .py文件都必须与data文件夹在同一目录

4. data下的文件：
	根目录：
		chinese_wiki.txt	预训练语料
		train.csv		训练集
		test.csv		与train.csv同领域的测试集
		test_weibo.xlsx	微博评论测试集
	model文件夹：
		bilstm_wiki.h5	最终版bilstm模型，非字嵌入
		bilstm_wiki_emb.h5	最终版bilstm模型，字嵌入（最优）
		bilstm+lstm.h5	Baseline，非字嵌入
		bilstm+lstm_emb.h5	Baseline，字嵌入
		dictionary.pkl	train.csv的字典，用于评估登录词、非登录词召回率
		hmm.model	极大似然估计HMM模型，非字嵌入
		hmm_emb.model	极大似然估计HMM模型，字嵌入
		trans.pkl		状态转移概率
		w2v_passage.model	用train.csv训练的字向量预训练模型
		w2v_wiki.model	用chinese_wiki.txt训练的字向量预训练模型
	evaluate文件夹：
		内部文件为使用不同分词模型对不同测试集进行分词的结果
	check_points文件夹：
		Keras的check_points