# tf_model_object_detection_config
一、下载coco-trained models:
github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
把下载的模型放到dataset文件夹中，例如：
	faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
	然后解压：
	tar -vxf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
	生成faster_rcnn_inception_v2_coco_2018_01_28文件夹目录
	
faster_rcnn_inception_v2_coco_2018_01_28/
├── checkpoint
├── frozen_inference_graph.pb
├── model.ckpt.data-00000-of-00001
├── model.ckpt.index
├── model.ckpt.meta
├── pipeline.config
└── saved_model
    ├── saved_model.pb
    └── variables
	
	将里面的pipeline.config复制到dataset文件夹中，并按照本文件夹的名字重新命名（这样的目的是为了规范，当你拥有很多个模型时就不会乱）
	
二、数据集的处理

1.将数据集放到JPEGImages文件夹中，xml标签文件放到Annotations文件夹中

2.执行:

-----------------------
python classfier.py
--------------------

生成ImageSets/Main文件夹下的三个文件:test.txt ,train.txt,val.txt以及一个trainval.txt


3.cd到dataset目录，生成tfrecord，包括验证集，测试集和训练集

--------------------------
python ./create_dataset_tf_record.py --data_dir=./VOCdevkit --year=VOC2007 --label_map_path=./dataset_label_map.pbtxt --set=val --output_path=./tf_text_data/dataset_val.record

python ./create_dataset_tf_record.py --data_dir=./VOCdevkit --year=VOC2007 --label_map_path=./dataset_label_map.pbtxt --set=val --output_path=./tf_text_data/dataset_test.record

python ./create_dataset_tf_record.py --data_dir=./VOCdevkit --year=VOC2007 --label_map_path=./dataset_label_map.pbtxt --set=train --output_path=./tf_text_data/dataset_train.record

---------------

4.修改dataset_label_map.pbtxt，有几类写几类，里面的标签必须和xml文件中的一致

三、修改pipeline.config文件
总共包括四个部分：
model：
    主要修改：
    	num_classes(分类的类别数）
    	    image_resizer {
      fixed_shape_resizer {
        height: 300
        width: 300      （输入网络图像的大小尺寸，一般默认）
      }
    }
    		
train_config：	
    主要修改：
	batch_size: 64（按照电脑的配置来定）
	
	############################################################################
	####       如果你想使用faster-rcnn，batch_size只能设为1      ################
	############################################################################
	
	initial_learning_rate: 0.0023（初始学习率，可以自己调节）
	decay_steps: 600（每多少个steps变化一次学习率）
	decay_factor: 0.96（每次变化的学习率：current_learning_rate = decay_factor*initial_learning_rate）
	num_steps: 50000（训练的次数）
	
	fine_tune_checkpoint: "#####/output/model.ckpt-18508"（迁移学习的模型文件）
	
train_input_reader：
	input_path: "#####/tf_text_data/dataset_train.record"（训练集的位置）
	label_map_path: "#####/dataset_label_map.pbtxt"（标签的定义文件）
	
eval_config：
	metrics_set:"coco_detection_metrics"（测量的方式，主要是用mAP来度量）
	num_examples: 300(验证集的数量)
	max_evals: 1（验证的循环次数）
	use_moving_averages:false（采用滑动平均）
	
eval_input_reader：
	tf_record_input_reader {
    	input_path: "#####/tf_text_data/dataset_val.record"（验证集的位置）
 	 }
	label_map_path: "#####/dataset_label_map.pbtxt"（标签的定义文件）
	
四、训练

对于ssd：
--------------------
python ../research/object_detection/legacy/train.py --logtostderr --train_dir=./output --pipeline_config_path=./ssd_mobilenet_v1_coco_2018_01_28/pipeline.config
-----------------------
对于faster-rcnn:
---------------------
python ../research/object_detection/legacy/train.py --logtostderr --train_dir=./output --pipeline_config_path=./faster_rcnn_resnet101_coco_2018_01_28/pipeline.config
--------------------

五、保存节点pb
----------------------
python ../research/object_detection/export_inference_graph.py input_type image_tensor --pipeline_config_path ./ssd_mobilenet_v1_coco_2018_01_28/pipeline.config --trained_checkpoint_prefix ./output/model.ckpt-50000 --output_directory ./output

python ../research/object_detection/export_inference_graph.py input_type image_tensor --pipeline_config_path ./faster_rcnn_resnet101_coco_2018_01_28/pipeline.config --trained_checkpoint_prefix ./output/model.ckpt-0 --output_directory ./output

------------------------

六、生成pbtxt文件，opencv调用需要，生成pbtxt的py文件位于opencv源码文件/samples/dnn中，
地址：github.com/opencv/opencv/tree/4.5.2/samples/dnn/tf_text_graph_ssd.py 

-------------------------
python ./tf_text_graph/tf_text_graph_ssd.py --input=./output/frozen_inference_graph.pb --output=./output/frozen_inference_graph.pbtxt  --config=./ssd_mobilenet_v1_coco_2018_01_28/pipeline.config

-------------------------
	
七、测试图片
打开object_detection_test.py文件,修改：
模型位置：
MODEL_NAME = './output'
模型名
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
模型标签文件
PATH_TO_LABELS = './dataset_label_map.pbtxt'
类别数
NUM_CLASSES = 7


----------------------------------------
python ./object_detection_test.py --image ./VOCdevkit/VOC2007/JPEGImages/v03_062250.jpg
----------------------------------------
或者：
-------------------------------------------------------
python ./object_detection_test.py  --data ./VOCdevkit/VOC2007/ImageSets/Main/val.txt
-------------------------------------------------------

