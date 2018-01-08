import tensorflow as tf

class DetectionGraph():

	MODEL_NAME = '../ssd_mobilenet_v1_coco_11_06_2017'
	PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
	
	def __init__(self):
		self.__detection_graph = tf.Graph()
		self.__load_model()

	def __load_model(self):
		with self.__detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(DetectionGraph.PATH_TO_CKPT, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

	def get_model(self):
		return self.__detection_graph
