import tensorflow as tf
import time
import os
from scipy.misc import imsave
import skimage.transform
import skimage.color
import numpy as np
from dcgan import ConditionalGAN
from utils import DataManager
np.random.seed(seed=123)

def run_test_epoch(sess, dm, model, PARA):

	if not os.path.exists(PARA.test_img_dir):
		os.makedirs(PARA.test_img_dir)

	total_batch_num = dm.total_batch_num(PARA.batch_size, mode='test')

	for i in range(total_batch_num):

		data, bz = dm.draw_batch(PARA.batch_size, PARA.z_dim, mode='test')
		bh = [d.tags for d in data]
		images = sess.run(model.x_,
			feed_dict={
				model.z:bz,
				model.h:bh,
				model.training:False
			}
		)
		for i, (image, d) in enumerate(zip(images, data)):
			image = (image + 1.0) / 2.0
			img_resized = image

			tag_text = d.tag_text
			img_id = d.img_id
			img_filename = "sample_{}.jpg".format(img_id)
			imsave(os.path.join(PARA.test_img_dir, img_filename), img_resized)


def test(PARA):

	dm = DataManager(PARA.mode,
		PARA.tag_file, PARA.img_dir, PARA.test_text, PARA.vocab, PARA.z_dim, PARA.generator_output_layer)

	with tf.Graph().as_default():

		model = ConditionalGAN(
			PARA.z_dim, 2 * dm.vocab.vocab_size, PARA.learning_rate, PARA.scale, PARA.generator_output_layer)

		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 0.5

		saver = tf.train.Saver(max_to_keep=25)

		def load_pretrain(sess):
			saver.restore(sess, os.path.join(PARA.log, "checkpoint"))

		sv = tf.train.Supervisor(logdir=PARA.log, saver=saver, init_fn=load_pretrain)

		with sv.managed_session(config=config) as sess:

			run_test_epoch(sess, dm, model, PARA)


if __name__ == "__main__":
	tf.flags.DEFINE_integer("mode", 1, "") # 1=test
	tf.flags.DEFINE_integer("batch_size", 100, "")
	tf.flags.DEFINE_integer("z_dim", 100, "")

	tf.flags.DEFINE_float("scale", 10.0, "")
	tf.flags.DEFINE_float("learning_rate", 2e-4, "")

	tf.flags.DEFINE_string("tag_file", "./data/tags_clean.csv", "")
	tf.flags.DEFINE_string("img_dir", "./data/faces/", "")
	tf.flags.DEFINE_string("test_text", "./data/sample_testing_text.txt", "")
	tf.flags.DEFINE_string("vocab", "./vocab", "")
	tf.flags.DEFINE_string("log", "./model", "")
	tf.flags.DEFINE_string("test_img_dir", "./samples/", "")
	tf.flags.DEFINE_string("generator_output_layer", 'tanh', "")

	PARA = tf.flags.FLAGS
	PARA._parse_flags()
	test(PARA)
