import tensorflow as tf
import time
import os
from scipy.misc import imsave
import skimage.transform
import skimage.color
import numpy as np
from dcgan import ConditionalGAN
from utils import DataManager

def run_train_epoch(epoch, sess, dm, model):
	total_batch_num = dm.total_batch_num(PARA.batch_size)
	maxval = total_batch_num
	total_d_loss = 0
	d_count = 0
	total_g_loss = 0
	g_count = 0
	for i in range(total_batch_num):

		data, bz, bh_, bx_w_ = dm.draw_batch(PARA.batch_size, PARA.z_dim, mode='train')
		bx = [d.img for d in data]
		bh = [d.tags for d in data]
		bwith_text = [d.with_text for d in data]

		if i % 1 == 0:
			for d_i in range(PARA.d_epochs):
				_, d_loss = sess.run([model.d_opt, model.d_loss],
					feed_dict={
						model.x:bx,
						model.z:bz,
						model.h:bh,
						model.h_:bh_,
						model.x_w_:bx_w_,
						model.training:True,
						model.with_text:bwith_text
					}
				)
				total_d_loss += d_loss
				d_count += 1

		for g_i in range(PARA.g_epochs):
			_, g_loss = sess.run([model.g_opt, model.g_loss],
				feed_dict={
					model.x:bx,
					model.z:bz,
					model.h:bh,
					model.h_:bh_,
					model.x_w_:bx_w_,
					model.training:True,
					model.with_text:bwith_text
				}
			)
			total_g_loss += g_loss
			g_count += 1

	return total_d_loss / d_count, total_g_loss / g_count

def train(PARA):

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

		if PARA.load == 1:
			sv = tf.train.Supervisor(logdir=PARA.log, saver=saver, init_fn=load_pretrain)
		elif PARA.load != 1:
			sv = tf.train.Supervisor(logdir=PARA.log, saver=saver)

		with sv.managed_session(config=config) as sess:

			for epoch in range(PARA.epochs):
				print(epoch, end= " ")
				d_loss, g_loss = run_train_epoch(epoch, sess, dm, model)
				print("d_loss: {}, g_loss: {}".format(d_loss, g_loss))

if __name__ == "__main__":
	tf.flags.DEFINE_integer("mode", 2, "") # 1=test, 2=train
	tf.flags.DEFINE_integer("epochs", 200, "")
	tf.flags.DEFINE_integer("d_epochs", 1, "") # discriminator update epochs
	tf.flags.DEFINE_integer("g_epochs", 1, "") # generator update epochs
	tf.flags.DEFINE_integer("batch_size", 100, "")
	tf.flags.DEFINE_integer("z_dim", 100, "")
	tf.flags.DEFINE_integer("load", 0, "") # load model

	tf.flags.DEFINE_float("scale", 10.0, "")
	tf.flags.DEFINE_float("learning_rate", 1e-4, "")

	tf.flags.DEFINE_string("tag_file", "./data/tags_clean.csv", "")
	tf.flags.DEFINE_string("img_dir", "./data/faces/", "")
	tf.flags.DEFINE_string("test_text", "./data/sample_testing_text.txt", "")
	tf.flags.DEFINE_string("vocab", "./vocab", "Model vocab path")
	tf.flags.DEFINE_string("log", "./model", "Model log directory")
	tf.flags.DEFINE_string("test_img_dir", "./samples/", "")
	tf.flags.DEFINE_string("generator_output_layer", 'tanh', "")

	PARA = tf.flags.FLAGS
	PARA._parse_flags()
	train(PARA)
