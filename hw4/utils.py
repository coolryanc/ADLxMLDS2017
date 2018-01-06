import numpy as np
import skimage
import skimage.io
import skimage.transform
from collections import defaultdict, Counter
import pickle
import copy
import csv
import os
import random
import time
import scipy.stats as stats
import re

class Vocab(object):
	def __init__(self, min_count=5, vocab_path=None, vocab_text_list=None):
		self.w2i = {} # word to index
		self.i2w = {} # index to word
		self.word_count = defaultdict(int)
		self.total_words = 0
		self.vocab_size = 0
		self.unknown = "_UNK_"
		self.add_word(self.unknown, count=0)
		self.min_count = min_count
		if vocab_text_list:
			self.construct(vocab_text_list, min_count)
			pickle.dump(self, open(vocab_path, "wb"))
		elif vocab_path:
			self.__dict__.update(pickle.load(open(vocab_path, "rb")).__dict__)

	def dump(self, vocab_path):
		pickle.dump(self, open(vocab_path, "wb"))

	def add_word(self, word, count=1):
		if word not in self.w2i:
			index = len(self.w2i)
			self.w2i[word] = index
			self.i2w[index] = word
			self.vocab_size += 1
		self.word_count[word] += count

	def construct(self, words, min_count):
		self.cnt = Counter(words)
		for word, count in self.cnt.most_common():
			if count >= min_count:
				self.add_word(word, count=count)
		self.total_words = sum(self.word_count.values())
		self.vocab_size = len(self.word_count)

	def encode(self, word):
		if word not in self.w2i:
			word = self.unknown
		return self.w2i[word]

	def decode(self, index):
		return self.i2w[index]

class Data(object):
	def __init__(self, img_id, img, tags, tag_text, with_text=1):
		self.img_id = img_id
		self.img = img
		self.tags = tags
		self.tag_text = tag_text
		self.with_text = with_text

class Sampler(object):
	def __init__(self):
		pass
	def sample(self, batch_size, z_dim):
		return np.random.normal(0, 1, size=[batch_size, z_dim])

class DataManager(object):
	def __init__(self,
				 mode,
				 tag_file_path=None,
				 img_dir_path=None,
				 test_text_path=None,
				 vocab_path=None,
				 z_dim=100,
				 generator_output_layer='tanh'):

		self.index = {'train':0, 'test':0}
		self.tag_num = 2
		self.label_data = {}
		self.nega_data = {}
		self.z_sampler = Sampler()
		self.unk_counter = 0
		self.generator_output_layer = generator_output_layer

		if mode == 1: # test
			self.vocab = Vocab(vocab_path=vocab_path)
			self.test_data = self.load_test_data(test_text_path, self.vocab)

		elif mode == 2: # training
			self.train_data, self.vocab = self.load_train_data(tag_file_path, img_dir_path)
			self.vocab.dump(vocab_path)

	def load_train_data(self, tag_file_path, img_dir_path):
		hairColor = ['orange hair', 'white hair', 'aqua hair', 'gray hair', \
					 'green hair', 'red hair', 'purple hair', 'pink hair', \
					 'blue hair', 'black hair', 'brown hair', 'blonde hair', 'bicolored hair']
		eyeColor = ['gray eyes', 'black eyes', 'orange eyes', \
					'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes', \
					'green eyes', 'brown eyes', 'red eyes', 'blue eyes']
		colors = []
		for c in hairColor:
			h_c = c.split(' ')[0]
			if h_c not in colors:
				colors.append(h_c)
		for c in eyeColor:
			e_c = c.split(' ')[0]
			if e_c not in colors:
				colors.append(e_c)
		vocab = Vocab(min_count=0)

		for color in colors:
			vocab.add_word(color)
		data = []
		with open(tag_file_path, "r") as tag_f:
			reader = csv.reader(tag_f)
			for img_id, tag_str in reader:
				# convert img_id to integer
				img_id = int(img_id)

				tags = [s.split(":")[0].strip() for s in tag_str.lower().split("\t")]
				hair = [t.split(" ")[0] for t in tags if t.endswith('hair')]
				eyes = [t.split(" ")[0] for t in tags if t.endswith('eyes')]

				# filter all not color
				hair = [vocab.encode(h) for h in hair if h in vocab.w2i and vocab.encode(h) != vocab.unknown]
				eyes = [vocab.encode(e) for e in eyes if e in vocab.w2i and vocab.encode(e) != vocab.unknown]
				# skip no eye hair tag data
				if len(hair) == 0 and len(eyes) == 0:
					continue
				# skip > 1 hair or >1 eyes, because they are someone not shown in images
				if len(hair) > 1 or len(eyes) > 1:
					continue

				with_text = 1
				with_unk = 0

				if len(hair) == 0 or len(hair) > 1 or len(eyes) == 0 or len(eyes) > 1:

					if len(hair) == 1:
						eyes = [vocab.encode(vocab.unknown)]
						with_unk = 1

					elif len(eyes) == 1:
						hair = [vocab.encode(vocab.unknown)]
						with_unk = 1

					else:
						hair = []
						eyes = []
						with_text = 0
						with_unk = 1

				hair_str = [vocab.decode(h) for h in hair]
				eyes_str = [vocab.decode(e) for e in eyes]
				tag_text = "{}_hair_{}_eyes".format("_".join(hair_str), "_".join(eyes_str))

				hair = set(hair)
				eyes = set(eyes)
				feature = np.zeros((self.tag_num * vocab.vocab_size))

				for c_id in hair:
					feature[c_id] += 1
				for c_id in eyes:
					feature[c_id + vocab.vocab_size] += 1

				# image
				img_path = os.path.join(img_dir_path, str(img_id) + ".jpg")
				# convert img to -1, 1
				img = skimage.io.imread(img_path) / 127.5 - 1
				# resize to 64 * 64
				img_resized = skimage.transform.resize(img, (64, 64), mode='constant')
				no_text = "{}_hair_{}_eyes".format('', '')

				if tag_text == no_text:
					feature = np.zeros((self.tag_num * vocab.vocab_size)) / (vocab.vocab_size)

				for angle in [-20, -10, 0, 10, 20]:
					img_rotated = skimage.transform.rotate(img_resized, angle, mode='edge')
					for flip in [0, 1]:
						if flip:
							d = Data(img_id, np.fliplr(img_rotated), feature, tag_text, with_text)
						else:
							d = Data(img_id, img_rotated, feature, tag_text, with_text)
						if tag_text not in self.label_data:
							self.label_data[tag_text] = []
						if with_text:
							self.label_data[tag_text].append(d)
						if with_unk:
							self.unk_counter += 1
						data.append(d)
		return data, vocab

	def load_test_data(self, test_text_path, vocab):
		data = []

		with open(test_text_path, "r") as f:
			reader = csv.reader(f)
			for text_id, text in reader:
				text_id = int(text_id)

				text_list = text.lower().split(" ")
				hair_color_id = vocab.encode(text_list[0])
				eyes_color_id = vocab.encode(text_list[2])

				feature = np.zeros((self.tag_num * vocab.vocab_size))

				feature[hair_color_id] += 1
				feature[eyes_color_id + vocab.vocab_size] += 1

				for img_id in range(1, 5+1):
					data.append(Data("{}_{}".format(text_id, img_id), None, feature, text.lower().replace(" ", "_")))

		return data

	def draw_batch(self, batch_size, z_dim, mode='train'):
		if mode == 'train':
			data = self.train_data[self.index['train'] : self.index['train'] + batch_size]
			if self.index['train'] + batch_size >= len(self.train_data):
				self.index['train'] = 0
				np.random.shuffle(self.train_data)
			else:
				self.index['train'] += batch_size
			noise = self.z_sampler.sample(len(data), z_dim)
			noise_h = []
			wrong_img = []
			for d in data:
				nega_d = random.sample(self.nega_data[d.tag_text], 1)[0]
				noise_h.append(nega_d.tags)
				wrong_img.append(nega_d.img)
			return data, noise, noise_h, wrong_img

		if mode == 'test':
			data = self.test_data[self.index['test'] : self.index['test'] + batch_size]
			if self.index['test'] + batch_size >= len(self.test_data):
				self.index['test'] = 0
			else:
				self.index['test'] += batch_size
			noise = self.z_sampler.sample(len(data), z_dim)
			return data, noise

	def total_batch_num(self, batch_size, mode='train'):

		if mode == 'train':
			return int(np.ceil(len(self.train_data) / batch_size))

		if mode == 'test':
			return int(np.ceil(len(self.test_data) / batch_size))
