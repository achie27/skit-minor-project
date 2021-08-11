import numpy as np
import tensorflow as tf

class SketchRec():
	def __init__(self, max_len, total_cats):
		
		self.max_len = max_len
		self.total_cats = total_cats


	def conv_layers(self, batch):
		conv1 = tf.layers.conv1d(
			batch, filters = 48, kernel_size = 5, padding = "same", activation = None   
		)

		conv2 = tf.layers.conv1d(
			conv1, filters = 64, kernel_size = 5, padding = "same", activation = None
		)

		conv3 = tf.layers.conv1d(
			conv2, filters = 96, kernel_size = 3, padding = "same", activation = None
		)

		return conv3


	def lstm_layers(self, inp, lengths):

		cell = tf.nn.rnn_cell.BasicLSTMCell
		cells_bw = [cell(128), cell(128)]
		cells_fw = [cell(128), cell(128)]

		op, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
		    cells_fw = cells_fw,
		    cells_bw = cells_bw,
		    inputs = inp,
		    sequence_length = lengths,
		    dtype = tf.float32
		)
		mask = tf.tile(
		    tf.expand_dims(
		        tf.sequence_mask(
		            lengths, tf.shape(op)[1]
		        ), 2
		    ), 
		    [1, 1, tf.shape(op)[2]]
		)
		zero_outside = tf.where(mask, op, tf.zeros_like(op))
		op = tf.reduce_sum(zero_outside, axis=1)

		return op

	def fc_layers(self, inp):
		l = tf.layers.dense(inp, self.total_cats)
		return l

	def network(self, inks, labels, shapes, length):
		lengths = tf.squeeze(tf.slice(
			shapes, begin=[0, 0], size = [length, 1]
		))

		convolved = self.conv_layers(inks)
		lstmed = self.lstm_layers(convolved, lengths)
		logits = self.fc_layers(lstmed)
		return logits

	def predict(self, inks, shapes, labels, ink_predict, 
			labels_predict, shapes_predict, model = 'mark1'):
		
		op = self.network(inks, labels, shapes, len(ink_predict))
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			saver = tf.train.Saver()
			saver.restore(sess, model+'.ckpt') 

			return (
				sess.run(
					tf.argmax(op, 1), feed_dict = {
						inks : ink_predict,
						labels : labels_predict,
						shapes : shapes_predict
					}
				)
			)

if __name__ == "__main__":
	

	from dataset_parser import unpack_drawings
	from sklearn.metrics import classification_report
	import glob

	sketches = []
	shpes = []
	lbls = []

	mx_len = 250
	ttal_cats = 30

	inks = tf.placeholder('float32', [None, mx_len, 3])
	labels = tf.placeholder('int32', None)
	shapes = tf.placeholder('int32', [None, 2])

	files = glob.glob('./dataset/*.bin')
	d = [unpack_drawings(f) for f in files[:2]]
	cnt = 0

	for n, cat in enumerate(d):
		cnt, lulz = 0, 0
		for sketch in cat:
			if lulz < 50:
				lulz+=1
				continue

			if cnt >= 10:
				break
	        
			inkarray = sketch['image']
			stroke_lengths = [len(stroke[0]) for stroke in inkarray]
			total_points = sum(stroke_lengths)
			if total_points > mx_len :
				continue

			cnt+=1    
			np_ink = np.zeros((total_points, 3), dtype=np.float32)
			current_t = 0
			for stroke in inkarray:
				for i in [0, 1]:
					np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
				current_t += len(stroke[0])
				np_ink[current_t - 1, 2] = 1  # stroke_end
	    
			lower = np.min(np_ink[:, 0:2], axis=0)
			upper = np.max(np_ink[:, 0:2], axis=0)
			scale = upper - lower
			scale[scale == 0] = 1
			np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale

			np_ink = np_ink[1: ] - np_ink[0:-1]
			np_ink[np_ink[:, 2] == -1] = 0

			sketches.append(np_ink)
			shpes.append(np_ink.shape)
			lbls.append(n)


	for i in range(len(sketches)):
		sh = shpes[i][0]
		if mx_len-sh <= 0:
			continue

		x = np.zeros((mx_len-sh, 3))
		if sketches[i][-1][2] == 0 :
			x[0][2] = 1

		sketches[i] = np.vstack((sketches[i], x))

	tob = SketchRec(mx_len, ttal_cats)
	op = tob.predict(inks, shapes, labels, sketches, lbls, shpes, model = 'mk2')
	print(classification_report(op, lbls))

"""

import numpy as np
import tensorflow as tf

batch_size, max_len = 5, 250
total_cats = 30

inks = tf.placeholder('float32', [None, max_len, 3])
labels = tf.placeholder('int32', None)
shapes = tf.placeholder('int32', [None, 2])

def conv_layers(batch):
  conv1 = tf.layers.conv1d(
      batch, filters = 48, kernel_size = 5, padding = "same", activation = None   
  )

  conv2 = tf.layers.conv1d(
      conv1, filters = 64, kernel_size = 3, padding = "same", activation = None
  )

  return conv2


def lstm_layers(inp, lengths):

  cell = tf.nn.rnn_cell.BasicLSTMCell
  cells_bw = [cell(128), cell(128)]
  cells_fw = [cell(128), cell(128)]

  op, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
      cells_fw = cells_fw,
      cells_bw = cells_bw,
      inputs = inp,
      sequence_length = lengths,
      dtype = tf.float32
  )
  mask = tf.tile(
      tf.expand_dims(
          tf.sequence_mask(
              lengths, tf.shape(op)[1]
          ), 2
      ), 
      [1, 1, tf.shape(op)[2]]
  )
  zero_outside = tf.where(mask, op, tf.zeros_like(op))
  op = tf.reduce_sum(zero_outside, axis=1)

  return op

def fc_layers(inp):
  l = tf.layers.dense(inp, total_cats)
  return l

def network(inks, labels, shapes, length):
  lengths = tf.squeeze(tf.slice(shapes, begin=[0, 0], size = [length, 1]))    
  convolved = conv_layers(inks)
  lstmed = lstm_layers(convolved, lengths)
  logits = fc_layers(lstmed)
  return logits

def predict(inks, shapes, labels, ink_predict, labels_predict, shapes_predict, model = 'mark1'):
  op = network(inks, labels, shapes, len(ink_predict))
  with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver()
      saver.restore(sess, model+'.ckpt') 

      return (
          sess.run(
              tf.argmax(op, 1), feed_dict = {
                  inks : ink_predict,
                  labels : labels_predict,
                  shapes : shapes_predict
              }
          )
      )

if __name__ == "__main__":
    

  from dataset_parser import unpack_drawings
  from sklearn.metrics import classification_report
  import glob

  sketches = []
  shpes = []
  lbls = []

  files = glob.glob('./dataset/*.bin')
  d = [unpack_drawings(f) for f in files[:10]]
  cnt = 0

  for n, cat in enumerate(d):
      cnt, lulz = 0, 0
      for sketch in cat:
          if lulz < 50:
              lulz+=1
              continue

          if cnt >= 200:
              break
            
          inkarray = sketch['image']
          stroke_lengths = [len(stroke[0]) for stroke in inkarray]
          total_points = sum(stroke_lengths)
          if total_points > max_len :
              continue

          cnt+=1    
          np_ink = np.zeros((total_points, 3), dtype=np.float32)
          current_t = 0
          for stroke in inkarray:
              for i in [0, 1]:
                  np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
              current_t += len(stroke[0])
              np_ink[current_t - 1, 2] = 1  # stroke_end
        
          lower = np.min(np_ink[:, 0:2], axis=0)
          upper = np.max(np_ink[:, 0:2], axis=0)
          scale = upper - lower
          scale[scale == 0] = 1
          np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale

          np_ink = np_ink[1: ] - np_ink[0:-1]
          np_ink[np_ink[:, 2] == -1] = 0

          sketches.append(np_ink)
          shpes.append(np_ink.shape)
          lbls.append(n)


  for i in range(len(sketches)):
      sh = shpes[i][0]
      if max_len-sh <= 0:
          continue

      x = np.zeros((max_len-sh, 3))
      if sketches[i][-1][2] == 0 :
          x[0][2] = 1

      sketches[i] = np.vstack((sketches[i], x))

  op = predict(inks, shapes, labels, sketches, lbls, shpes, model = 'mk2')
  print(classification_report(op, lbls))

"""