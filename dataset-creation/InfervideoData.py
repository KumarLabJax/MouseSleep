import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import cv2
import imageio
import os, sys
import scipy.ndimage.morphology as morph
import argparse
import multiprocessing as mp
from time import time

output_order = ['m00','m10','m01','m20','m11','m02','m30','m21','m12','m03','mu20','mu11','mu02','mu30','mu21','mu12','mu03','nu20','nu11','nu02','nu30','nu21','nu12','nu03']

def writeCSVHeader(filename):
	writer = open(filename, 'w')
	writer.write("m00,m10,m01,m20,m11,m02,m30,m21,m12,m03,mu20,mu11,mu02,mu30,mu21,mu12,mu03,nu20,nu11,nu02,nu30,nu21,nu12,nu03,perimeter\n")
	writer.close()
	append_writer = open(filename, 'a')
	return append_writer

def readAndQueueFrames(queue, queue2, args):
	reader = imageio.get_reader(args.input_movie, 'ffmpeg')
	im_iter = reader.iter_data()
	stillReading = True
	while True:
		frames = np.zeros([args.batch_size, reader.get_meta_data()['size'][1], reader.get_meta_data()['size'][0], 3], dtype=np.uint8)
		for i in range(args.batch_size):
			try:
				frames[i,:,:,:] = np.uint8(next(im_iter))
			except StopIteration:
				# Do we want to run the frames not divisible by batch size?
				stillReading = False
				break
			except RuntimeError:
				stillReading = False
				break
		if stillReading:
			queue.put(frames)
			queue2.put(frames)
		else:
			break

def inferFrames(input_queue, output_queue, args):
	sess, seg_output, input_placeholder = loadNetwork(args)
	count_waits = 0
	while True:
		try:
			frames = input_queue.get(block=True, timeout=1)
			count_waits = 0
			result_seg = sess.run(fetches=[seg_output], feed_dict={input_placeholder:frames})[0]
			output_queue.put(result_seg)
		except mp.queues.Empty:
			count_waits = count_waits + 1
			if count_waits > 10:
				break

def processMovie(args):
	
	writer_base_name, file_extension = os.path.splitext(args.input_movie)
	framenum = 0
	chunk_num = 1
	small_writer = None
	full_writer = None
	writer_has_changed = True

	start_time = time()
	frame_batch_queue = mp.Queue(args.batch_size*5)
	frame_batch_queue_2 = mp.Queue(args.batch_size*5)
	frame_mp_pool = mp.Process(target=readAndQueueFrames, args=(frame_batch_queue, frame_batch_queue_2, args,), daemon=True)
	frame_mp_pool.start()

	infered_queue = mp.Queue(args.batch_size*5)
	infer_mp_pool = mp.Process(target=inferFrames, args=(frame_batch_queue, infered_queue, args,), daemon=True)
	infer_mp_pool.start()

	# Setup video writer if writing
	if args.export_video:
		video_writer = imageio.get_writer(writer_base_name + '_seg.avi', fps=30, codec='mpeg4', quality=10)

	# Only setup one writer if no fragmentation
	if args.fragment_target <= 0:
		small_writer = writeCSVHeader(writer_base_name + "_SegMask.csv")
		full_writer = writeCSVHeader(writer_base_name + "_DarkMask.csv")

	while True:
		# Writer resets
		if writer_has_changed and (args.fragment_target > 0) and (framenum % args.fragment_target < args.batch_size):
			if small_writer is not None:
				small_writer.close()
			if full_writer is not None:
				full_writer.close()
			small_writer = writeCSVHeader(writer_base_name + "_SegMask_" + str(chunk_num) + ".csv")
			full_writer = writeCSVHeader(writer_base_name + "_DarkMask_" + str(chunk_num) + ".csv")
			chunk_num = chunk_num + 1
			writer_has_changed = False
		# Writing thread
		try:
			result_seg = infered_queue.get(block=True, timeout=1)
			frames = frame_batch_queue_2.get()
			writer_has_changed = True
			if args.fragment_target > 0:
				framenum = framenum + args.batch_size
			for j in np.arange(np.shape(result_seg)[0]):
				out_seg = result_seg[j]
				# Run the opencv functions on the data
				thresh = np.zeros_like(out_seg)
				thresh[out_seg > 0.5] = 1
				#cv2.imshow('Input', frames[j])
				#cv2.imshow('Threshold', thresh)
				contours, hierarchy = cv2.findContours(np.uint8(thresh), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
				if len(contours) < 1:
					# Default values
					moments = {'m00': 0, 'm10': 0, 'm01': 0, 'm20': 0, 'm11': 0, 'm02': 0, 'm30': 0, 'm21': 0, 'm12': 0, 'm03': 0, 'mu20': 0, 'mu11': 0, 'mu02': 0, 'mu30': 0, 'mu21': 0, 'mu12': 0, 'mu03': 0, 'nu20': 0, 'nu11': 0, 'nu02': 0, 'nu30': 0, 'nu21': 0, 'nu12': 0, 'nu03': 0}
					moments2 = moments
					perimeter = 0
					perimeter2 = 0
					np.savetxt(small_writer, [list(moments.values()) + [perimeter]], delimiter=',')
					np.savetxt(full_writer, [list(moments2.values()) + [perimeter2]], delimiter=',')
					if args.export_video:
						video_writer.append_data(np.zeros([1080, 1080, 1], dtype=np.uint8))
				else:
					max_contour = None
					max_size = -1
					for k in contours:
						blob_size = cv2.contourArea(k)
						if blob_size > max_size:
							max_contour = k
							max_size = blob_size
					masked_seg = np.zeros_like(thresh)
					cv2.drawContours(masked_seg, [max_contour], -1, 1, -1)
					moments = cv2.moments(masked_seg)
					perimeter = cv2.arcLength(max_contour, True)
					# Copy the "holes" in the contour
					#masked_seg[masked_seg>0] = thresh[masked_seg>0]
					#cv2.imshow('Contour', masked_seg)
					masked_full_frame = np.zeros((1080, 1080))
					resized_mask = cv2.resize(masked_seg, (1080, 1080))
					masked_full_frame[resized_mask > 0] = 1
					masked_full_frame[cv2.cvtColor(frames[j][:,420:1080+420,:], cv2.COLOR_BGR2GRAY) > 30] = 0
					#cv2.imshow('Larger Mask', masked_full_frame)
					#cv2.waitKey()
					moments2 = cv2.moments(masked_full_frame)
					contours2, hierarchy = cv2.findContours(np.uint8(masked_full_frame), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
					if len(contours2) < 1:
						perimeter2 = 0
					else:
						max_contour2 = None
						max_size = -1
						for k in contours2:
							blob_size = cv2.contourArea(k)
							if blob_size > max_size:
								max_contour2 = k
								max_size = blob_size
						perimeter2 = cv2.arcLength(max_contour2, True)
					np.savetxt(small_writer, [list([moments[x] for x in output_order]) + [perimeter]], delimiter=',')
					np.savetxt(full_writer, [list([moments2[x] for x in output_order]) + [perimeter2]], delimiter=',')
					if args.export_video:
						video_writer.append_data(np.uint8(masked_full_frame*254))
			print("Frames per second: " + str(args.batch_size/(time()-start_time)))
			start_time = time()
		except mp.queues.Empty:
			if frame_batch_queue.empty() and infered_queue.empty() and frame_batch_queue_2.empty():
				# Try and close the threads
				if frame_mp_pool.is_alive():
					frame_mp_pool.join()
				if infer_mp_pool.is_alive():
					infer_mp_pool.join()
			if not frame_mp_pool.is_alive() and not infer_mp_pool.is_alive():
				break
	if small_writer is not None:
		small_writer.close()
	if full_writer is not None:
		full_writer.close()

def loadNetwork(args):
	sys.path.append(args.model_def_path)
	from utils.models import construct_segsoft_v5
	input_placeholder = tf.placeholder(tf.uint8, [args.batch_size, 1080, 1920, 3])
	inputs = tf.image.rgb_to_grayscale(input_placeholder)
	inputs = tf.image.crop_and_resize(inputs, tf.tile([[0,420./1920.,1,(1080.+420.)/1920.]], [args.batch_size,1]), tf.range(args.batch_size), [480,480])
	with tf.variable_scope('Network'):
		seg = construct_segsoft_v5(inputs, False)
	# Apply the morphological filtering
	seg_morph = tf.slice(tf.nn.softmax(seg,-1),[0,0,0,0],[-1,-1,-1,1])-tf.slice(tf.nn.softmax(seg,-1),[0,0,0,1],[-1,-1,-1,1])
	filter1 = tf.expand_dims(tf.constant(morph.iterate_structure(morph.generate_binary_structure(2,1),4),dtype=tf.float32),-1)
	seg_morph = tf.nn.dilation2d(tf.nn.erosion2d(seg_morph,filter1,[1,1,1,1],[1,1,1,1],"SAME"),filter1,[1,1,1,1],[1,1,1,1],"SAME")
	filter2 = tf.expand_dims(tf.constant(morph.iterate_structure(morph.generate_binary_structure(2,1),5),dtype=tf.float32),-1)
	seg_morph = tf.nn.erosion2d(tf.nn.dilation2d(seg_morph,filter2,[1,1,1,1],[1,1,1,1],"SAME"),filter2,[1,1,1,1],[1,1,1,1],"SAME")
	sess = tf.Session()
	saver = tf.train.Saver(slim.get_variables_to_restore())
	sess.run(tf.global_variables_initializer())
	try:
		saver.restore(sess, args.model_file)
	except:
		print('Failed to import model definition')
		exit(0)
	return sess, seg_morph, input_placeholder

def main(argv):
	parser = argparse.ArgumentParser(description='Processes a UPenn video for image moment output.')
	parser.add_argument('--model_def_path', help='Folder where definition of model resides', default='/inference-environment-code/')
	parser.add_argument('--model_file', help='Trained model', default='/inference-environment-model/model.ckpt-250000')
	parser.add_argument('--batch_size', help='Size of batches to compute', default=5, type=int)
	parser.add_argument('--input_movie', help='Name of the video to process', required=True)
	parser.add_argument('--fragment_target', help='Target size to fragment the output files by (-1 to not fragment)', default=100000, type=int)
	parser.add_argument('--export_video', help='Export the segmentation video', default=False, action='store_true')
	#
	args = parser.parse_args()
	processMovie(args)

if __name__  == '__main__':
	main(sys.argv[1:])