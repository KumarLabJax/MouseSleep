import imageio
import cv2
import numpy as np
import os, sys
import argparse

output_order = ['m00','m10','m01','m20','m11','m02','m30','m21','m12','m03','mu20','mu11','mu02','mu30','mu21','mu12','mu03','nu20','nu11','nu02','nu30','nu21','nu12','nu03']

def writeCSVHeader(filename):
	writer = open(filename, 'w')
	writer.write("m00,m10,m01,m20,m11,m02,m30,m21,m12,m03,mu20,mu11,mu02,mu30,mu21,mu12,mu03,nu20,nu11,nu02,nu30,nu21,nu12,nu03,perimeter\n")
	writer.close()
	append_writer = open(filename, 'a')
	return append_writer

def process_video(args):
	vid_reader = imageio.get_reader(args.input_file)
	full_writer = writeCSVHeader(os.path.splitext(args.input_file)[0] + '_DarkMask_' + str(args.frame_size) + '.csv')
	for frame in vid_reader:
		frame = frame[:,:,0]
		frame = cv2.resize(frame, (args.frame_size, args.frame_size))
		masked_full_frame = np.zeros_like(frame)
		masked_full_frame[frame > 128] = 1
		moments = cv2.moments(masked_full_frame)
		contours, hierarchy = cv2.findContours(np.uint8(masked_full_frame), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		if len(contours) < 1:
			# Default values
			moments = {'m00': 0, 'm10': 0, 'm01': 0, 'm20': 0, 'm11': 0, 'm02': 0, 'm30': 0, 'm21': 0, 'm12': 0, 'm03': 0, 'mu20': 0, 'mu11': 0, 'mu02': 0, 'mu30': 0, 'mu21': 0, 'mu12': 0, 'mu03': 0, 'nu20': 0, 'nu11': 0, 'nu02': 0, 'nu30': 0, 'nu21': 0, 'nu12': 0, 'nu03': 0}
			perimeter = 0
		else:
			max_contour = None
			max_size = -1
			for k in contours:
				blob_size = cv2.contourArea(k)
				if blob_size > max_size:
					max_contour = k
					max_size = blob_size
			perimeter = cv2.arcLength(max_contour, True)
		np.savetxt(full_writer, [list([moments[x] for x in output_order]) + [perimeter]], delimiter=',')

	vid_reader.close()
	full_writer.close()
	# cv2.imwrite('masked.png',masked_full_frame*254)
	# cv2.imwrite('frame.png',frame)

def main(argv):
	parser = argparse.ArgumentParser(description='Exports ')
	parser.add_argument('--input_file', help='Input dataset to process', required=True)
	parser.add_argument('--frame_size', help='Scaled frame size to use', default=1080, type=int)
	args = parser.parse_args()
	process_video(args)

if __name__ == '__main__':
	main(sys.argv[1:])
