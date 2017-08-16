import sys
import io
import numpy as np
import scipy
import math

class SkvpVideoInvalidInitError(Exception):
	pass

class SkvpVideoInvalidValueError(Exception):
	pass

class SkvpForbiddenOperationError(Exception):
	pass

class SkvpFileSyntaxError(Exception):
	pass

class SkvpUsageError(Exception):
	pass


SKVP_HEADER_TITLE_LINE = '*SKeleton Video Player File Header*'
SKVP_VIDEO_TITLE_LINE = '*Video*'
SKVP_HEADER_NUM_JOINTS_ENTRY = 'NumJoints'
SKVP_HEADER_NUM_FRAMES_ENTRY = 'NumberOfFrames'
SKVP_HEADER_FPS_ENTRY = 'FPS'
SKVP_HEADER_CONNECTIONS_ENTRY = 'Connections'
SKVP_HEADER_JOINT_RADIUSES_ENTRY = 'JointRadiuses'
SKVP_HEADER_CONNECTIONS_RADIUS_ENTRY = 'ConnectionsRadius'
SKVP_HEADER_CAMERA_LOCATION_ENTRY = 'CameraLocation'
SKVP_HEADER_CAMERA_DESTINATION_ENTRY = 'CameraDestination'
SKVP_HEADER_CAMERA_SCENE_ROTATION_ENTRY = 'CameraSceneRotation'


class SkvpVideo:
	def __init__(self, input_file_handler = None, fps = None, num_joints = None, 
			connections = None, joint_radiuses = None,
			connections_radius = None, camera_location = None,
			camera_destination = None, camera_scene_rotation = None):

		got_parameters = fps != None or num_joints != None or connections != None or joint_radiuses != None or connections_radius != None or camera_location != None or camera_destination != None or camera_scene_rotation != None

		if input_file_handler != None:
			if got_parameters:
				raise SkvpVideoInvalidInitError('Cannot specify both input file header and Skvp parameters')
			self.parse_skvp_file(input_file_handler)
			return

		self.set_fps(fps)
		self.set_num_joints(num_joints)
		self.set_connections(connections)
		self.set_joint_radiuses(joint_radiuses)
		self.set_connections_radius(connections_radius)
		self.set_camera_location(camera_location)
		self.set_camera_destination(camera_destination)
		self.set_camera_scene_rotation(camera_scene_rotation)
		self.frames = []
		self.invideo_camera_settings = {}

	def validate_fps(self, fps):
		if fps == None:
			return
		if not (type(fps) is int or type(fps) is float):
			raise SkvpVideoInvalidValueError('\'fps\' must be a positive real number')
		if fps <= 0:
			raise SkvpVideoInvalidValueError('\'fps\' must be a positive real number')

	def set_fps(self, fps):
		self.validate_fps(fps)
		self.fps = fps if fps != None else None
	
	def get_fps(self):
		return self.fps

	def validate_num_joints(self, num_joints):
		if hasattr(self, 'frames'):
			if len(self.frames) > 0:
				raise SkvpForbiddenOperationError('Cannot modify number of joints while frames list is not empty')
		if hasattr(self, 'connections'):
			if self.connections != None and len(self.connections) > 0:
				raise SkvpForbiddenOperationError('Cannot modify number of joints while connections list is not empty')
		if hasattr(self, 'joint_radiuses'):
			if self.joint_radiuses != None and len(self.joint_radiuses) > 0:
				raise SkvpForbiddenOperationError('Cannot modify number of joints while joint radiuses are defined')
		if num_joints == None:
			return
		if not (type(num_joints) is int):
			raise SkvpVideoInvalidValueError('\'num_joints\' must be a positive integer')
		if num_joints <= 0:
			raise SkvpVideoInvalidValueError('\'num_joints\' must be a positive integer')

	def set_num_joints(self, num_joints):
		self.validate_num_joints(num_joints)
		self.num_joints = num_joints if num_joints != None else None

	def get_num_joints(self):
		return self.num_joints

	def validate_connections(self, connections):
		if connections == None:
			return
		if len(connections) == 0:
			return
		if not hasattr(self, 'num_joints'):
			raise SkvpForbiddenOperationError('Cannot define connections while number of joints is not defined')
		if self.num_joints == None:
			raise SkvpForbiddenOperationError('Cannot define connections while number of joints is not defined')
		for (j1, j2) in connections:
			if not (type(j1) is int and type(j2) is int):
				raise SkvpVideoInvalidValueError('Joint numbers must be integers between 1 and NUM_JOINTS')
			if j1 < 1 or j1 > self.num_joints or j2 < 1 or j2 > self.num_joints:
				raise SkvpVideoInvalidValueError('Joint numbers must be integers between 1 and NUM_JOINTS')
			if j1 == j2:
				raise SkvpVideoInvalidValueError('Cannot define a connection from a joint into itself')


	def set_connections(self, connections):
		self.validate_connections(connections)
		self.connections = tuple(connections) if connections != None else None

	def get_connections(self):
		return self.connections

	def validate_joint_radiuses(self, joint_radiuses):
		if joint_radiuses == None:
			return
		if len(joint_radiuses) == 0:
			return
		if not hasattr(self, 'num_joints'):
			raise SkvpForbiddenOperationError('Cannot define joint radiuses while number of joints is not defined')
		if self.num_joints == None:
			raise SkvpForbiddenOperationError('Cannot define joint radiuses while number of joints is not defined')
		if len(joint_radiuses) != self.num_joints:
			raise SkvpVideoInvalidValueError('Joint radiuses array must contain exactly J values, where J is number of joints')
		for val in joint_radiuses:
			if not (type(val) is int or type(val) is float):
				raise SkvpVideoInvalidValueError('Joint radiuses must be non-negative real numbers')
			if val < 0:
				raise SkvpVideoInvalidValueError('Joint radiuses must be non-negative real numbers')


	def set_joint_radiuses(self, joint_radiuses):
		self.validate_joint_radiuses(joint_radiuses)
		self.joint_radiuses = tuple(joint_radiuses) if joint_radiuses != None else None

	def get_joint_radiuses(self):
		return self.joint_radiuses

	def validate_connections_radius(self, connections_radius):
		if connections_radius == None:
			return
		if not (type(connections_radius) is int or type(connections_radius) is float):
			raise SkvpVideoInvalidValueError('Connections radius must be a single non_negative real number')
		if connections_radius < 0:
			raise SkvpVideoInvalidValueError('Connections radius must be a single non_negative real number')


	def set_connections_radius(self, connections_radius):
		self.validate_connections_radius(connections_radius)
		self.connections_radius = connections_radius if connections_radius != None else None

	def get_connections_radius(self):
		return self.connections_radius

	def validate_camera_coordinate(self, camera_coordinate, coordinate_type):
		if camera_coordinate == None:
			return
		if len(camera_coordinate) != 3:
				raise SkvpVideoInvalidValueError('Camera {0} should be an array containing 3 numbers (X,Y,Z)'.format(coordinate_type))
		for val in camera_coordinate:
			val_type = type(val)
			if not (val_type is int or val_type is float):
				raise SkvpVideoInvalidValueError('Camera (X,Y,Z) coordinates must be real numbers')

	def set_camera_location(self, camera_location):
		self.validate_camera_coordinate(camera_location, 'location')
		self.camera_location = tuple(camera_location) if camera_location != None else None
	
	def get_camera_location(self):
		return self.camera_location

	def set_camera_destination(self, camera_destination):
		self.validate_camera_coordinate(camera_destination, 'destination')
		self.camera_destination = tuple(camera_destination) if camera_destination != None else None
	
	def get_camera_destination(self):
		return self.camera_destination
	
	def validate_camera_scene_rotation(self, camera_scene_rotation):
		if camera_scene_rotation == None:
			return
		val_type = type(camera_scene_rotation)
		if not (val_type is int or val_type is float):
			raise SkvpVideoInvalidValueError('Camera scene rotation must be a real number')

	def set_camera_scene_rotation(self, camera_scene_rotation):
		self.validate_camera_scene_rotation(camera_scene_rotation)
		self.camera_scene_rotation = camera_scene_rotation if camera_scene_rotation != None else None

	def get_camera_scene_rotation(self):
		return self.camera_scene_rotation

	def add_frame(self, joint_locations_iterable):
		if not hasattr(self, 'num_joints'):
			raise SkvpForbiddenOperationError('Cannot add frames while number of joints is not defined')
		if self.num_joints == None:
			raise SkvpForbiddenOperationError('Cannot add frames while number of joints is not defined')
		frame = [np.array(joint_loc) for joint_loc in joint_locations_iterable]
		if len(frame) != self.num_joints:
			raise SkvpVideoInvalidValueError('Frame joint locations must contain exactly J locations, where J is number of joints')
		self.frames.append(frame)

	def set_frame_camera_settings(self, i, camera_settings):
		if i < 0 or i >= len(self.num_frames):
			raise SkvpForbiddenOperationError('Index out of bounds')
		self.invideo_camera_settings[i] = camera_settings

	def get_frame_camera_settings(self, i):
		# This method returns the **LATEST**, *invideo* camera settings for frame i (can be defined in less than i)
		if i < 0 or i >= len(self.frames):
			raise SkvpForbiddenOperationError('frame index is out of video bounds: ' + str(i))
		# We want to find largest index that is smaller or equals to i
		if len(self.invideo_camera_settings) == 0:
			return None
		if i in self.invideo_camera_settings:
			return self.invideo_camera_settings[i]
		relevant_frames = sorted([frame_num for frame_num in self.invideo_camera_settings.keys() if frame_num <= i])
		if len(relevant_frames) == 0:
			return None
		return self.invideo_camera_settings[relevant_frames[-1]]
	
	def get_video_length(self, seconds = False):
		if seconds:
			if self.fps == None:
				raise SkvpForbiddenOperationError('Cannot calculate length in seconds while FPS is not defined')
			return len(self.frames) / float(self.fps)
		
		return len(self)

	def __len__(self):
		return len(self.frames)

	def __getitem__(self, val):
		if type(val) is int:
			start = len(self) + val if val < 0 else val
			end = start + 1
			step = 1
		elif type(val) is slice:
			start = val.start
			end = val.stop
			step = val.step
		if start == None:
			start = 0
		if end == None:
			end = len(self)
		if step == None:
			step = 1
		if not (type(start) is int and type(end) is int and type(step) is int):
			raise SkvpForbiddenOperationError('Slicing parameters must be integers')
		new_vid = SkvpVideo(fps = self.fps, num_joints = self.num_joints, connections = self.connections, joint_radiuses = self.joint_radiuses, connections_radius = self.connections_radius, camera_location = self.camera_location, camera_destination = self.camera_destination, camera_scene_rotation = self.camera_scene_rotation)
		new_vid.frames = self.frames[start:end:step]
		start_pos = (len(self) + start) if start < 0 else start
		end_pos = (len(self) + end) if end < 0 else end
		last_camera_settings = None
		for i, frame_key in enumerate(range(start_pos, end_pos, step)):
			camera_settings = self.get_frame_camera_settings(frame_key)
			if camera_settings != None:
				if camera_settings == last_camera_settings:
					continue
				new_vid.invideo_camera_settings[i] = dict(camera_settings)
				last_camera_settings = camera_settings

		return new_vid

	def __add__(self, vid_2):
		if type(vid_2) is not SkvpVideo:
			raise SkvpForbiddenOperationError('Cannot add a non SkvpVideo object')
		if self.num_joints != vid_2.num_joints:
			raise SkvpForbiddenOperationError('Cannot add two videos with different number of joints')
		if self.fps != vid_2.fps:
			sys.stderr.write('Warning: adding videos with mismatching frame rates. Frames will be copied without modification\n')
		new_vid = SkvpVideo(fps = self.fps, num_joints = self.num_joints, connections = self.connections, joint_radiuses = self.joint_radiuses, connections_radius = self.connections_radius, camera_location = self.camera_location, camera_destination = self.camera_destination, camera_scene_rotation = self.camera_scene_rotation)
		new_vid.frames = self.frames + vid_2.frames
		for frame_index, camera_settings in self.invideo_camera_settings.items():
			new_vid.invideo_camera_settings[frame_index] = dict(camera_settings)
		num_frames_self = len(self)
		for frame_index, camera_settings in vid_2.invideo_camera_settings:
			new_vid.invideo_camera_settings[frame_index + num_frames_self] = dict(camera_settings)
		if num_frames_self not in new_vid.invideo_camera_settings:
			camera_settings = {}
			if vid_2.get_camera_location() != None:
				camera_settings['camera_location'] = vid_2.get_camera_location()
			if vid_2.get_camera_destination() != None:
				camera_settings['camera_destination'] = vid_2.get_camera_destination()
			if vid_2.get_camera_scene_rotation() != None:
				camera_settings['camera_scene_rotation'] = vid_2.get_camera_scene_rotation()
			if len(camera_settings) > 0:
				new_vid.invideo_camera_settings[num_frames_self] = camera_settings

		return new_vid
			
def dump(skvp_video, ostream):
	self = skvp_video 
	if self.fps == None:
		raise SkvpForbiddenOperationError('Cannot dump SkvpVideo while FPS is not defined')
	if self.num_joints == None:
		raise SkvpForbiddenOperationError('Cannot dump SkvpVideo while number of joints is not defined')
	ostream.write(SKVP_HEADER_TITLE_LINE)
	ostream.write('\n')
	ostream.write('{0}={1}'.format(SKVP_HEADER_NUM_JOINTS_ENTRY, str(self.num_joints)))
	ostream.write('\n')
	ostream.write('{0}={1}'.format(SKVP_HEADER_NUM_FRAMES_ENTRY, str(len(self.frames))))
	ostream.write('\n')
	ostream.write('{0}={1}'.format(SKVP_HEADER_FPS_ENTRY, str(self.fps)))
	ostream.write('\n')
	if self.connections != None:
		ostream.write('{0}={1}'.format(SKVP_HEADER_CONNECTIONS_ENTRY, ','.join(['{0}-{1}'.format(str(c[0]), str(c[1])) for c in self.connections])))
		ostream.write('\n')
	if self.joint_radiuses != None:
		ostream.write('{0}={1}'.format(SKVP_HEADER_JOINT_RADIUSES_ENTRY, ','.join([str(val) for val in self.joint_radiuses])))
		ostream.write('\n')
	if self.connections_radius != None:
		ostream.write('{0}={1}'.format(SKVP_HEADER_CONNECTIONS_RADIUS_ENTRY, str(self.connections_radius)))
		ostream.write('\n')
	if self.camera_location != None:
		ostream.write('{0}={1}'.format(SKVP_HEADER_CAMERA_LOCATION_ENTRY, ','.join([str(val) for val in self.camera_location])))
		ostream.write('\n')
	if self.camera_destination != None:
		ostream.write('{0}={1}'.format(SKVP_HEADER_CAMERA_DESTINATION_ENTRY, ','.join([str(val) for val in self.camera_destination])))
		ostream.write('\n')
	if self.camera_scene_rotation != None:
		ostream.write('{0}={1}'.format(SKVP_HEADER_CAMERA_SCENE_ROTATION_ENTRY, str(self.camera_scene_rotation)))
		ostream.write('\n')
	ostream.write('\n')
	ostream.write(SKVP_VIDEO_TITLE_LINE)
	ostream.write('\n')
	for i, frame in enumerate(self.frames):
		if i in self.invideo_camera_settings:
			if 'camera_location' in self.invideo_camera_settings[i]:
				ostream.write('{0}={1}'.format(SKVP_HEADER_CAMERA_LOCATION_ENTRY, ','.join([str(val) for val in self.invideo_camera_settings[i]['camera_location']])))
				ostream.write('\n')
			if 'camera_destination' in self.invideo_camera_settings[i]:
				ostream.write('{0}={1}'.format(SKVP_HEADER_CAMERA_DESTINATION_ENTRY, ','.join([str(val) for val in self.invideo_camera_settings[i]['camera_destination']])))
				ostream.write('\n')
			if 'camera_scene_rotation' in self.invideo_camera_settings[i]:
				ostream.write('{0}={1}'.format(SKVP_HEADER_CAMERA_SCENE_ROTATION_ENTRY, str(self.invideo_camera_settings[i]['camera_scene_rotation'])))
				ostream.write('\n')
		ostream.write(';'.join([','.join([str(coor) for coor in joint_xyz]) for joint_xyz in frame]))
		if i < len(self.frames) - 1:
			ostream.write('\n')
		

def dumps(skvp_video):
	string_stream = io.StringIO()
	dump(skvp_video, string_stream)

	return string_stream.getvalue()

def find_skvp_header_beginning(istream):
	while True:
		line = istream.readline()
		if line == '':    # Empty line within file must contain '\n' char
			raise SkvpFileSyntaxError('SKVP file is empty')
		line = line.strip()
		if line == '':
			continue
		if line == SKVP_HEADER_TITLE_LINE:
			return
		raise SkvpFileSyntaxError('SKVP file must start with line: ' + SKVP_HEADER_TITLE_LINE)

def find_skvp_video_start_and_get_header(istream):
	header = {}
	while True:
		line = istream.readline()
		if line == '':    # Empty line within file must contain '\n' char
			raise SkvpFileSyntaxError('Could not find line which indicates starting of video: ' + SKVP_VIDEO_TITLE_LINE)
		line = line.strip()
		if line == '':
			continue
		if line == SKVP_VIDEO_TITLE_LINE:
			return header
		parts = line.split('=', 1)
		if len(parts) != 2:
			raise SkvpFileSyntaxError('Header lines must be in the format: ENTRY_NAME=ENTRY_VALUE')
		header[parts[0].strip()] = parts[1].strip()

def header_to_video_object(header, skvp_video):
	if SKVP_HEADER_NUM_JOINTS_ENTRY not in header:
		raise SkvpFileSyntaxError('Header does not contain entry: ' + SKVP_HEADER_NUM_JOINTS_ENTRY)
	try:
		num_joints = int(header[SKVP_HEADER_NUM_JOINTS_ENTRY])
	except:
		raise SkvpFileSyntaxError(SKVP_HEADER_NUM_JOINTS_ENTRY + ' must be a natural number')
	skvp_video.set_num_joints(num_joints)
	if SKVP_HEADER_FPS_ENTRY not in header:
		raise SkvpFileSyntaxError('Header does not contain entry: ' + SKVP_HEADER_FPS_ENTRY)
	try:
		fps = float(header[SKVP_HEADER_FPS_ENTRY])
	except:
		raise SkvpFileSyntaxError(SKVP_HEADER_FPS_ENTRY + ' must be a real number')
	skvp_video.set_fps(fps)
	if SKVP_HEADER_CONNECTIONS_ENTRY in header:
		try:
			connections = [(int(p.split('-')[0]), int(p.split('-')[1])) for p in header[SKVP_HEADER_CONNECTIONS_ENTRY].split(',')]
		except:
			raise SkvpFileSyntaxError(SKVP_HEADER_CONNECTIONS_ENTRY + ' must be pairs of natural numbers separated by commas, where each pair is separated by a hyphen')
		skvp_video.set_connections(connections)
	if SKVP_HEADER_JOINT_RADIUSES_ENTRY in header:
		try:
			radiuses = [float(rad) for rad in header[SKVP_HEADER_JOINT_RADIUSES_ENTRY].split(',')]
		except:
			raise SkvpFileSyntaxError(SKVP_HEADER_JOINT_RADIUSES_ENTRY + ' must be a list of real numbers, separated by commas')
		skvp_video.set_joint_radiuses(radiuses)
	if SKVP_HEADER_CONNECTIONS_RADIUS_ENTRY in header:
		try:
			radius = float(header[SKVP_HEADER_CONNECTIONS_RADIUS_ENTRY])
		except:
			raise SkvpFileSyntaxError(SKVP_HEADER_CONNECTIONS_RADIUS_ENTRY + ' must be a real number')
		skvp_video.set_connections_radius(radius)
	if SKVP_HEADER_CAMERA_LOCATION_ENTRY in header:
		try:
			location = [float(val) for val in header[SKVP_HEADER_CAMERA_LOCATION_ENTRY].split(',')]
		except:
			raise SkvpFileSyntaxError(SKVP_HEADER_CAMERA_LOCATION_ENTRY + ' must be a list of real numbers, separated by commas')
		skvp_video.set_camera_location(location)
	if SKVP_HEADER_CAMERA_DESTINATION_ENTRY in header:
		try:
			destination = [float(val) for val in header[SKVP_HEADER_CAMERA_DESTINATION_ENTRY].split(',')]
		except:
			raise SkvpFileSyntaxError(SKVP_HEADER_CAMERA_DESTINATION_ENTRY + ' must be a list of real numbers, separated by commas')
		skvp_video.set_camera_destination(destination)
	if SKVP_HEADER_CAMERA_SCENE_ROTATION_ENTRY in header:
		try:
			rotation = float(header[SKVP_HEADER_CAMERA_SCENE_ROTATION_ENTRY])
		except:
			raise SkvpFileSyntaxError(SKVP_HEADER_CAMERA_SCENE_ROTATION_ENTRY + ' must be a real number')
		skvp_video.set_camera_scene_rotation(rotation)

def get_invideo_camera_parameter(line):
	parts = line.split('=', 1)
	if len(parts) != 2:
		raise SkvpFileSyntaxError('Invalid invideo camera settings line: ' + line)
	return parts[1].strip()

def read_frames_into_video_object(istream, skvp_video):
	num_frames_read = 0
	camera_location = None
	camera_destination = None
	camera_scene_rotation = None
	while True:
		line = istream.readline()
		if line == '':    # Empty line within file must contain '\n' char
			break
		line = line.strip()
		if line == '':
			continue
		if line.startswith(SKVP_HEADER_CAMERA_LOCATION_ENTRY):
			param_str = get_invideo_camera_parameter(line)
			continue
		elif line.startswith(SKVP_HEADER_CAMERA_DESTINATION_ENTRY):
			param_str = get_invideo_camera_parameter(line)
			continue
		elif line.startswith(SKVP_HEADER_CAMERA_SCENE_ROTATION_ENTRY):
			param_str = get_invideo_camera_parameter(line)
			continue
		joint_locations_as_strings = line.split(';')
		joint_locations_as_np_arrays = [np.array([float(coor) for coor in joint_loc_str.split(',')]) for joint_loc_str in joint_locations_as_strings]
		skvp_video.add_frame(joint_locations_as_np_arrays)
		camera_settings = {}
		if camera_location != None:
			camera_settings['camera_location'] = camera_location
		if camera_destination != None:
			camera_settings['camera_destination'] = camera_destination
		if camera_scene_rotation != None:
			camera_settings['camera_scene_rotation'] = camera_scene_roatation
		if len(camera_settings) > 0:
			skvp_video.set_frame_camera_settings(num_frames_read, camera_settings)
		camera_location = None
		camera_destination = None
		camera_scene_rotation = None

		num_frames_read += 1


def load(istream):
	find_skvp_header_beginning(istream)
	header = find_skvp_video_start_and_get_header(istream)
	skvp_video = SkvpVideo()
	header_to_video_object(header, skvp_video)
	read_frames_into_video_object(istream, skvp_video)

	return skvp_video

def loads(video_string):
	string_stream = io.StringIO(video_string)
	return load(string_stream)


def create_length_scaled_video(ref_video, scale = None, num_frames = None, num_seconds = None):
	if scale == None and num_frames == None and num_seconds == None:
		raise SkvpUsageError('Must specify one of parameters {scale, num_frames, num_seconds}')
	if (scale != None and num_frames != None) or (scale != None and num_seconds != None) or (num_frames != None and num_seconds != None):
		raise SkvpUsageError('Must specify only one of parameters {scale, num_frames, num_seconds}')
	if scale != None:
		if not (type(scale) is int or type(scale) is float):
			raise SkvpUsageError('scale parameter must be a number')
		if scale <= 0:
			raise SkvpForbiddenOperationError('Scale must be a positive real number')
	if num_frames != None:
		if not type(num_frames) is int:
			raise SkvpUsageError('num_frames parameter must be an ineteger')
		if num_frames <= 0:
			raise SkvpForbiddenOperationError('Number of target frames must be a positive number')
	if num_seconds != None:
		if not (type(num_seconds) is int or type(num_seconds) is float):
			raise SkvpUsageError('num_seconds parameter must be a number')
		if num_seconds <= 0:
			raise SkvpForbiddenOperationError('Number of target seconds must be a positive number')
	if len(ref_video) == 0:
		raise SkvpForbiddenOperationError('Cannot scale empty video')
	final_scale = scale
	if num_frames != None:
		final_scale = num_frames / float(len(ref_video))
	elif num_seconds != None:
		final_scale = num_seconds / float(ref_video.get_video_length(seconds = True))
	if final_scale == 1.0:
		return ref_video[:]
	new_vid = ref_video[0:0]     # Creating empty video with same header
	num_frames_target_vid = int(round(len(ref_video) * final_scale))
	num_frames_ref_vid = len(ref_video)
	for i in range(num_frames_target_vid):
		if num_frames_target_vid == 1:
			frame_perc = 0
		else:
			frame_perc = (i) / float(num_frames_target_vid - 1)
		ref_frame = (num_frames_ref_vid - 1) * frame_perc
		upper_weight = ref_frame - int(ref_frame)
		lower_weight = 1.0 - upper_weight
		lower_index = int(ref_frame)
		if lower_weight == 1 or upper_weight == 1:
			new_vid.add_frame(ref_video.frames[lower_index])
		else:
			new_frame = []
			for j in range(ref_video.get_num_joints()):
				new_frame.append(lower_weight * ref_video.frames[lower_index][j] + upper_weight * ref_video.frames[lower_index + 1][j])
			new_vid.add_frame(new_frame)

	return new_vid

def create_fps_changed_video(ref_video, new_fps):
	if ref_video.fps == None:
		raise SkvpForbiddenOperationError('Cannot create fps-changed video while reference video has no fps defined')
	scale = float(new_fps) / ref_video.fps

	new_vid = create_length_scaled_video(ref_video, scale = scale)
	new_vid.set_fps(new_fps)

	return new_vid

def gradient(ref_video):
	new_vid = ref_video[0:0]
	for i in range(len(ref_video) - 1):
		next_frame = ref_video.frames[i + 1]
		curr_frame = ref_video.frames[i]
		der_frame = []
		for next_joint, curr_joint in zip(next_frame, curr_frame):
			der_frame.append(next_joint - curr_joint)
		new_vid.add_frame(der_frame)

	return new_vid

def norm(ref_video):
	new_vid = ref_video[0:0]
	for i in range(len(ref_video)):
		frame = []
		for joint in ref_video.frames[i]:
			vec_magn = np.linalg.norm(joint)
			if vec_magn == 0:
				vec_magn = 1
			frame.append(joint / float(vec_magn))
		new_vid.add_frame(frame)

	return new_vid

def conv(ref_video, mask):
	if len(mask) % 2 != 1 or len(mask) < 0:
		raise Exception('Mask size must be positive and odd')
	if len(mask) > len(ref_video):
		raise Exception('Mask size must not be higher than video length')

	x_per_joint = []
	y_per_joint = []
	z_per_joint = []
	x_per_joint_filtered = []
	y_per_joint_filtered = []
	z_per_joint_filtered = []


	new_vid = ref_video[0:0]
	for frame in ref_video.frames:
		for i, joint in enumerate(frame):
			if i >= len(x_per_joint):
				x_per_joint.append([])
				y_per_joint.append([])
				z_per_joint.append([])
			x_per_joint[i].append(joint[0])
			y_per_joint[i].append(joint[1])
			z_per_joint[i].append(joint[2])
	for joint_num in range(ref_video.get_num_joints()):
		x_per_joint_filtered.append(np.convolve(x_per_joint[joint_num], mask, 'valid'))
		y_per_joint_filtered.append(np.convolve(y_per_joint[joint_num], mask, 'valid'))
		z_per_joint_filtered.append(np.convolve(z_per_joint[joint_num], mask, 'valid'))
	print(y_per_joint[0][:5])
	print(y_per_joint_filtered[0][:5])
	new_num_frames = len(x_per_joint_filtered[0])
	for i in range(new_num_frames):
		frame = []
		for joint_num in range(ref_video.get_num_joints()):
			joint_at_frame = np.array((x_per_joint_filtered[joint_num][i], y_per_joint_filtered[joint_num][i], z_per_joint_filtered[joint_num][i]))
			frame.append(joint_at_frame)
		new_vid.add_frame(frame)

	return new_vid

def get_gaussian_value_at(std, mean, x):
	a = 1 / (2 * math.pi * (std ** 2)) ** 0.5
	b = ((-1) * (x - mean) ** 2) / (2 * (std ** 2))

	return a * (math.e ** b)

def generate_gaussian_mask(std, length):
	if length % 2 == 0:
		raise Exception('Mask size must be odd')
	mask_center_idx = int(length / 2) # Counting indices from 0 :)
	mask = []
	for i in range(length):
		at_std = abs(mask_center_idx - i)
		mask.append(get_gaussian_value_at(std, 0, at_std))
	#current_norm = np.linalg.norm(mask)
	#factor = 1.0 / current_norm
	mask_sum = np.sum(mask)
	factor = 1.0 / mask_sum
	
	return [factor * val for val in mask]

def gaussian_filter(ref_vid, mask_size, std):
	mask = generate_gaussian_mask(std, mask_size)

	return conv(ref_vid, mask)

def to_vector(ref_video):
	num_vals = len(ref_video) * ref_video.get_num_joints() * 3
	vec = np.zeros(num_vals)
	offset = 0
	for frame in ref_video.frames:
		for joint in frame:
			vec[offset:offset + 3] = joint
			offset += 3

	return vec


## TODO: Implement methods for
# 1. Median Filter
# 2. Gaussian Filter    V
# 2a. conv              V
# 3. Skeleton edges scaling
# 4. Skeleton projection to normalized coordinate system









