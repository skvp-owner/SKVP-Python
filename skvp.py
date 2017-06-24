
class SkvpVideoInvalidInitError(Exception):
	pass

class SkvpVideoInvalidValueError(Exception):
	pass

class SkvpForbiddenOperationError(Exception):
	pass


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
			if len(self.connections) > 0:
				raise SkvpForbiddenOperationError('Cannot modify number of joints while connections list is not empty')
		if hasattr(self, 'joint_radiuses'):
			if len(self.joint_radiuses) > 0:
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

	def get_frame_camera_settings(self, i):
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
			if camera_settings == None and i == 0:
				camera_settings = {}
				if self.camera_location != None:
					camera_settings['camera_location'] = self.camera_location
				if self.camera_destination != None:
					camera_settings['camera_destination'] = self.camera_destination
				if self.camera_scene_rotation != None:
					camera_settings['camera_scene_rotation'] = self.camera_scene_rotation
				if len(camera_settings) > 0:
					new_vid.invideo_camera_settings[i] = camera_settings
			elif camera_settings != None:
				if camera_settings == last_camera_settings:
					continue
				new_vid.invideo_camera_settings[i] = dict(camera_settings)
				last_camera_settings = camera_settings

		return new_vid







