
class SkvpVideo:
	def __init__(self, input_file_handler = None, fps = None, num_joints = None, 
			connections = None, joint_radiuses = None,
			connections_radius = None, camera_location = None):

		got_parameters = fps != None or num_joints != None or connections != None or joint_radiuses != None or connections_radius != None or camera_location != None

		if input_file_handler != None:
			if got_parameters:
				return
				# Throw exception - can't specify both
			self.parse_skvp_file(input_file_handler)
			return

		self.set_fps(fps)
		self.set_num_joints(num_joints)
		self.set_connections(connections)
		self.set_joint_radiuses(joint_radiuses)
		self.set_connections_radius(connections_radius)
		self.set_camera_location(camera_location)
		self.frames = []
		self.invideo_camera_locations = []

	def set_fps(self, fps):
		self.validate_fps(fps)
		self.fps = fps
	
	def get_fps(self):
		return self.fps

	def set_num_joints(self, num_joints):
		if len(self.frames) > 0:
			return
			# Fix - throw exception!
		self.validate_num_joints(num_joints)
		self.num_joints = num_joints

	def get_num_joints(self):
		return self.num_joints

	def set_connections(self, connections):
		self.validate_connections(connections)
		self.connections = tuple(connections)

	def get_connections(self):
		return self.connections

	def set_joint_radiuses(self, joint_radiuses):
		if len(self.frames) > 0:
			return
			# Fix - throw exception!
		self.validate_joint_radiuses(joint_radiuses)
		self.joint_radiuses = tuple(joint_radiuses)

	def get_joint_radiuses(self):
		return self.joint_radiuses

	def set_connections_radius(self, connections_radius):
		self.validate_connections_radius(connections_radius)
		self.connections_radius = connections_radius

	def get_connections_radius(self):
		return self.connections_radius

	def set_camera_location(self, camera_location):
		self.validate_camera_location(camera_location)
		self.camera_location = tuple(camera_location)
	
	def get_camera_location(self):
		return self.camera_location

	



