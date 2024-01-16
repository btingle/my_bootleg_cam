from math import *

def rotate_x(p, theta):
	return (
		p[0],
		cos(theta) * p[1] - sin(theta) * p[2],
		sin(theta) * p[1] + cos(theta) * p[2]
	)

def rotate_y(p, theta):
	return (
		cos(theta) * p[0] + sin(theta) * p[2],
		p[1],
	   -sin(theta) * p[0] + cos(theta) * p[2]
	)

def rotate_z(p, theta):
	return (
		cos(theta) * p[0] - sin(theta) * p[1],
		sin(theta) * p[0] + cos(theta) * p[1],
		p[2]
	)

def cross(a, b):
	return a[1]*b[2]-a[2]*b[1], a[0]*b[2]-a[2]*b[0], a[0]*b[1]-a[1]*b[0]

def normalize(v):
	mag = sqrt(v[0]**2+v[1]**2+v[2]**2)
	return v[0]/mag, v[1]/mag, v[2]/mag

def addvec(a, b):
	return tuple(ai+bi for ai, bi in zip(a, b))

def subvec(a, b):
	return tuple(ai-bi for ai, bi in zip(a, b))

def mulvec(a, s):
	return tuple(ai*s for ai in a)

# turn an open 2D curve into a 3D surface that is linearly offset and twisted around the specified axis
# returns an array of curves, one curve for each input vertex (minus the first and last vertex- explained below)
# the first and last vertex of the input array should reflect continuations of the input curve between those vertices
# points : array of 2D x, y positions
# offset : amount of shift in the x position over all iterations
# theta : total angle to screw over in the x axis
# iterations : number of vertices to generate per point
def screw(points, offset, theta, iterations=64, axis='X'):

	d_theta_x = theta / iterations
	d_x = offset / iterations
	screw_points = []
	screw_norms = []

	# redoing normals- do it right
	# rules: 
	# triangle surface for normal calculation:
	# pv -> v -> nv
	# pv- previous vertex in curve
	# v- current vertex in curve
	# nv- current vertex in next curve
	# 
	# operation assumes that points along profile curve represent adjacent points in the overall curve
	# order of points matters, and influences which direction the normals will face
	# how to get it right? try it once, look at it, then add "reversed()" to the profile array if they are wrong

	# step 1- transform vertices, the easy bit

	for point in points:

		line_points = []
		if axis == 'X':
			def func(p, theta, dx):
				p = rotate_x(p, theta)
				p = (p[0]+dx, p[1], p[2])
				return p
			f = func
		elif axis == 'Y':
			def func(p, theta, dx):
				p = rotate_y(p, theta)
				p = (p[0], p[1]+dx, p[2])
				return p
			f = func

		p = (point[0], point[1], 0)
		screw_points.append([f(p, d_theta_x*i, d_x*i) for i in range(-1, iterations+2)])

	# step 2- calculate normals on each triangle surface defined by the rules listed above

	for i in range(0, len(points)-1):
		line_norms = []
		for j in range(1, iterations+3):
			cv = screw_points[i][j]
			pv = screw_points[i][j-1]
			nv = screw_points[i+1][j]

			l1 = (cv[0]-pv[0], cv[1]-pv[1], cv[2]-pv[2])
			l2 = (nv[0]-pv[0], nv[1]-pv[1], nv[2]-pv[2])

			n = normalize(cross(l1, l2))
			line_norms.append(n)
		screw_norms.append(line_norms)

	# step 3- average normals for each vertex to be in the final output

	screw_points_new = []
	screw_norms_new = []
	for i in range(1, len(points)-1):
		line_points = []
		line_norms = []
		for j in range(1, iterations+2):
			line_points.append(screw_points[i][j])

			n1 = screw_norms[i][j]
			n2 = screw_norms[i][j-1]
			n3 = screw_norms[i-1][j-1]

			na = mulvec(addvec(addvec(n1, n2), n3), 1/3)
			line_norms.append(na)
		screw_norms_new.append(line_norms)
		screw_points_new.append(line_points)

	return screw_points_new, screw_norms_new

# surface must fulfill these requirements:
# neighboring vertices in the 2D array must be adjacent on the surface
# each curve must be the same length (array-wise, not distance-wise)
# 
# the output curves + surface omit the first and last curve of the input- to hide any discontinuities
def calc_norms(all_points):

	size_l = len(all_points[0])
	size_w = len(all_points)
	all_norms = []

	for i in range(0, size_w-1):
		line_norms = []
		for j in range(1, size_l-1):
			cv = all_points[i][j]
			pv = all_points[i][j-1]
			nv = all_points[i+1][j]

			l1 = (cv[0]-pv[0], cv[1]-pv[1], cv[2]-pv[2])
			l2 = (nv[0]-pv[0], nv[1]-pv[1], nv[2]-pv[2])

			n = normalize(cross(l1, l2))
			line_norms.append(n)
		all_norms.append(line_norms)

	# step 3- average normals for each vertex to be in the final output

	all_points_new = []
	all_norms_new = []
	for i in range(1, size_w-1):
		line_points = []
		line_norms = []
		for j in range(1, size_l-2):
			line_points.append(all_points[i][j])

			n1 = all_norms[i][j]
			n2 = all_norms[i][j-1]
			n3 = all_norms[i-1][j-1]

			na = mulvec(addvec(addvec(n1, n2), n3), 1/3)
			line_norms.append(na)
		all_norms_new.append(line_norms)
		all_points_new.append(line_points)

	return all_points_new, all_norms_new

def print_point(point):
	if len(point) == 3:
		return "<{:.2f}, {:.2f}, {:.2f}>".format(point[0], point[1], point[2])
	elif len(point) == 2:
		return "<{:.2f}, {:.2f}>".format(point[0], point[1])
	else:
		raise Exception('cant print non point')

def debug_skin(points, norms, radius):
	skin = []

	for norm, point in zip(norms, points):

		dx = norm[0] * radius
		dy = norm[1] * radius
		dz = norm[2] * radius

		skin.append((point[0]+dx, point[1]+dy, point[2]+dz))

	return skin
# y is "up" for these purposes
def bullnose_skin(points, norms, radius):
	skin = []

	for norm, point in zip(norms, points):
		#if norm[1] < 0:
		#	print(">>>", print_point(point), print_point(norm))
		#	print("\n".join([print_point(p) + ", " + print_point(n) for p, n in zip(points, norms)]))
			#raise Exception("cant cut upside down bits silly!")
		# dy = -r if ny = 0
		# dy = 0 if ny = 1

		dx = norm[0] * radius
		dy = (norm[1] - 1) * radius
		dz = norm[2] * radius

		skin.append((point[0]+dx, point[1]+dy, point[2]+dz))

	return skin

def zigzag_paths(paths):
	patho = []
	paths = list(paths)
	for i in range(0, len(paths), 2):
		patho.extend(paths[i])
		if (i+1) < len(paths):
			patho.extend(reversed(loop_skin[i+1]))
	return patho

def append_paths(paths):
	patho = []
	paths = list(paths)
	for i in range(0, len(paths), 2):
		patho.extend(paths[i])
		if (i+1) < len(paths):
			patho.extend(loop_skin[i+1])
	return patho

def square_skin(points, norms, radius):
	skin = []

	for norm, point in zip(norms, points):
		if norm[1] < 0:
			raise Exception("seriously. you can't cut upside down")

		dx = radius if norm[0] > 0 else -radius if norm[0] < 0 else 0
		dz = radius if norm[2] > 0 else -radius if norm[2] < 0 else 0
		dy = 0

		skin.append((point[0]+dx, point[1]+dy, point[2]+dz))

	return skin

def make_obj_lines(lines):

	obj = ""
	vi = 1
	for line in lines:
		for point in line:

			obj += "v {:.2f} {:.2f} {:.2f}\n".format(point[0], point[1], point[2])

		batchsize = 500
		for s in range((len(line)//batchsize)+1):
			end = (s*batchsize + batchsize + 1) if (s*batchsize + batchsize) < len(line) else len(line)+1
			obj += "l " + " ".join("{}".format(i) for i in range(vi+s*batchsize, vi+end)) + "\n"
			#obj += "l " + " ".join("{}".format(i) for i in range(vi, vi+len(line))) + "\n"
		vi += len(line)

	return obj

def distance(p1, p2):
	return sqrt(sum([(e1-e2)**2 for e1, e2 in zip(p1, p2)]))

# approximates dense path through removal of any very close-by vertices
def cleanpath(path, tolerance=0.02):
	newpath = []
	prev = path[0]
	prev_save = None
	i = 1
	while i < len(path):
		pt = path[i]
		d = distance(pt, prev)
		while d <= tolerance and i < len(path)-1:
			i += 1
			pt = path[i]
			d = distance(pt, prev)
		if i == (len(path)-1) and d <= tolerance:
			newpath.append(prev)
			newpath.append(pt)
		else:
			newpath.append(prev)
			newpath.append(path[i-1])
			prev = pt
			i += 1
		i += 1
	newpath.append(path[-1])
	return newpath, len(path)-len(newpath)
