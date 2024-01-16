from math import *

"""*************
COMMON FUNCTIONS
*************"""

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

"""*************
LOOP DE LOOP
*************"""

def loopdeloop(theta, loop_radius, loop_travel, track_height, track_radius, circle_iterations=32, iterations=64):
			
	base_points = []
	base_norms = []
	for i in range(circle_iterations+1):
		t = pi * (i/circle_iterations)
		x = cos(t) * track_radius
		y = (loop_radius - track_height) + sin(t) * track_radius
		y = -y
		base_points.append((x, y))
		xn = -cos(t)
		yn = sin(t)
		base_norms.append((xn, yn))

	# rotate everything 45deg back to form segment with least y height along principal axis
	loop_points, loop_norms = screw(base_points, base_norms, loop_travel, theta, iterations=iterations)
	lp_n, ln_n = [], []
	for line_pt, line_norm in zip(loop_points, loop_norms):
		lnp_n, lnn_n = [], []
		for pt, norm in zip(line_pt, line_norm):
			lnp_n.append(rotate_x(pt, pi/4))
			lnn_n.append(rotate_x(norm, -pi/4))
		lp_n.append(lnp_n)
		ln_n.append(lnn_n)

	loop_points = lp_n
	loop_norms = ln_n

	#return make_obj_lines(loop_points)
	loop_skin = []
	for line_points, line_norms in zip(loop_points, loop_norms):
		loop_skin.append(bullnose_skin(line_points, line_norms, 6.35/2))

	loop_skin_path = append_paths(loop_skin)

	return make_obj_lines(loop_points), make_obj_lines(loop_skin), make_obj_lines([loop_skin_path])

"""*************
HELICAL TRACK
*************"""

# factory function for making certain track geometries of interest
def helicaltrack(track_width, track_radius, track_spacing, track_height, track_inner_radius, iterations=64, side="TOP"):

	base_points = []

	if side == "BOT":
		base_points.extend[(((x/iterations)*track_width)+(track_radius-track_width), 0) for x in range(-1 iterations+2)]
	elif side == "TOP":
		for i in range(-1, iterations+2):
			t = pi * (i/iterations)
			x = cos(t) * track_inner_radius + (track_radius - track_width/2)
			y = -sin(t) * track_inner_radius
			y = y
			base_points.append((x, y))
	elif side.startswith("TOPSIDE"):
		side_width = 	 (track_width - (track_inner_radius*2))/2
		iterations_mod = int(iterations * (side_width/track_width) * 2)
		if side.endswith("OUTER"):
			base_points.extend([(track_radius - side_width + side_width * (i/iterations_mod), 0) for i in range(-1, iterations_mod+2)])
		if side.endswith("INNER")
			base_points.extend([((track_radius - 2*(side_width + track_inner_radius)) + (side_width * (i/iterations_mod)), 0) for i in range(-1, iterations_mod+2)])
	elif side == "INNER":
		base_points.extend([((track_radius-track_width), (y/iterations)*track_height) for y in range(-1, iterations+2)])
		base_points = reversed(base_points)

	track_points, track_norms = screw(base_points, 6, pi/2, axis='Y')
	return track_points, track_norms

def helicaltrack_assembly(track_width, track_radius, track_spacing, track_height, track_inner_radius, iterations=64, side="TOP"):

	track_points, track_norms = helicaltrack(track_width, track_radius, track_spacing, track_height, track_inner_radius, iterations=iterations, side=side)
	# # # # # # #
	# UL | UR # > H <
	#    +    # L   L
	# LL | LR # > H <
	# # # # # #
	
	ul_points = track_points
	ul_norms = track_norms

	ur_points = [[(-p[0]-track_spacing, p[1], p[2]) for p in line] for line in track_points]
	ur_norms = [[(-n[0], n[1], n[2]) for n in line] for line in track_norms]

	ll_points = [[(p[0], p[1], -p[2]-track_spacing) for p in line] for line in track_points]
	ll_norms = [[(n[0], n[1], -n[2]) for n in line] for line in track_norms]

	lr_points = [[(-p[0]-track_spacing, p[1], -p[2]-track_spacing) for p in line] for line in track_points]
	lr_norms = [[(-n[0], n[1], -n[2]) for n in line] for line in track_norms]

	track_paths = []
	track_paths_norms = []
		
	for i in range(len(track_points)):
		track_path = []
		track_path_norms = []
		track_path.extend(ul_points[i])
		track_path_norms.extend(ul_norms[i])

		n = (-base_norms[i][0], base_norms[i][1], 0)

		p, pp = ul_points[i][-1], ur_points[i][-1]
		track_path.append((p[0], p[1], p[2]))
		track_path.append((pp[0], pp[1], pp[2]))
		n = rotate_y(n, -pi/2)
		track_path_norms.extend([n, n])

		track_path.extend(reversed(ur_points[i]))
		track_path_norms.extend(reversed(ur_norms[i]))

		p, pp = ur_points[i][0], lr_points[i][0]
		track_path.append((p[0], p[1], p[2]-1))
		track_path.append((pp[0], pp[1], pp[2]+1))
		n = rotate_y(n, -pi/2)
		track_path_norms.extend([n, n])

		track_path.extend(lr_points[i])
		track_path_norms.extend(lr_norms[i])

		p, pp = lr_points[i][-1], ll_points[i][-1]
		track_path.append((p[0], p[1], p[2]))
		track_path.append((pp[0], pp[1], pp[2]))
		n = rotate_y(n, -pi/2)
		track_path_norms.extend([n, n])

		track_path.extend(reversed(ll_points[i]))
		track_path_norms.extend(reversed(ll_norms[i]))

		p, pp = ul_points[i][0], ll_points[i][0]
		track_path.append((pp[0], pp[1], pp[2]+1))
		track_path.append((p[0], p[1], p[2]-1))
		n = rotate_y(n, -pi/2)
		track_path_norms.extend([n, n])

		track_paths.append(track_path)
		track_paths_norms.append(track_path_norms)

	track_skin = []
	for line_points, line_norms in zip(track_paths, track_paths_norms):
		track_skin.append(bullnose_skin(line_points, line_norms, 6.35/2))
	
	track_cpath = append_paths(track_skin)

	return make_obj_lines(track_paths), make_obj_lines([track_cpath])

def helicaltrack_inner_single(track_width, track_radius, track_spacing, track_height, track_inner_radius, iterations=64):
	track_path, track_norms = helicaltrack(track_width, track_radius, track_spacing, track_height, track_inner_radius, iterations=64, side="INNER")
	track_norms =  [[addvec(p, n) for p, n in zip(line_points, line_norms)] for line_points, line_norms in zip(track_points, track_norms)]
	track_points = [[rotate_x(p, -pi/2)  for p in line_points] for line_points in track_points]
	track_norms =  [[rotate_x(n,  -pi/2) for n in line_norms ] for line_norms in track_norms]
	track_points = [[rotate_z(p, -pi/4)  for p in line_points] for line_points in track_points]
	track_norms =  [[rotate_z(n,  -pi/4) for n in line_norms ] for line_norms in track_norms]
	track_norms =  [[subvec(n, p) for p, n in zip(line_points, line_norms)] for line_points, line_norms in zip(track_points, track_norms)]

	track_paths = []
	track_paths_norms = []
	for i in range(0, len(track_points), 2):
		track_paths.append(track_points[i])
		track_paths_norms.append(track_norms[i])
		if i < (len(track_points)-1):
			track_paths.append(list(reversed(track_points[i+1])))
			track_paths_norms.append(list(reversed(track_norms[i+1])))

	track_skin = []
	for line_points, line_norms in zip(track_paths, track_paths_norms):
		track_skin.append(bullnose_skin(line_points, line_norms, 6.35/2))
	
	track_cpath = append_paths(track_skin)

	return make_obj_lines(track_paths), make_obj_lines([track_cpath])

"""
LOOP DE LOOP MAIN
"""

import sys
fn = sys.argv[1]

mode = 'loopdeloop'

base = fn + "_base.obj"
cpath = fn + "_cpath.obj"
obj_base, obj_path, obj_cpath = loopdeloop(pi/2, 36, 6, 12, 8, circle_iterations=32, iterations=64)

with open(base, 'w') as bf:
	bf.write(obj_base)
with open(cpath, 'w') as cf:
	cf.write(obj_cpath)

""" 
HELIX MAIN
"""

import sys
fn = 'helix'

mode = 'all' if len(sys.argv) < 2 else sys.argv[1]

iterations = 64
track_width = 20
track_radius = 28
track_assembly_spacing = 10
track_height = 12
track_inner_radius = 8

def writeout_obj(fn, base, path):
	with open(fn+'.base.obj', 'w') as bf, open(fn+'.path.obj', 'w') as pf:
		bf.write(obj_base)
		pf.write(obj_cpath)

if mode == "all":
	for mode in ["BOT", "TOP", "TOPSIDEOUTER", "TOPSIDEINNER"]:
		obj_base, obj_cpath = helicaltrack(track_width, track_radius, track_assembly_spacing, track_height, track_inner_radius, iterations=iterations, side=mode)
		writeout_obj(mode.lower(), obj_base, obj_cpath)
	obj_base, obj_cpath = helicaltrack_inner_single(track_width, track_radius, track_assembly_spacing, track_height, track_inner_radius, iterations=iterations)
	writeout_obj("inner", obj_base, obj_cpath)

else:
	if not mode == "INNER":
		obj_base, obj_cpath = helicaltrack(track_width, track_radius, track_assembly_spacing, track_height, track_inner_radius, iterations=iterations, side=mode)
		writeout_obj(mode.lower(), obj_base, obj_cpath)
	else:
		obj_base, obj_cpath = helicaltrack_inner_single(track_width, track_radius, track_assembly_spacing, track_height, track_inner_radius, iterations=iterations)
		writeout_obj("inner", obj_base, obj_cpath)