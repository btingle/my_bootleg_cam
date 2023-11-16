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

# points : array of 2D x, y positions
# norms : array of normalized 2D normals per point
# offset : amount of shift in the x position over all iterations
# theta : total angle to screw over in the x axis
# iterations : number of vertices to generate per point
# returns array of 3D x, y, z positions
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
	# operation assumes that each curve line has the same number of vertices and are points on curves are roughly adjacent to their "neighbors" on nearby curves
	# i don't know how to state the requirement more precisely... indicates a lack of understanding on my part
	#
	# normals on first and last curve must be fudged unless continuation values are supplied
	# going to insist that continuation values are supplied
	#

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
		line_points.append(f(p, -d_theta_x, -d_x)) # continuation value for screw
		for i in range(iterations+1):
			line_points.append(f(p, d_theta_x*i, d_x*i))
		line_points.append(f(p, theta + d_theta_x, offset+d_x)) # also continuation
		screw_points.append(line_points)

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

	# step 2- calculate normals on each triangle surface defined by the rules listed above
	# step 3- average normals for each vertex to be in the final output
""" JAIL 
	for point, norm in zip(points, norms):

		#print("****", print_point(point), print_point(norm))

		line_points = []
		line_norms = []

		if axis == 'X':
			xp = point[0] - d_x
			yp = point[1] 
			zp = point[1] * sin(-d_theta_x)
		elif axis == 'Y':
			xp = point[0]
			yp = point[1] - d_x
			zp = point[0] * sin(-d_theta_x)

		for i in range(iterations+1):

			theta_x = d_theta_x * i

			n = (norm[0], norm[1], 0)
			if axis == 'X':
				x = point[0] + d_x * i
				y = point[1] * cos(theta_x)
				z = point[1] * sin(theta_x)
				n = rotate_x(n, -theta_x)
				#n = (point[0] + norm[0], point[1] + norm[1], 0)
				#n = rotate_x(n, theta_x)
				#n = (n[0] - x, n[1] - y, n[2] - z)
			elif axis == 'Y':
				x = point[0] * cos(theta_x)
				y = point[1] + d_x * i
				z = point[0] * sin(theta_x)
				n = rotate_y(n, -theta_x)
				#n = (point[0] + norm[0], point[1] + norm[1], 0)
				#n = rotate_y(n, theta_x)
				#n = (n[0] - x, n[1] - y, n[2] - z)

			if interpolate_normals:
				pv = normalize((x-xp, y-yp, z-zp))
				#pv = normalize(((x-xp), (y-yp), (z-zp)))
				tangent = cross(n, pv)
				tangent = normalize(tangent)
				
				n = cross(pv, tangent)
				n = (n[0], -n[1], n[2])
				n = normalize(n)
			xp = x
			yp = y
			zp = z
			#print(n, norm)

			#print(pv, tangent, n, norm)
			#print(x, y, z)
			#print(xp, yp, zp)

			line_points.append((x, y, z))
			try:
				line_norms.append(normalize(n))
			except:
				print(n, (x, y, z))
				print(line_points)
				print(line_norms)
				raise Exception('weehoo')


		# to not mess with continuity of normals
		line_norms[0] = line_norms[1]

		screw_points.append(line_points)
		screw_norms.append(line_norms)

	return screw_points, screw_norms
"""

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

	loop_skin_path = []
	for i in range(0, len(loop_skin), 2):
		loop_skin_path.extend(loop_skin[i])
		if (i+1) < len(loop_skin):
			loop_skin_path.extend(reversed(loop_skin[i+1]))

	return make_obj_lines(loop_points), make_obj_lines(loop_skin), make_obj_lines([loop_skin_path])

def helicaltrack(track_width, track_radius, track_spacing, track_height, track_inner_radius, iterations=64, side="TOP", single=False):
	if side == "BOT":
		base_points = [(((x/iterations)*track_width)+(track_radius-track_width), 0) for x in range(iterations+1)]
		base_norms  = [(0, 1) for x in range(iterations+1)]
	elif side == "TOP":
		base_points = []
		base_norms = []
		#base_points = [(track_radius, 0)]
		#base_norms = [(0, 1)]
		for i in range(iterations+1):
			t = pi * (i/iterations)
			x = cos(t) * track_inner_radius + (track_radius - track_width/2)
			y = -sin(t) * track_inner_radius
			y = y
			base_points.append((x, y))
			xn = cos(t)
			yn = sin(t)
			base_norms.append((xn, yn))
	elif side == "TOPSIDE":
		side_width = (track_width - (track_inner_radius*2))/2
		iterations_mod = int(iterations * (side_width/track_width) * 2)
		base_points = [(track_radius - side_width + side_width * (i/iterations_mod), 0) for i in range(iterations_mod+1)]
		base_points.extend([((track_radius - 2*(side_width + track_inner_radius)) + (side_width * (i/iterations_mod)), 0) for i in range(iterations_mod+1)])
		base_norms = []
		base_norms.extend([(0, 1) for i in range((iterations_mod+1)*2)])
		#base_points.append((track_radius-track_width, 0))
		#base_norms.append((0, 1))
	elif side == "INNER":
		base_points =      [(track_radius-track_width, (-1/iterations)*track_height)] # continuation point
		base_points.extend([((track_radius-track_width), (y/iterations)*track_height) for y in range(iterations+1)])
		base_points.append((track_radius-track_width, track_height+(1/iterations))) # cp
		base_points = list(reversed(base_points))

	#print(base_points)
	#print(base_norms)

	track_points, track_norms = screw(base_points, 6, pi/2, axis='Y')

	# # # # # # #
	# UL | UR # > H <
	#    +    # L   L
	# LL | LR # > H <
	# # # # # #

	if side == "INNER":
		track_norms = [[addvec(p, n) for p, n in zip(line_points, line_norms)] for line_points, line_norms in zip(track_points, track_norms)]
		track_points = [[rotate_x(p, -pi/2) for p in line_points] for line_points in track_points]
		track_norms = [[rotate_x(n,  -pi/2) for n in line_norms]  for line_norms in track_norms]
		track_points = [[rotate_z(p, -pi/4) for p in line_points] for line_points in track_points]
		track_norms = [[rotate_z(n,  -pi/4) for n in line_norms]  for line_norms in track_norms]
		track_norms = [[subvec(n, p) for p, n in zip(line_points, line_norms)] for line_points, line_norms in zip(track_points, track_norms)]
		pass
	if single == False:
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

	else:
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
	
	track_cpath = []
	for i in range(0, len(track_skin), 2):
		track_cpath.extend(track_skin[i])
		if (i+1) < len(track_skin):
			track_cpath.extend(track_skin[i+1])

	return make_obj_lines(track_paths), make_obj_lines(track_skin), make_obj_lines([track_cpath])


import sys
fn = sys.argv[1]

mode = 'helix'

base = fn + "_base.obj"
path = fn + "_path.obj"
cpath = fn + "_cpath.obj"
if mode == 'loop':
	obj_base, obj_path, obj_cpath = loopdeloop(pi/2, 36, 6, 12, 8, circle_iterations=32, iterations=64)
elif mode == 'helix':
	obj_base, obj_path, obj_cpath = helicaltrack(20, 28, 10, 12, 8, side="INNER", single=True)
#obj_base = loopdeloop(pi/2, 36, 6, 12, 8, iterations=64)

with open(base, 'w') as bf:
	bf.write(obj_base)
with open(path, 'w') as pf:
	pf.write(obj_path)
with open(cpath, 'w') as cf:
	cf.write(obj_cpath)