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

# axis should be normalized, pls
def rotate_axis(p, axis, theta):
	cost = cos(theta)
	sint = sin(theta)
	uxy = axis[0]*axis[1]
	uxz = axis[0]*axis[2]
	uyz = axis[1]*axis[2]
	x = (p[0] * (cost + axis[0]**2*(1-cost))  + p[1] * (uxy*(1-cost) - axis[2]*sint) + p[2] * (uxz*(1-cost) + axis[1]*sint))
	y = (p[0] * (uyx*(1-cost) + axis[2]*sint) + p[1] * (cost + axis[1]**2*(1-cost))  + p[2] * (uyz*(1-cost) - axis[0]*sint))
	z = (p[0] * (uxz*(1-cost) - axis[1]*sint) + p[1] * (uyz*(1-cost) + axis[0]*sint) + p[2] * (cost + axis[2]**2*(1-cost)) )
	return (x, y, z)

def mirrorX(p, pivot):
	d = p[0]-pivot
	return (pivot-d, p[1], p[2])
	

def cross(a, b):
	return a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]

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

	# roughing algorithm-
	# need to know geometry of workpiece and where it lies relative to the piece (for now we will assume rect prism, centered on origin)
	# need to figure out which areas are "inside" piece, and which are "outside"
	# algorithms I know break the piece into Z-levels, extract contours, and generate roughing from those contours
	# how to break into Z-levels?

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
# neighboring vertices in the matrix should be adjacent on the surface grid
# each curve must be the same length (array-wise, not distance-wise)
# 
# the output curves + surface omit the first and last curve of the input- to hide any discontinuities
def calc_norms(all_points):

	size_l = len(all_points[0])
	size_w = len(all_points)
	all_norms = []

	for i in range(0, size_w-1):
		line_norms = [None]
		for j in range(1, size_l):
			cv = all_points[i][j]
			pv = all_points[i][j-1]
			nv = all_points[i+1][j]

			l1 = (cv[0]-pv[0], cv[1]-pv[1], cv[2]-pv[2])
			l2 = (nv[0]-pv[0], nv[1]-pv[1], nv[2]-pv[2])

			n = normalize(cross(l1, l2))
			line_norms.append(n)
		all_norms.append(line_norms)
	all_norms.append(None)

	# step 3- average normals for each vertex to be in the final output
	all_points_new = []
	all_norms_new = []
	for i in range(1, size_w-1):
		line_points = []
		line_norms = []
		for j in range(1, size_l-1):
			line_points.append(all_points[i][j])

			n1 = all_norms[i][j] # cv
			n2 = all_norms[i][j+1]
			n3 = all_norms[i-1][j+1]

			na = mulvec(addvec(addvec(n1, n2), n3), 1/3)
			line_norms.append(na)
		all_norms_new.append(line_norms)
		all_points_new.append(line_points)

	return all_points_new, all_norms_new

# designed to work for output of calc_norms- i.e well-defined vertices that make up the whole of a surface
def make_mesh(all_points, all_norms=None):
	size_w = len(all_points)
	size_l = len(all_points[0])

	all_faces = []
	verts_flat = []
	norms_flat = []
	i = 1
	for k in range(len(all_points)):
		verts_flat.extend(all_points[k])
		if all_norms:
			norms_flat.extend(all_norms[k])
		tris = [
					((size_l*i + j-1), (size_l*i + j), (size_l*(i-1) + j), (size_l*(i-1) + j-1)) for j in range(2, size_l+1)
		]
		tris.append(((i*size_l + size_l), (i*size_l+1), ((i-1)*size_l+1), ((i-1)*size_l + size_l)))
		all_faces.extend(tris)
		i += 1

	print(len(all_faces), len(verts_flat), size_w, size_l)
	
	text = ""
	for point in verts_flat:
		text += "v {:.2f} {:.2f} {:.2f}\n".format(point[0], point[1], point[2])
	for norm in norms_flat:
		text += "vn {:.2f} {:.2f} {:.2f}\n".format(norm[0], norm[1], norm[2])
	for tri in all_faces:
		text += "f {} {} {} {}\n".format(tri[0], tri[1], tri[2], tri[3])

	return text

def load_mesh(fn):
	verts = []
	norms = []
	tris = []
	with open(fn, 'r') as f:
		for line in f:
			tok = line.strip().split()
			if tok[0] == 'v':
				verts.append((float(tok[1]), float(tok[2]), float(tok[3])))
			if tok[0] == 'vn':
				norms.append((float(tok[1]), float(tok[2]), float(tok[3])))
			if tok[0] == 'f':
				tok[1:] = [t.split('//')[0] for t in tok[1:]]
				if len(tok) > 4:
					tris.append((int(tok[1])-1, int(tok[2])-1, int(tok[3])-1, int(tok[4])-1))
				else:
					tris.append((int(tok[1])-1, int(tok[2])-1, int(tok[3])-1))
	return verts, norms, tris

def save_mesh(fn, verts, norms, tris):
	facemode = 'tri' if len(tris[0]) == 3 else 'quad' if len(tris[0]) == 4 else None
	with open(fn, 'w') as f:
		for vert in verts:
			f.write("v {:.2f} {:.2f} {:.2f}\n".format(vert[0], vert[1], vert[2]))
		for norm in norms:
			f.write("vn {:.2f} {:.2f} {:.2f}\n".format(norm[0], norm[1], norm[2]))
		if facemode == 'tri':
			for tri in tris:
				f.write("f {} {} {}\n".format(tri[0], tri[1], tri[2]))
		elif facemode == 'quad':
			for tri in tris:
				f.write("f {} {} {} {}\n".format(tri[0], tri[1], tri[2], tri[3]))

# expects 2d points that define a profile
def profile_cut_spiral_down(profile, depth, cutdepth):
	dists = [0]
	dist = 0
	prev = profile[0]
	for p in profile[1:]:
		dx = subvec(p, prev)
		dd = sqrt((dx[0]**2 + dx[1]**2))
		dists.append(dd)
		dist += dd
		prev = p

	path = [(profile[0][0], profile[0][1], 0)]
	z = 0
	c = 1

	while z < depth:
		p = profile[c]
		z += (dists[c] / dist) * cutdepth
		z  = min(z, depth)
		path.append((p[0], p[1], z))
		c = (c + 1) % len(profile)

	while c < len(profile):
		p = profile[c]
		path.append((p[0], p[1], depth))
		c += 1

	return path

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

def debug_norms(points, norms, radius=1):
	spikes = []

	for norm, point in zip(norms, points):

		dx = norm[0] * radius
		dy = norm[1] * radius
		dz = norm[2] * radius

		spikes.append([point, (point[0]+dx, point[1]+dy, point[2]+dz)])

	return make_obj_lines(spikes)
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

# I'm only going to consider ballnose tool- any other tool makes little sense for 4axis detailing
# returns vec3s- with y coordinate replaced by angle
def bullnose_skin_4axis(points, norms, radius):
	skin = []

	for norm, point in zip(norms, points):
		p = addvec(mulvec(norm, radius), point)
		theta = -atan2(p[2], p[1])
		p1  = rotate_x(p, theta)
		p0 = rotate_x(point,  theta)
		n  = subvec(p1, p0)
		skin.append( ( p1[0], (n[1] - 1) * radius + p0[1], theta ) )

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
			patho.extend(paths[i+1])
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

		#for point in line[:-1]:

		#	obj += "f {} {}\n".format(vi, vi+1)
		#	vi += 1

		#vi += 1
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
	newpath = [path[0]]
	prev = path[0]
	prev_i = 0
	i = 1
	while i < len(path):
		pt = path[i]
		d = distance(pt, prev)
		while d <= tolerance and i < len(path)-1:
			i += 1
			pt = path[i]
			d = distance(pt, prev)
		if i == (len(path)-1) and d <= tolerance:
			newpath.append(pt)
		else:
			newpath.append(pt)
			prev = pt
		i += 1
	return newpath, len(path)-len(newpath)

if __name__ == "__main__":
	pts = [(0, 1, 0), (0, 0, 1)]
	nrm = [(0, 0, 1), (0, 1, 0)]
	skin = bullnose_skin_4axis(pts, nrm, 1)
	print(skin)