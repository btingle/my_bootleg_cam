from math import *
from cambootleg import *
from collections import defaultdict

# roughing procedure for ballnose and square end mills in 3 axis
# should allow for scan-line, unidirectional, contour, or combined roughing modes
# mesh can be used instead of path contours if desired

# gist of algorithm-
# divide space into Z-levels
# calculate where geometry intersects at particular Z-locations to extract Z-contour data
# sort vertices into buckets based on Z-level prior
# ... but how am i supposed to figure out the order of contour points?
# i mean.. just look at the closest vertex in euclidian space? too many problematic edge cases
# the thing is- a collection of paths does not make a solid geometry, i doubt there is a rigorous way to determine the order
# the paths must form a manifold- we can determine which point is next in the contour by calculating their distance on the manifold surface
# this only applies to points that share a manifold surface- for a mesh with many manifolds (unclosed surfaces) the results from each manifold must be patched together
#
# so, for a triangular mesh-
# if a triangle is intersected by a plane, there are three options for what can happen- two are edge cases
# 1 point is intersected, 2 points are intersected (most likely), or the triangle is entirely coplanar
# 2 point intersections are the meat and potatoes, but we must be aware of these edge cases and account for them nicely
# in a two point intersection, finding your neighbor intersections is easy! just look next to you!
# this should make generating the contours pretty easy (should)

# aftewards- we have a bunch of contours with 2D normals, sorted by Z-level. great.
# but not all of these contours can actually be cut- and some of them may not form closed shapes
# if a contour is fully contained within another, it should be excluded
# if a contour is partially intersecting another- the intersecting parts should be excluded from both
# this is where the algorithm might get complicated- it is a 2D intersection problem. there are probably efficient algorithms tho
# test intersection first with bounding box- then move on to edge-by-edge

def rough_3axis(verts, norms, tris, radius, startheight, endheight, cutdepth):#, area, margin):
    # z0 + dz*t = Z
    # t = (z - z0) / dz

    Zlevels = ceil((startheight-endheight)/cutdepth)
    edges_by_z = [[] for _ in range(Zlevels)]
    #counter = 0
    for tri in tris:
        def zsort(a, b, tri):
            #nonlocal counter
            i, v1 = min((a, verts[a]), (b, verts[b]), key=lambda x: x[1][2])
            j, v2 = max((a, verts[a]), (b, verts[b]), key=lambda x: x[1][2])

            zi = ceil((v1[2] - endheight) / cutdepth)
            zj = floor((v2[2] - endheight) / cutdepth)

            #if (zi < 0 and zj < 0) or (zi > Zlevels-1 and zj > Zlevels-1):
            #    return

            zi = max(min(Zlevels, zi), 0)
            zj = max(min(Zlevels-1, zj), -1)
            for z in range(zi, zj+1):
                edges_by_z[z].append((i, j, tri))
            
            #counter += 1
        
        zsort(tri[0], tri[1], tri)
        zsort(tri[1], tri[2], tri)
        if len(tri) == 4:
            zsort(tri[2], tri[3], tri)
            zsort(tri[3], tri[0], tri)
        else:
            zsort(tri[2], tri[0], tri)

    adjacency_by_z = [{} for _ in range(Zlevels)]
    contour_verts_by_z = [{} for _ in range(Zlevels)]
    for zi in range(Zlevels):
        result = adjacency_by_z[zi]
        result_verts = contour_verts_by_z[zi]
        result_tmp = {}
        Z = zi * cutdepth + endheight
        for a, b, tri in edges_by_z[zi]:
            def addresult(x, y):
                nonlocal result, result_tmp, result_verts
                r0 = result.get((a, b))
                rr = result_tmp.get(tri)
                if not r0:
                    result_verts[(a, b)] = (x, y, Z)
                if not rr:
                    result_tmp[tri] = (a, b)
                    if not r0:
                        result[(a, b)] = [tri]
                    else:
                        result[(a, b)].append(tri)
                else:
                    try:
                        i = result[rr].index(tri)
                        result[rr][i] = (a, b)
                    except:
                        pass
                    if not r0:
                        result[(a, b)] = [rr]
                    else:
                        result[(a, b)].append(rr)
                #text += "v {:.2f} {:.2f} {:.2f}\n".format(x, y, Z)
            v1, v2 = verts[a], verts[b]
            dz = v2[2]-v1[2]
            dx = v2[0]-v1[0]
            dy = v2[1]-v1[1]

            if dz == 0:
                if Z == v1[2]:
                    print(a, b, tri, v1, v2, "is coplanar")
                    #addresult(v1[0], v1[1])
                    #addresult(v2[0], v2[1])
            elif dx == 0 and dy == 0:
                addresult(v1[0], v1[1])
            else:
                t = ((Z - v1[2]) * dz)
                #if (t >= 0 and t <= 1):
                addresult(v1[0] + dx*t, v1[1] + dy*t)

    contour_lines = []
    for i in range(Zlevels):
        result_graph = adjacency_by_z[i]
        verts = contour_verts_by_z[i]
        visited = {}
        print(i, len(result_graph))
        for node in result_graph.keys():
            if visited.get(node):
                continue
            line = []
            orig = node
            prev = None
            pprev = None
            
            while True:
                if visited.get(node):
                    break
                visited[node] = True
                line.append(verts[node])
                nextn = None
                test_edge_valid = lambda e: len(e) == 2
                test_dead_end   = lambda e: len(list(filter(test_edge_valid, result_graph.get(e) or []))) > 1
                edges = (filter(lambda e: test_edge_valid(e) and test_dead_end(e), result_graph.get(node) or []))

                # so here is the theory-
                # if our mesh is all quads, and we construct this intersection point adjacency graph etc...
                # most nodes should have exactly two edges- one leading to each neighbor
                # some nodes may have three or more edges- only when the Z-plane intersects a point exactly
                # thus any polygon touching that point (typically four, in a regular quad mesh) is an edge
                # however- the "incorrect" edges should be dead ends, so we can nip them off and forget about them
                for edge in edges:
                    if edge == orig and prev != orig:
                        line.append(verts[orig])
                        break
                    if not edge == prev:
                        nextn = edge
                if not nextn:
                    break
                node = nextn
                prev = node
                pprev = prev

            contour_lines.append(line)

    return make_obj_lines(contour_lines)

    #return text

if __name__ == "__main__":
    verts, norms, tris = load_mesh('test_rough.obj')
    with open('test_rough_result.obj', 'w') as f:
        f.write(rough_3axis(verts, norms, tris, 6.35, 6, -6, 1))
    #save_mesh('test_rough_result.obj', verts, norms, tris)