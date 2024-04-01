from math import *
# object to porn dot python
# turns any object- conceptual or otherwise- into internet pornography!
# all in a mere 20 lines of code!

import sys, os

obj_file = sys.argv[1]
prn_file = sys.argv[2]

spindle_speed = 9001
cutter_vel = 30 # 50 being max, for some reason
clearance_height = 10

# this will take all line objects from an obj file and concatenate them to form a prn, performing a repositioning move per line
# done very jank- we ignore the line data and assume all vertices are already ordered
# this happens to be the case when converting a long line to a curve prior to export in blender
# so big stretches of "l" lines will be treated as a delimiter between paths, and nothing else

with open(obj_file) as src, open(prn_file, 'w') as dest:

    pv = None
    prn = ";;^IN;!MC0;V50.0;^PR;Z0,0,20000;^PA;!RC{};!MC1;".format(spindle_speed)
    seen_line = True

    for line in src:

        if line.startswith('v'):
            xyz = line.strip().split()[1:]
            x = float(xyz[0])
            y = float(xyz[1])
            z = float(xyz[2])
            px = int(x*100)
            py = int(y*100)
            pz = int(z*100)
            if seen_line:
                prn += "V{:.1f};Z{},{},1000;\n".format(cutter_vel, px, py)
                seen_line = False
            zline = "Z{},{},{};\n".format(px, py, pz)
            prn += zline
            pv = (px, py, pz)

        if not seen_line and line.startswith('l'):
            prn += "V50.0;Z{},{},1000;\n".format(pv[0], pv[1])
            seen_line = True

    prn += "V50.0;^PA;!ZM0;!MC0;^IN;"
    dest.write(prn)
                