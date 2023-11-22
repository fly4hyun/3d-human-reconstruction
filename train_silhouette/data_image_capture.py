











from vedo import *

# Load a polygonal mesh, make it white and glossy:
man = Mesh('output_2.obj')
man.c('white').lighting('glossy')

# Create two points:
p1 = Point([ 1,0,2], c='pink5')
p2 = Point([ 1,0,-2], c='orange')
p3 = Point([-1,0,2], c='blue4')
p4 = Point([-1,0,-2], c='green')

# Add colored light sources at the point positions:
l1 = Light(p1, c='pink5')
l2 = Light(p2, c='orange')
l3 = Light(p3, c='blue4')
l4 = Light(p4, c='green')

# Show everything in one go:
#show(man, l1, l2, l3, l4, p1, p2, p3, p4, axes=False, camera = {'pos':(0,0,4000), 'thickness':10,})
#screenshot('output_img.png')
#show(man, l1, l2, p1, p2, axes=False, camera = {'pos':(0,0,4000), 'thickness':10,})
vp = Plotter(interactive=False, offscreen=True)
#vp.show(man, axes=False, camera = {'pos':(0,0,4000), 'thickness':0,})


vp.show(man, l1, l2, l3, l4, p1, p2, p3, p4, axes=False, camera = {'pos':(0,0,4000), 'thickness':10,}).screenshot('output_img.png')
#aaa = man.show()
vp.screenshot('./output_img_p.png')
























