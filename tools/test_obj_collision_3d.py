import numpy as np
from obj_collision_3d import candidate_box,overlaps,check_collision

def vehicle(x=10,y=0,yaw=0):return candidate_box(x,y,0,[4,2,1.5],yaw)
a=vehicle(); assert overlaps(a,vehicle(13,0),0)
assert not overlaps(a,vehicle(15,0),0)
assert not overlaps(a,vehicle(10,2.6),.25)
assert overlaps(a,vehicle(12,1,45),0)
assert overlaps(a,vehicle(14.1,0),.2)
labels=[dict(Wx=10,Wy=0,Wz=.75,L=4,W=2,H=1.5,yaw=0,kind='car',tracking_id='real1')]
empty=np.empty((0,3))
assert check_collision(vehicle(10,0),labels,empty,ground_z=0)['reason']=='real_3d_overlap'
assert check_collision(vehicle(10,3),labels,empty,ground_z=0)['ok']
assert check_collision(vehicle(10,3),labels,None,ground_z=0)['reason']=='missing_lidar_or_ground'
points=np.array([[10.,3.,.8]]*6)
assert check_collision(vehicle(10,3),labels,points,ground_z=0)['reason']=='lidar_occupied'
ground=np.array([[10.,3.,.1]]*50)
assert check_collision(vehicle(10,3),labels,ground,ground_z=0)['ok']
print('collision tests passed')
