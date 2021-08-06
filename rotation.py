import numpy as np
import numpy.linalg as npLA
import matplotlib.pyplot as plt
import scipy.linalg as sciLA
import glob

#---------------------------------------------
# Author: Branislav Avramov
# Date: Jan 2020 
# Description: Given a stellar particle distribution and a black hole binary, this code rotates the system according to the eigenvectors of the moment of inertia tensor and then calculates the cumulative angular momentum vector in the new reference frame. 
# This code was written for the purpose of the project  published in Avramov et al., 2021, Astronomy & Astrophysics, Volume 649, id.A41, 17 pp.
# !!! Attention:  Do not go to black hole center of mass  frame! stay in inertial system

#============================================================================

#functions for reading the data
def read_header(fname):
  with open(fname, 'rb') as dat:
    head = dat.readlines()[:3]
    snapshot_no = head[0]
    part_no = head[1]
    time = head[2]
    print(snapshot_no)
    print(part_no)
    print(time)
  return (snapshot_no,part_no, time)
  

def checkValue(value):
  if value == "---":
    value = np.ma.masked
  else:
    value = float(value)
  return value

def ReadData(fname):
	with open(fname, 'rb') as dat:

		for i in range(3):
			dat.readline()
		data_block = dat.readlines()	
	colnames = ('part_index', 'mass', 'x1', 'x2', 'x3', 'v1', 'v2', 'v3')
	data = {}
	for col_name in colnames:
		data[col_name] = np.zeros(len(data_block), 'f')
	for (line_count, line) in enumerate(data_block):
		items = line.split()
		for (col_count, col_name) in enumerate(colnames):
			value = items[col_count]
			data[col_name][line_count] = checkValue(value)
	return data

#function for making arrays of  r and v of data
def r_l_v(data):
  
	r_bh = np.array([data['x1'][0],data['x2'][0], data['x3'][0]])   # black hole 1
	r_bh2 = np.array([data['x1'][1],data['x2'][1], data['x3'][1]])  #black hole 2
	v_bh = np.array([data['v1'][0],data['v2'][0], data['v3'][0]])
	v_bh2 = np.array([data['v1'][1],data['v2'][1], data['v3'][1]])
	r_part = np.array([data['x1'][2:],data['x2'][2:], data['x3'][2:]])  #all particles except black holes
	v_part = np.array([data['v1'][2:],data['v2'][2:], data['v3'][2:]])
	r_tot = np.array([data['x1'],data['x2'], data['x3']])               #all particles
	v_tot = np.array([data['v1'],data['v2'], data['v3']])
	return r_bh,r_bh2, v_bh, v_bh2, r_part, v_part, r_tot, v_tot

def masses(data):
	mbh1 = data['mass'][0]
	mbh2 = data['mass'][1]
	m = data['mass'][2:]
	return mbh1, mbh2, m

def angular(r, v, m):   #function that calculates angular momentum
	l = np.cross(r, v, axis = 0)*m #ang mom vector of all particles
	print("in ang momentum")
	l_cum = np.sum(l, axis = 1) #cum ang momentum vector
	L_cum = np.linalg.norm(l_cum) #cum ang momentum amplitude
	L = np.linalg.norm(l, axis = 0) #amplitude of all particles
	return l, L, l_cum, l_cum/L_cum

def get_infl(mbh, m, R): #calculate influence radius of black holes
	R_sort = np.sort(R)
	ind = np.argsort(R)
	m_sort = m[ind]
	m_cum = np.cumsum(m_sort)
	print("m_cum",  m_cum)
	j = np.argwhere(m_cum > 2*(mbh))[0]
	print("Mass at infl radius: %.10E, bh binary mass*2: %.10E " %(m_cum[j], 2*(mbh1+mbh2)))
	return R_sort[j]
	
	
# position and velocity of BH center of mass
r00 = -3.313323E-02
r01 = -5.208672E-02
r02 = 1.595698E-01

v00 =  0.00886273808219
v01 =  0.00706501163546
v02 =  0.0205784912928	


file_list = glob.glob("00*.dat")
file_list.sort()

print("My file list is:", file_list)


#Define variables
l_cum = np.zeros([3, len(file_list)])
L_cum = np.zeros([3, len(file_list)])
l_bh = np.zeros([3, len(file_list)])
L_bh = np.zeros([3, len(file_list)])
l_bh2 = np.zeros([3, len(file_list)])
L_bh2 = np.zeros([3, len(file_list)])
l_tot = np.zeros([3, len(file_list)])
L_tot = np.zeros([3, len(file_list)])
tt = np.zeros([len(file_list)])

Inertia = np.zeros((3,3))
I_rot = np.zeros((3,3))

file1 = open('ang_mom_cum_python','w')
file2 = open('eigen_vectors', 'w')

for (j, file) in enumerate(file_list):
	#Read original file
	print("================")
	print("================")
	print("================")
	print("Starting to read file...")
	ff3 = open('eig_%s.march'%file, 'w')
	snap, part, time = read_header(file)
	data = ReadData(file)
	tt[j] = float(time)
	#include comoving correction for old data:
	if (tt[j] < 4.0):
		data['x1'] -= r00
		data['x2'] -= r01
		data['x3'] -= r02
		data['v1'] -= v00
		data['v2'] -= v01
		data['v3'] -= v02
	
	print("File read:", time)
	ids = data['part_index']
	r_bh, r_bh2,v_bh, v_bh2,  r_part, v_part, r_tot, v_tot = r_l_v(data)
	mbh1, mbh2, m = masses(data)
	R_star = np.linalg.norm(r_part, axis = 0)
	#find influence radius
	r_infl = get_infl(mbh1+mbh2, m, R_star)
	print("Finding eigenvectors within 5*R_infl:", 5*r_infl[0])
	inds_infl = np.where(R_star < 5*r_infl )

	#select particles within 5 influence radii
	my_r = r_part[:, inds_infl[0]].T
	my_v = v_part[:, inds_infl[0]].T
	my_m = m[inds_infl[0]]	
	
	#construct moment of inertia tensor
	Inertia[0][0] = np.sum(my_m*my_r[:, 0]*my_r[:, 0])  #xx
	Inertia[0][1] = np.sum(my_m*my_r[:, 0]*my_r[:, 1])  #xy
	Inertia[0][2] = np.sum(my_m*my_r[:, 0]*my_r[:, 2])  #xz
	Inertia[1][1] = np.sum(my_m*my_r[:, 1]*my_r[:, 1])  #yy
	Inertia[1][2] = np.sum(my_m*my_r[:, 1]*my_r[:, 2])  #yz
	Inertia[2][2] = np.sum(my_m*my_r[:, 2]*my_r[:, 2])  #zz

	Inertia[1][0] = Inertia[0][1] 	
	Inertia[2][0] = Inertia[0][2] 
	Inertia[2][1] = Inertia[1][2] 
	
	#find eigenvalues and eigenvectors and sort them in descending order
	ev_np, evec_np = npLA.eigh(Inertia)
	
	inds = np.argsort(-ev_np)
	ev_np = -np.sort(-ev_np)
	evec_np = evec_np[:, inds]
	
	#check if every first element of eigenvectors is positive, for consistent eigenvector sign notation
	for i in range(3):
		if evec_np[0, i]< 0:
			evec_np[:, i]=-evec_np[:, i]
	ff3.write('------------------------ \n')
	ff3.write('Eigenvalue: %.8e  and eigenvector 1: [%.8e \t %.8e  \t %.8e]  \n'% (ev_np[0], evec_np[0, 0], evec_np[1, 0], evec_np[2, 0]) )
	ff3.write('Eigenvalue: %.8e  and eigenvector 2: [%.8e \t %.8e  \t %.8e]  \n'% (ev_np[1], evec_np[0, 1], evec_np[1, 1], evec_np[2, 1]) )
	ff3.write('Eigenvalue: %.8e  and eigenvector 3: [%.8e \t %.8e  \t %.8e]  \n'% (ev_np[2], evec_np[0, 2], evec_np[1, 2], evec_np[2, 2]) )
	ff3.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n \n')
	ff3.close()

	#perform the rotation with matrix multiplication
	rr = np.matmul(r_tot.T, evec_np).T
	vv= np.matmul(v_tot.T, evec_np).T
	
	
	#find cumulative angular momentum
	l, L, L_cum[:, j], l_cum[:, j] = angular(rr[:, inds_infl[0]+2], vv[:, inds_infl[0]+2], my_m)
	print("ang mom:", l_cum[:, j])
	
	flag = 0
	#if angular momentum oriented along -z, perform 180 deg rotation 
	if (l_cum[2,j] < 0):
		print("in rotation ")
		rr[1] *= -1  #Rotation around x axis for 180 deg 
		rr[2] *= -1
		vv[1] *= -1
		vv[2] *= -1
		flag = 1
		l, L, L_cum[:, j], l_cum[:, j] = angular(rr[:, inds_infl[0]+2], vv[:, inds_infl[0]+2], my_m)



	#find agnular momentum of BH 1
	ll_bh = np.cross(rr[:, 0], vv[:, 0], axis = 0)
	LL_bh = np.sqrt(ll_bh[0]**2+ll_bh[1]**2+ll_bh[2]**2)
	L_bh[:, j] = ll_bh
	l_bh[:, j] =ll_bh/LL_bh
	#find angular momentum of BH 2
	ll_bh2 = np.cross(rr[:, 1], vv[:, 1], axis = 0)
	LL_bh2 = np.sqrt(ll_bh2[0]**2+ll_bh2[1]**2+ll_bh2[2]**2)
	L_bh2[:, j] = ll_bh2
	l_bh2[:, j] =ll_bh2/LL_bh2

	
	file1.write('%.10E \t  %.10E  \t  %.10E  \t  %.10E \t %.10E  \t  %.10E  \t  %.10E \t  %.10E  \t  %.10E  \t  %.10E \t  %.10E  \t  %.10E  \t  %.10E \n' % (tt[j], L_cum[0,j], L_cum[1,j], L_cum[2,j], l_cum[0,j], l_cum[1,j], l_cum[2,j], L_bh[0, j],L_bh[1, j],L_bh[2, j], L_bh2[0, j],L_bh2[1, j],L_bh2[2, j]))
	print("ang mom:", l_cum[:, j])

	file2.write('%.10E \t %d %.10E  \t  %.10E \t %.10E \t %.10E  \t  %.10E \t %.10E \t  %.10E  \t  %.10E \t %.10E'% (tt[j], flag, evec_np[0, 0], evec_np[1, 0], evec_np[2, 0],  evec_np[0, 1], evec_np[1, 1], evec_np[2, 1],  evec_np[0, 2], evec_np[1, 2], evec_np[2, 2]))   # time, mask eigenvec1, eigvec2, eigvec3
	print("===================\n ===================== /n =================")
	f = open(file+"_ROT_march", "w")
	#open file for writing rotated data
	f.write("%06d  \n" %int(snap))
	f.write("%07d \n" %int(part))
	f.write("%.10E \n" %tt[j])



	for (i, val) in enumerate(data['x1']):
		f.write("%07d \t %.10E \t % .10E % .10E % .10E \t % .10E % .10E % .10E \n" %(data['part_index'][i], data['mass'][i], rr[0,i], rr[1,i], rr[2,i], vv[0,i], vv[1,i], vv[2,i]))
	
		
f.close()




file1.close()
		
file2.close()
