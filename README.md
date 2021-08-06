# rotate-triaxial
The aim of the code is to calculate the properties of a galaxy hosting a supermassive black hole binary. Given a stellar particle distribution and a black hole binary, this code rotates the system according to the eigenvectors of the moment of inertia tensor and then calculates the cumulative angular momentum vector in the new reference frame.  This code was written for the purpose of the project  published in Avramov et al., 2021, Astronomy & Astrophysics, Volume 649, id.A41, 17 pp.


The code is divided into three main steps: 
- eigenvector calculation 
- particle distribution rotation
- angular momentum calculation

Eigenvector calculation: Using the eigenvectors of the moment of inertia tensor, the shape of the galaxy is determined. If the galaxy is triaxial, the eigenvectors will represent the three axes of the ellipsoid. 

Particle distribution rotation: In this step all of the particles are rotated according to the eigenvectors obtained in the last step, in order to orient the system according to its shape. 

Angular momentum calculation: In the new reference frame, the cumulative angular momentum is calculated in order to identify the rotation axis of the system. 


# Author
Branislav Avramov

# Date
January 2020


