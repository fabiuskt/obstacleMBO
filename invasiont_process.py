import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt 
import datetime
import cv2
import os

from jax.experimental import sparse

#construction of heatkernel in Fourier space
def build_transforme_heat_kernel_2d(N,M,t):
    ith, jth = jnp.meshgrid(np.arange(0,N,1), jnp.arange(0,M,1), indexing='ij')
    eigenvalues = N*N*(2-2 *jnp.cos(2*jnp.pi*ith/N)) + M*M*(2- 2 *jnp.cos(2*jnp.pi*jth/M)) 
    kernel = jnp.exp(-t*eigenvalues)/N*M
    return kernel


#convolution of heat kernel with 2d-array 
def diffuse_on_2dgrid(image, kernel):
    transformed_image = jnp.fft.fft2(image)
    product = transformed_image*kernel
    return jnp.fft.ifft2(product).real



barAsyst = 10000 #dimesionless systemsize 
initial_C = 0.3 #initial concentration of discs
N = 5000 #space discretization, number of pixels in x direction
M = N #number of pixels in y direction
# number of pixels in total is N*M

radius_obstacle =  np.sqrt(N*N/(np.pi*barAsyst)) #radius of obstacles for given system size barAsyst

t = np.pow(radius_obstacle/N,2)/16 #diffusion time h in dependence of obstacle radius
 
initial_number_of_obstacles = int(initial_C*N*N/(np.pi * radius_obstacle * radius_obstacle)) #initial number of obstacles depending on the concentration C

kernel = build_transforme_heat_kernel_2d(N,M,t) #build heat kernel

############construct initial conditions####################
np.random.seed(9234881)

a0, b0 = jnp.meshgrid(
    jnp.arange(-radius_obstacle, radius_obstacle+1),
    jnp.arange(-radius_obstacle, radius_obstacle+1),
    indexing="ij"
)

mask0 =  (a0*a0 + b0*b0) <= radius_obstacle**2   

mask0_sparse = sparse.BCOO.fromdense(mask0)
mask_inds = mask0_sparse.indices             


x_centers = np.random.randint(radius_obstacle, N - radius_obstacle, size=initial_number_of_obstacles)
y_centers = np.random.randint(radius_obstacle, M - radius_obstacle, size=initial_number_of_obstacles)

x_centers = jnp.asarray(x_centers)
y_centers = jnp.asarray(y_centers)

mask_inds_exp = mask_inds[None, :, :]   

shifts = jnp.stack([x_centers, y_centers], axis=1)[:, None, :]  

all_inds = mask_inds_exp + shifts

all_inds = all_inds.reshape(-1, 2)

data = jnp.ones(all_inds.shape[0], dtype=jnp.float32)

obstacle_one_hot = sparse.BCOO((data, all_inds), shape=(N, M)).sum_duplicates()
obstacle_one_hot = sparse.BCOO((jnp.ones(obstacle_one_hot.data.shape[0]), obstacle_one_hot.indices), shape=(N, M)).sum_duplicates()

obstacle_one_hot = obstacle_one_hot.todense()

image = obstacle_one_hot

##########################run scheme######################


timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"animation_{timestamp}"

#create directory for storing the results
if not os.path.exists(filename):
    os.makedirs(filename)

i = 0
prev = np.zeros((N,M))
while(np.sum(np.sum(prev))!= np.sum(np.sum(image))):#running until a steady state is reached
    prev= image

    diffused = diffuse_on_2dgrid(image, kernel)  #diffusion step

    #safe image
    if(i%3 == 0):      
        index_str = ("00000" + str(int(i/4)))[-5:]
        plt.imshow(image)
        plt.savefig(filename +"/" +index_str + '.png')
        plt.close()


    image = (diffused > 0.5) | (obstacle_one_hot > 0) #thresholding and updating on obstacles
    i+=1



