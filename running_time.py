import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt 
import datetime
import time

from jax.experimental import sparse


#heat kernel in Fourierspace
def build_transforme_heat_kernel_2d(N,M,t):
    ith, jth = jnp.meshgrid(jnp.arange(0,N,1), jnp.arange(0,M,1), indexing='ij')
    eigenvalues = 2 *jnp.cos(2*jnp.pi*ith/N) + 2 *jnp.cos(2*jnp.pi*jth/M) - 4
    kernel = jnp.exp(t*eigenvalues)
    return kernel


#computation of solution of heat equation via convolution with heat kernel
def diffuse_on_2dgrid(image, kernel):
    transformed_image = jnp.fft.fft2(image)
    product = transformed_image*kernel
    return jnp.fft.ifft2(product).real


#one hot encoding
def diffused_to_onehot(u):
        onehot = jnp.zeros_like(u, dtype=jnp.float32)
        onehot[jnp.arange(u.shape[0]), jnp.argmax(u, axis=1)] = 1.0
        return onehot


####################### setup parameter #########################
barAsyst = 10000
initial_C = 0.3
times = []
iters = 100

#different side length of images for the tests
Ns = np.arange(500,10000,500)
for N in Ns:
    M = N
    radius_obstacle =  np.sqrt(N*N/(np.pi*barAsyst))
    t = np.pow(radius_obstacle/5,2)
    P = 2


    initial_number_of_obstacles = int(initial_C*N*N/(np.pi * radius_obstacle * radius_obstacle))

    initial_number_of_obstacles = 100

    kernel = build_transforme_heat_kernel_2d(N,M,t)

    ####################### init obstacles #########################


    image = jnp.zeros((N,M))
    image_one_hot = jnp.zeros((N, M))


    np.random.seed(9234881)


    obstacle_one_hot = sparse.empty((N,M))

    a,b = jnp.meshgrid(jnp.arange(0,N,1), jnp.arange(0,M,1))
    mask = (a- N/2)**2 + (b- M/2)**2 < radius_obstacle**2

    x = np.random.randint(low= -N/2 + radius_obstacle, high=N/2-radius_obstacle, size=initial_number_of_obstacles)
    y = np.random.randint(low=-M/2 + radius_obstacle, high=M/2-radius_obstacle, size=initial_number_of_obstacles)



    mask0 = sparse.BCOO.fromdense(mask)
    inds = mask0.indices
    for j in range(initial_number_of_obstacles):
        new_ind = jnp.concat(((inds[:,0] + x[j]).reshape((inds.shape[0],1)),(inds[:,1] + y[j]).reshape((inds.shape[0],1))), axis=1)
        new_data = jnp.ones(new_ind.shape[0])
        new_sparse = sparse.BCOO((new_data, new_ind), shape=(N,M))
        obstacle_one_hot = obstacle_one_hot + new_sparse

    obstacle_one_hot = obstacle_one_hot.sum_duplicates()
    obstacle_one_hot = sparse.BCOO((jnp.ones(obstacle_one_hot.indices.shape[0]), obstacle_one_hot.indices), shape=(N,M)).todense()



    image_one_hot = obstacle_one_hot


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"animation_{timestamp}"

    ####################### time measurement #########################

    start = time.time()


    for i in range(iters):
        diffused = diffuse_on_2dgrid(image_one_hot, kernel)
        image_one_hot = (diffused > 0.5) | (obstacle_one_hot > 0)
    end = time.time()
    times.append((end-start)/iters)
    print(times[-1])


plt.plot(Ns**2, times)
plt.xlabel(r'side length $\sqrt{N}$')
plt.ylabel(r'time per iteration in $s$')
plt.show()
print(times)



