# ==== GPU selection ====
#from autocvd import autocvd
#autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================


import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt 
import datetime
import time
import os

import jax
from jax import lax

from functools import partial
from jax.experimental import sparse
from jax import debug


#Diffusion-Matrix in Fourier space
@partial(jax.jit, static_argnames=('N', 'M'))
def build_transforme_heat_kernel_2d(N, M, t):
    i = jnp.arange(N)[:, None]
    j = jnp.arange(M)[None, :]

    eigenvalues = (
        M * M * (2 - 2 * jnp.cos(2 * jnp.pi * i / N)) +
        M * M * (2 - 2 * jnp.cos(2 * jnp.pi * j / M))
    )
    return jnp.exp(-t * eigenvalues)


#solution to heat equation via convolution with heat kernel /diffusion matrix
@jax.jit
def diffuse_on_2dgrid_batch(images, kernel):
    # images: (B, N, M)
    fft_images = jnp.fft.fft2(images, axes=(-2, -1))
    out = jnp.fft.ifft2(fft_images * kernel, axes=(-2, -1))
    return out.real


#l^1 norm between two different sized images.
#The side length of the bigger image needs to be an integer multiple of the side length of the small image.
def l1(small_chi, big_chi):
   faktor = big_chi.shape[0]// small_chi.shape[0]
   small_chi = small_chi.repeat(faktor, 1).repeat(faktor,0)
   return jnp.sum(jnp.sum(jnp.abs(small_chi-big_chi)))


#estimation of radius by the area of a given set 
def comp_r(chi):
    area = chi.shape[0]*chi.shape[1]
    print(area)
    print(jnp.sum(jnp.sum(chi)))
    area_circle = jnp.sum(jnp.sum(chi))/area
    radius = jnp.sqrt(area_circle/jnp.pi)
    return radius


#implementation of the obstacleMBO algorithm. This code is written to  optionally run on a gpu.
def obstacle_MBO_gpu(configurations, obstacles_in, obstacles_out, kernel, max_steps, power):


    def cond_fn(state):
        _, _, moving, step = state
        return jnp.logical_and(jnp.any(moving), step < max_steps)

    def body_fn(state):

        configs, prev_sum, moving, step = state
        debug.print("step {}", step)

        diffused = diffuse_on_2dgrid_batch(configs, kernel)
        new_configs = jnp.logical_and(jnp.logical_or(diffused > 0.5, obstacles_in), jnp.logical_not(obstacles_out))
        new_sum = jnp.sum(new_configs, axis=(0,1))
        new_moving = new_sum != prev_sum


        return (
            new_configs,
            new_sum,
            new_moving,
            step + 1,
        )

    init_sum = jnp.sum(configurations, axis=(0,1))
    init_state = (
        jnp.array(configurations,dtype=bool),
        jnp.array(init_sum, dtype=jnp.int32),
        jnp.ones_like(init_sum, dtype=bool),
        0,
    )

    final_configs, _, _, _ = lax.while_loop(cond_fn, body_fn, init_state)

    return final_configs



#main method to setup parameters and build initial condition and then run obstacleMBO
#There are two different initial condition implemented which can be chosen by the <<circle>> variable.
def main(power=13, power_t=16, analyze=True, circle=False):
    M = int(2**power)
    N = int(2**(power-1))
    if(circle):
        N = int(2**power)

    #final_time = 1/2**6
    t = jnp.power(2.0,-power_t)
    reps = 2.0**(power_t-6)
    

    kernel = build_transforme_heat_kernel_2d(N,M,t)


    ################################## mask stripe ###########################


    a0, b0 = jnp.meshgrid(
        jnp.arange(0, N),
        jnp.arange(0, M),
        indexing="ij"
    )
    if(circle):
        mask_config = (a0 - N/2)**2 + (b0 - M/2)**2 <= (M/4)**2
        mask_balls_in = jnp.zeros_like(mask_config)
        mask_balls_out  = jnp.zeros_like(mask_config)
    else:
        mask_balls_in = jnp.logical_or((a0 - N/2)**2 + (b0 - (M/2 + M/3.5))**2 <= (M/8)**2,  (a0 - N/2)**2 + (b0 - (M/2 - M/3.5))**2 <= (M/8)**2)
        mask_balls_out = jnp.logical_or((a0 - N/2 - M/6)**2 + (b0 - (M/2 ))**2 <= (M/8)**2,  (a0 - N/2 + M/6)**2 + (b0 - (M/2 ))**2 <= (M/8)**2)
        mask_config = jnp.logical_and( jnp.logical_and(b0 >= 1.5*M/10, b0 <= 8.5*M/10) , jnp.logical_and(a0 >= 2*N/10, a0 <= 8*N/10))
        mask_config = jnp.logical_and(mask_config, jnp.logical_not(mask_balls_out))
        mask_config = jnp.logical_or(mask_config, mask_balls_in)
    plt.imshow(-0.25*mask_balls_in  + 0.3 * mask_balls_out + mask_config)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    configuration = obstacle_MBO_gpu(mask_config, mask_balls_in, mask_balls_out, kernel, reps ,power) 
    error = 0
    if(analyze):
        if(circle):
            exact = jnp.sqrt(-2*t*reps + 0.25**2)
            print(exact)
            prox = comp_r(configuration*1)
            print(prox)
            error = jnp.abs(exact-prox)
            error = prox
            print("radius error for " + str(power) + " " + str(power_t) + ": " + str(error) )

        else:
            exact = jnp.load("exact.npy")
            error = l1(configuration*1, exact*1)
            print("L1-error for " + str(power) + " " + str(power_t) + ": " + str(error) )
    
    plt.imshow(-0.25*mask_balls_in  + 0.3 * mask_balls_out + configuration)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return configuration, error


####################### setup parameter #########################

#Compute a very precise approximation to compare to
if(True):
    exact, _=main(13,15,False,circle=False)

#number of different gridpoint in time and space
nx = 7
nt = 7

powers = jnp.arange(6,13)
power_ts = jnp.arange(6,20,2)


errors = np.zeros((nx,nt))
for i,power in enumerate(powers):
   for j, power_t in enumerate(power_ts):
      _, errors[i,j] = main(power, power_t, True )

np.savetxt("final_errors", errors)
