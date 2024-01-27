# trying out point cloud examples with subdivide.py's fitting

import numpy as np
from subdivide import *

def get_random_sphere_ar():
    '''Generate a random point cloud on the unit sphere'''
    # generate random points on the sphere
    n = 1000
    phi = np.random.uniform(0, 2*np.pi, n)
    theta = np.random.uniform(0, np.pi, n)
    r = np.ones(n)
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    
    # convert to ar format
    points = np.array([x, y, z]).T
    return points

def get_random_sphere_noise(temperature=0.1):
    '''Generate a random point cloud on the unit sphere with noise'''
    # generate random points on the sphere
    n = 1000
    phi = np.random.uniform(0, 2*np.pi, n)
    theta = np.random.uniform(0, np.pi, n)
    r = np.ones(n)
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    
    # add noise
    x += np.random.normal(0, temperature, n)
    y += np.random.normal(0, temperature, n)
    z += np.random.normal(0, temperature, n)
    
    # convert to ar format
    points = np.array([x, y, z]).T
    return points

def get_random_torus_ar():
    '''Generate a random point cloud on the unit torus'''
    # generate random points on the torus
    n = 1000
    phi = np.random.uniform(0, 2*np.pi, n)
    theta = np.random.uniform(0, 2*np.pi, n)
    r = 0.5
    x = (1 + r * np.cos(theta)) * np.cos(phi)
    y = (1 + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    
    # convert to ar format
    points = np.array([x, y, z]).T
    return points

def get_random_torus_noise(temperature=0.1):
    '''Generate a random point cloud on the unit torus with noise'''
    # generate random points on the torus
    n = 1000
    phi = np.random.uniform(0, 2*np.pi, n)
    theta = np.random.uniform(0, 2*np.pi, n)
    r = 0.5
    x = (1 + r * np.cos(theta)) * np.cos(phi)
    y = (1 + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)
    
    # add noise
    x += np.random.normal(0, temperature, n)
    y += np.random.normal(0, temperature, n)
    z += np.random.normal(0, temperature, n)
    
    # convert to ar format
    points = np.array([x, y, z]).T
    return points    

if __name__ == '__main__':
    
    # run 4 test cases
    # 1. random sphere
    # 2. random sphere with noise
    # 3. random torus
    # 4. random torus with noise

    noise = 1e-2
    
    # 1. random sphere
    # main_subdivision_ar(array=get_random_sphere_ar(),plot_intermediate=True, savename='sphere')

    # 2. random sphere with noise
    main_subdivision_ar(array=get_random_sphere_noise(temperature=noise),plot_intermediate=True, savename=f'sphere_noise_{noise}')

    # 3. random torus
    # main_subdivision_ar(array=get_random_torus_ar(),plot_intermediate=True, savename='torus')

    # 4. random torus with noise
    main_subdivision_ar(array=get_random_torus_noise(temperature=noise),plot_intermediate=True, savename=f'torus_noise_{noise}')
    