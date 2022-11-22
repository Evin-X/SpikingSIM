import numpy as np
import multiprocessing as mlp
import matplotlib.pyplot as plt
import cv2 as cv

def load_real_spike(input_path):
    """
    Load raw spike data from .npy.
    args:
        - input_path
    return:
        - real spike data (400*250*T) 
    """
    return np.load(input_path, allow_pickle=True)

def load_image(input_path):
    """
    Load images.
    args:
        - input_path
    return:
        - grayscale image 
    """
    img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (400, 250), interpolation=cv.INTER_CUBIC)
    return np.array(img)/np.amax(img)

def Diffuse_Noise(lamb=25, num=10):
    """
    Generate Poisson distributed diffusion noise.
    args:
        - lamb: Expectation of interva
        - num: Output shape
    return:
        - The array records the location of noise
    """
    poisson = np.random.poisson(lamb, num)
    noise = np.zeros(250)
    key = 0
    for item in poisson:
        key += item
        if key > 249:
            break
        noise[key] = 0.001 * (np.random.random() - 0.5)
    return noise

def Inherent_Noise(T, mu=140, std=50):
    """
    Generate Gaussian distributed inherent noise.
    args:
        - T: Simulation time length
        - mu: Mean
        - std: Standard deviation
    return:
        - The array records the location of noise
    """
    shape = [250, 400, T]
    size = 250 * 400 * T
    gaussian = np.random.normal(mu, std, size).astype(np.int16)
    noise = np.zeros(size)
    key = 0
    for item in gaussian:
        key += item
        if key > size - 2:
            break
        noise[key] = 1
    noise = np.reshape(noise, shape).astype(np.int16)
    return noise

def Simulation(I, T=50):
    """
    Simulate the spike generation process of spiking cameras.
    args:
        - I: luminance intensity of a given pixel 
        - T: simulation time length (var: sim_time)
    return:
        - 0/1 binary array (spikes, shape: 1xT)
    """
    # Initialize Sensor Parameters 
    Vth = 2.0
    Eta = 10**(-13)*1.09
    Lambda = 10**(-4)*1.83
    Cpd = 10.0**(-15)*15
    CLK = 10.0**(6)*10
    delta_t = 2 / CLK
    K = delta_t * Eta / (Lambda * Cpd) 
    
    syn_spike = 0
    syn_rec = []
    vol = 0
    for n in range(T):
        g = np.random.rand() if n==0 else I  # Random Initialization 
        noise_1 = Diffuse_Noise()   # Initialize Diffusion Noise
        for t in range(250):
            g = g + noise_1[t]  # Add Diffusion Noise
            vol += g * K
            if vol >= Vth:
                syn_spike = 1
                vol = 0
        if syn_spike == 1:      # Readout Spikes
            syn_rec.append(1)
        else:
            syn_rec.append(0)
        syn_spike = 0

    return syn_rec

def Simulation_Simplified(I, T):
    """
    A simplified version of spiking simulator.
    args:
        - I: grayscale of a given image 
        - T: simulation time length 
    return:
        - 0/1 binary array (spikes, shape: TxHxW)
    """
    Vth = 1.0
    K = 0.45
    spk_rec = []
    g = np.random.rand(I.shape[0], I.shape[1])
    for t in range(T+1):
        g += K * I
        spk = np.where(g >= Vth, 1, 0)
        spk_rec.append(spk)
        g = np.where(g >= Vth, g - Vth, g)
        
    return np.array(spk_rec[0:T])

def Visualization(real, sim):
    """
    Display raw spikes and sim spikes.
    args:
        - real: real spike data (250*400*T) 
        - sim: simulated spike data (250*400*T) 
    return:
        - 
    """
    image_real = real.mean(axis=2) * 255
    image_sim = sim.mean(axis=2) * 255
    plt.ion()
    for t in range(sim.shape[2]):
        plt.subplot(2,2,1)
        plt.imshow(image_real, cmap='gray')
        plt.axis('off')
        plt.title('(a) Recon from Real Spike')
        plt.subplot(2,2,2)
        plt.imshow(image_sim, cmap='gray')
        plt.axis('off')
        plt.title('(b) Recon from Sim Spike')
        plt.subplot(2,2,3)
        plt.imshow(real[:,:,t], cmap='gray')
        plt.axis('off')
        plt.title('(c) Real Spike')
        plt.subplot(2,2,4)
        plt.imshow(sim[:,:,t], cmap='gray')
        plt.axis('off')
        plt.title('(d) Sim Spike')
        plt.show()
        plt.pause(0.1)
        plt.clf()
    plt.close()

def Display(image, sim):
    """
    Display input images and sim spikes.
    args:
        - image: input 
        - sim: simulated spike data (250*400*T) 
    return:
        - 
    """
    image_sim = sim.mean(axis=2) * 255
    plt.ion()
    for t in range(sim.shape[2]):
        plt.subplot(1,3,1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.title('(a) Input Image')
        plt.subplot(1,3,2)
        plt.imshow(sim[:,:,t], cmap='gray')
        plt.axis('off')
        plt.title('(b) Sim Spike')
        plt.subplot(1,3,3)
        plt.imshow(image_sim, cmap='gray')
        plt.axis('off')
        plt.title('(c) Recon from Sim Spike')
        plt.show()
        plt.pause(0.3)
        plt.clf()
    plt.close()


if __name__ == '__main__':

    # input_path = 'Pic/sample/real_spike.npz'
    input_path = 'Source/sample/car.jpg'
    sim_time = 50   
    sim_type = 'image' # 'image' 'spike'

    # Simulation from Raw Spikes or Images
    if sim_type == 'spike':
        raw_spike = load_real_spike(input_path)['arr_0']
        image = raw_spike.mean(axis=2)
    elif sim_type == 'image':
        image = load_image(input_path)
    
    # Parallel Acceleration of Simulation
    max_cpu = mlp.cpu_count()
    with mlp.Pool(max_cpu) as p:
        sim_spike = p.map(Simulation, image.flatten())
    sim_spike = np.reshape(sim_spike, [250, 400, sim_time]).astype(np.int16)
    noise = Inherent_Noise(sim_time)  # Add Inherent Noise
    syn_spike = np.bitwise_or(sim_spike, noise)
    
    # Display Simulation Results
    if sim_type == 'spike':
        Visualization(raw_spike, syn_spike)
    elif sim_type == 'image':
        Display(image, syn_spike)
