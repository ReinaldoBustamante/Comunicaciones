import numpy as np
import cv2 as cv
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from scipy import fftpack
import time
def denoise(frame):
    #ELIMINADO RUIDO COMPULSIVO

    print("---------------FRAME----------------------")
    print(frame)
    print("------------------------------------------")

    print("---------------FILTRO RUIDO COMPULSIVO------------------")
    filtro = medfilt(frame)
    #filtro = filtro.astype('uint8')
    print(filtro)
    print("--------------------------------------------------------")
    frame = filtro
    
    # ELIMINADO RUIDO PERIODICO 

    def create_mask(dims, frequency, size=10):
        freq_int = int(frequency*dims[0])
        mask = np.ones(shape=(dims[0], dims[1]))
        mask[dims[0]//2-size-freq_int:dims[0]//2+size-freq_int, 
            dims[1]//2-size:dims[1]//2+size] = 0 
        mask[dims[0]//2-size+freq_int:dims[0]//2+size+freq_int, 
            dims[1]//2-size:dims[1]//2+size] = 0
        return mask

    S_img = fftpack.fftshift(fftpack.fft2(frame))
    espectro_filtrado = S_img*create_mask(S_img.shape, 0.03)   
    # Reconstrucción
    img_reconstructed = np.real(fftpack.ifft2(fftpack.ifftshift(espectro_filtrado)))
    img_reconstructed = img_reconstructed.astype('uint8')
    frame = img_reconstructed
    return frame

def code(frame):
    #
    # Implementa en esta función el bloque transmisor: Transformación + Cuantización + Codificación de fuente
    #    
    return frame


def decode(message):
    #
    # Reemplaza la linea 24...
    #
    frame = np.frombuffer(bytes(memoryview(message)), dtype='uint8').reshape(480, 848)
    #
    # ...con tu implementación del bloque receptor: decodificador + transformación inversa
    #    
    return frame
