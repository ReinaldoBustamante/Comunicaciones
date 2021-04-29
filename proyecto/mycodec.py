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
    #cv.imshow('1',frame)
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
    #cv.imshow('1',frame)
    return frame

def code(frame):
    #
    # Implementa en esta función el bloque transmisor: Transformación + Cuantización + Codificación de fuente
    #    
    #-------------------------Transformacion------------------------------------------

    #Se transforma a estandar YCbCr
    img_ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # separacion en bloques y transformacion con DCT
    DCT = lambda g, norm='ortho': fftpack.dct(fftpack.dct(g, axis=0, norm=norm), axis=1, norm=norm)
    IDCT = lambda G, norm='ortho': fftpack.idct(fftpack.idct(G, axis=0, norm=norm), axis=1, norm=norm)
    imsize = img_ycrcb.shape
    dct_matrix = np.zeros(shape=imsize)
    for i in range(0, imsize[0], 8):
        for j in range(0, imsize[1], 8):
            dct_matrix[i:(i+8),j:(j+8)] = DCT(img_ycrcb[i:(i+8),j:(j+8)])
    #print(dct_matrix)
    #---------------------------------------------------------------------------------
    #------------------------Cuantizacion---------------------------------------------
    
    #---------------------------------------------------------------------------------
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
