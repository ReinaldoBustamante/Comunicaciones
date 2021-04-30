import numpy as np
import cv2 as cv
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from scipy import fftpack
import time
import heapq

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
    img_rgb = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)
    img_ycrcb = cv.cvtColor(img_rgb, cv.COLOR_RGB2YCrCb)[:, :, 0]
    # separacion en bloques y transformacion con DCT
    DCT = lambda g, norm='ortho': fftpack.dct(fftpack.dct(g, axis=0, norm=norm), axis=1, norm=norm)

    imsize = img_ycrcb.shape
    dct_matrix = np.zeros(shape=imsize)
    for i in range(0, imsize[0], 8):
        for j in range(0, imsize[1], 8):
            dct_matrix[i:(i+8),j:(j+8)] = DCT(img_ycrcb[i:(i+8),j:(j+8)])
    #print(dct_matrix)
    #---------------------------------------------------------------------------------
    #------------------------Cuantizacion---------------------------------------------
    Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])

    def quantize(percent):
        im_dct = np.zeros(imsize)
        nnz = np.zeros(dct_matrix.shape)
        if (percent < 50):
            S = 5000/percent
        else:
            S = 200 - 2*percent
        Q_dyn = np.floor((S*Q + 50) / 100);
        Q_dyn[Q_dyn == 0] = 1
        for i in range(0, imsize[0], 8):
            for j in range(0, imsize[1], 8):
                quant = np.round(dct_matrix[i:(i+8),j:(j+8)]/Q_dyn)
                #print(quant)
        return quant#im_dct, np.sum(nnz)

    #imc, nnz = quantize(90)
    quant = quantize(90)
    #zigzag algorithm
    def zigzag(n, matrix):
    	#zigzag rows
    	def compare(xy):
    		x, y = xy
    		return (x + y, y if (x + y) % 2 else -y)
    	xs = range(n)
    	a = ((x, y) for x in xs for y in xs)
    	d_mat = {}
    	for i in a:
            d_mat[i] = matrix[i[1]][i[0]]
    	return {index: int(d_mat[index]) for j, index in enumerate(sorted(((x, y) for x in xs for y in xs),key=compare))}

    im_zz = list(zigzag(8, quant).values())

    #RLE
    def rlencoding(vect):
        rle = []
        count = 1
        for i in range(len(vect)-1):
            if (vect[i] == vect[i+1]):
                count += 1
            else:
                rle.append((count, vect[i]))
                count = 1
        rle.append((count, vect[-1]))
        return rle
    rle = rlencoding(im_zz)
    print(rle)

    #Huffman

    rle_cpy = rle.copy()
    rle_cpy[-1] = (1, 'EOB')

    #sumar de (x, y) los x
    symbol_count = 0
    for i in range(len(rle_cpy)):
        symbol_count += rle_cpy[i][0]

    #sumar de(x, y) los x con misma y
    d = {}
    for i in range(len(rle_cpy)):
        if(d[rle_cpy[i][1]]==None):
            d[rle_cpy[i][1]] = rle_cpy[i][0]
        else:
            d[rle_cpy[i]] += rle_cpy[i][0]

    dendo = [[freq/symbol_count, [symbol, ""]] for symbol, freq in d]
    heapq.heapify(dendo)

    while len(dendo) > 1:
        lo = heapq.heappop(dendo)
        hi = heapq.heappop(dendo)
        for code in lo[1:]:
            code[1] = '0' + code[1]
        for code in hi[1:]:
            code[1] = '1' + code[1]
        heapq.heappush(dendo, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    dendo = sorted(heapq.heappop(dendo)[1:])
    dendo = {symbol : code for symbol, code in dendo}

    def huffencoding(array, dendo):
        text = ""
        for c, v in array:
            text += (str(v)*c)
        htext = ""
        for l in text:
            htext += dendo[l]
        b = bytearray()
        for i in range(0, len(htext), 8):
            byte = htext[i:i+8]
            b.append(int(byte, 2))
        return b

    frame = huffencoding(rle_cpy, dendo)
    #---------------------------------------------------------------------------------
    return (frame, dendo)


def decode(message):
    #
    #Huffman
    frame, dendo = message
    
    #IQuantize
    #IDCT = lambda G, norm='ortho': fftpack.idct(fftpack.idct(G, axis=0, norm=norm), axis=1, norm=norm)
    #for i in range(0, imsize[0], 8):
    #    for j in range(0, imsize[1], 8):
    #        im_idct[i:(i+8),j:(j+8)] = IDCT(quant)
    #        nnz[i, j] = np.count_nonzero(quant)
    #return im_dct, np.sum(nnz)
    #
    frame = np.frombuffer(bytes(memoryview(message)), dtype='uint8').reshape(480, 848)
    #
    # ...con tu implementación del bloque receptor: decodificador + transformación inversa
    #
    return frame
