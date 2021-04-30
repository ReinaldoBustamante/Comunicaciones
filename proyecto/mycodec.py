import numpy as np
import cv2 as cv
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from scipy import fftpack
import time
import heapq
import ast

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
    #print(dct_matrix.shape)
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
        if (percent < 50):
            S = 5000/percent
        else:
            S = 200 - 2*percent
        Q_dyn = np.floor((S*Q + 50) / 100);
        Q_dyn[Q_dyn == 0] = 1
        for i in range(0, imsize[0], 8):
            for j in range(0, imsize[1], 8):
                quant = np.round(dct_matrix[i:(i+8),j:(j+8)]/Q_dyn)
        return quant

    #imc, nnz = quantize(90)
    quant = quantize(90)
    print(quant)
    #zigzag algorithm
    def zigzagq(n, matrix):
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

    im_zz = list(zigzagq(8, quant).values())

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
    #print(rle)

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
        if(rle_cpy[i][1] in d):
            d[rle_cpy[i][1]] += rle_cpy[i][0]
        else:
            d[rle_cpy[i][1]] = rle_cpy[i][0]
    #print(d)
    dendo = [[freq/symbol_count, [str(symbol), ""]] for symbol, freq in d.items()]
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
    #print(dendo)

    def huffencoding(array, dendo):
        text = []
        for c, v in array:
            text += [str(v) for i in range(c)]
        htext = ""
        for l in text:
            htext += dendo[l]
        b = bytearray()
        for i in range(0, len(htext), 8):
            byte = htext[i:i+8]
            b.append(int(byte, 2))
        return b

    frame = str(dendo).encode()+b'\n'
    frame += huffencoding(rle_cpy, dendo)
    #---------------------------------------------------------------------------------
    return frame


def decode(message):
    #Huffman
    msg = message.split(b'\n')

    dendo = ast.literal_eval(msg[0].decode())
    fr = msg[1]
    fr = [value for k in range(len(fr)) for value in fr]
    fr = ["{0:08b}".format(value) for value in fr]
    fr = ''.join(fr)

    inv_dendo =  {code: symbol for symbol, code in dendo.items()}

    code = ""
    zz = []
    for bit in fr:
        code += str(bit)
        if code in inv_dendo:
            zz.append(inv_dendo[code])
            code = ""
    temp = zz.index('EOB')
    zz = zz[:temp+1]
    zeros = 64-temp
    #print(zeros)
    zz[temp] = 0
    for i in range(zeros-1):
        zz.append(0)

    #IQuantize
    def zigzag(n):
        def compare(xy):
            x, y = xy
            return (x + y, -y if (x + y) % 2 else y)
        xs = range(n)
        return {index: n for n, index in enumerate(sorted(
            ((x, y) for x in xs for y in xs),key=compare))}.values()
    aux = list(zigzag(8))

    relations = []
    for i in range(len(aux)):
        relations.append((aux[i], zz[i]))
    relations = sorted(relations, key=lambda p: p[0])
    print(relations)
    img_quant = np.array([p[1] for p in relations])
    print(img_quant)
    img_size = (480, 848)

    IDCT = lambda G, norm='ortho': fftpack.idct(fftpack.idct(G, axis=0, norm=norm), axis=1, norm=norm)

    im_idct = np.zeros(img_size)
    nnz = np.zeros(img_size)
    for i in range(0, img_size[0], 8):
        for j in range(0, img_size[1], 8):
            im_idct[i:(i+8),j:(j+8)] = IDCT(img_quant)
            nnz[i, j] = np.count_nonzero(img_quant)

    #
    frame = np.frombuffer(im_idct, dtype='uint8').reshape(480, 848)
    #frame = np.frombuffer(bytes(memoryview(message)), dtype='uint8').reshape(480, 848)
    # ...con tu implementación del bloque receptor: decodificador + transformación inversa
    #
    return frame
