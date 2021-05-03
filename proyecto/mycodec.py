import numpy as np
import cv2 as cv
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from scipy import fftpack
import time
import heapq
import ast
import json

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
    #print(frame.shape)
    #cv.imshow('1',frame)
    return frame

def code(frame):
    #
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
    original = np.prod(img_ycrcb.shape)*8/1e+6
    print("Imagen original: {:.3f} MB".format(original))
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
        Q_dyn = np.floor((S*Q + 50) / 100)
        Q_dyn[Q_dyn == 0] = 1
        quants = []
        for i in range(0, imsize[0], 8):
            for j in range(0, imsize[1], 8):
                quant = np.round(dct_matrix[i:(i+8),j:(j+8)]/Q_dyn)
                quants.append(quant)
        return quants

    quants = quantize(50)
    print(quants[0])
    #------------------------Codificación---------------------------------------------
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

    im_zz = []
    for i in range(len(quants)):
        im_zz.append(list(zigzagq(8, quants[i]).values()))

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

    rle =  []
    for i in range(len(im_zz)):
        rle.append(rlencoding(im_zz[i]))

    #Huffman

    rle_cpy = rle.copy()
    for i in rle_cpy:
        if i[-1][1] == 0:
            i[-1] = (1, 'EOB')

    def dendogram(rle):
        #sumar de (x, y) los x
        symbol_count = []
        for i in range(len(rle_cpy)):
            count = 0
            for j in range(len(rle_cpy[i])):
                count += rle_cpy[i][j][0]
            symbol_count.append(count)
        #sumar de(x, y) los x con misma y
        dendos = []
        for i in range(len(rle_cpy)):
            d = {}
            for j in range(len(rle_cpy[i])):
                if(rle_cpy[i][j][1] in d):
                    d[rle_cpy[i][j][1]] += rle_cpy[i][j][0]
                else:
                    d[rle_cpy[i][j][1]] = rle_cpy[i][j][0]
            den = [[freq/symbol_count[i], [str(symbol), ""]] for symbol, freq in d.items()]
            dendos.append(den)
        return dendos

    dendos = dendogram(rle_cpy)

    for i in range(len(dendos)):
        heapq.heapify(dendos[i])

        while len(dendos[i]) > 1:
            lo = heapq.heappop(dendos[i])
            hi = heapq.heappop(dendos[i])
            for code in lo[1:]:
                code[1] = '0' + code[1]
            for code in hi[1:]:
                code[1] = '1' + code[1]
            heapq.heappush(dendos[i], [lo[0] + hi[0]] + lo[1:] + hi[1:])

        dendos[i] = sorted(heapq.heappop(dendos[i])[1:])
        dendos[i] = {symbol : code for symbol, code in dendos[i]}


    def huffencoding(array, dendo):
        text = []
        for c, v in array:
            text += [str(v) for i in range(c)]
        htext = ""
        for l in text:
            htext += dendo[l]
        # b = bytearray()
        # for i in range(0, len(htext), 8):
        #     byte = htext[i:i+8]
        #     b.append(int(byte, 2))
        return htext

    # dict = str(dendos).encode()
    # frame = b''
    # for i in range(len(rle_cpy)):
    #     frame += huffencoding(rle_cpy[i], dendos[i])
    frame = ""
    for i in range(len(rle_cpy)):
        frame += huffencoding(rle_cpy[i], dendos[i])

    message = {
        'dict': dendos,
        'frame': frame,
        }
    #---------------------------------------------------------------------------------
    return json.dumps(message)


def decode(message):
    message = json.loads(message)
    #Huffman
    #dendos = ast.literal_eval(message[0].decode())
    dendos = message['dict']
    a = list(dendos)[0]
    print(a)
    fr = message['frame']
    print(fr[0:64])
    time.sleep(60)
    #fr = [value for k in range(len(fr)) for value in fr]
    #fr = ["{0:08b}".format(value) for value in fr]
    inv_dendos = []
    for dendo in dendos:
        inv_dendo =  {code: symbol for symbol, code in dendo.items()}
        inv_dendos.append(inv_dendo)

    code = ""
    zzs = []
    zz = []
    curr_den = 0
    for bit in fr:
        code += str(bit)
        if (curr_den < len(inv_dendos)) and (code in inv_dendos[curr_den]):
            zz.append(inv_dendos[curr_den][code])
            if (inv_dendos[curr_den][code] == 'EOB'):
                curr_den += 1
                zzs.append(zz)
                zz = []
            code = ""

    for i in range(len(zzs)):
        zeros = 64-len(zzs[i])
        if zzs[i][-1] == 'EOB':
            zzs[i][-1] = 0
        for j in range(zeros):
            zzs[i].append(0)

    print(len(zzs))

    # #IQuantize
    IDCT = lambda G, norm='ortho': fftpack.idct( fftpack.idct(G, axis=0, norm=norm), axis=1, norm=norm)

    def zigzag5(n):
        def compare(xy):
            x, y = xy
            return (x + y, -y if (x + y) % 2 else y)
        xs = range(n)
        return {index: n for n, index in enumerate(sorted(((x, y) for x in xs for y in xs),key=compare))}

    def rshp(zz):
        aux = zigzag5(8)
        relations = []
        for c, i in aux.items():
            relations.append((c, zz[i]))
        img_quant = [[0 for i in range(8)]for j in range(8)]
        for r in relations:
            img_quant[r[0][0]] [r[0][1]] = r[1]
        return np.array(img_quant)

    img_quant = []
    for i in range(len(zzs)):
        img_quant.append(rshp(zzs[i]))
    print(len(img_quant))

    img_size = (480, 848)

    im_idct = np.zeros(img_size)
    nnz = np.zeros(img_size)
    for i in range(0, img_size[0], 8):
        for j in range(0, img_size[1], 8):
            k = ((img_size[1]*i)//64)+(j//8)
            #print(k)
            quant = img_quant[k]
            im_idct[i:(i+8),j:(j+8)] = IDCT(quant)
            nnz[i, j] = np.count_nonzero(quant)
    #
    quality = np.sum(nnz)*8/1e+6
    print("50% de calidad {:.3f} MB".format(quality))
    #frame = np.frombuffer(bytes(memoryview(im_idct)), dtype='uint8').reshape(480, 848)
    #print(message.shape)
    # frame = np.frombuffer(bytes(memoryview(message)), dtype='uint8').reshape(480, 848)
    # ...con tu implementación del bloque receptor: decodificador + transformación inversa
    #
    return im_idct
