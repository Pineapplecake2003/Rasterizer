#!/usr/bin/env python3
import numpy as np
import copy
import math
import os
from PIL import Image
from tqdm import tqdm
from fxpmath import Fxp

CANVA_WIDTH = 640
CANVA_HEIGHT = 480
DIS_CEMERA_CANVA = 1000
DPI=1
COLOR = (0x4E, 0xFE, 0xB3)
class Canva:
    z_inv_buf:np.ndarray
    array:np.ndarray
    width:np.int32
    height:np.int32
    dpi:np.int32
    d:int
    def __init__(self, width, height, dpi, d):
        self.width = width
        self.height = height
        self.dpi = dpi
        self.array      = np.full((height * dpi, width * dpi, 3), 255, dtype=np.uint8)
        self.z_inv_buf  = np.zeros((height, width), dtype=np.float32)
        self.d = d
    
    def reset(self):
        self.array      = np.full((self.height * self.dpi, self.width * self.dpi, 3), 255, dtype=np.uint8)
        self.z_inv_buf  = np.zeros((self.height, self.width), dtype=np.float32)

def PutPixel(x:int, y:int, z_inv:float, canva:Canva, color:tuple):
    if 1 < canva.d * z_inv:
        return
    
    h, w, _ = canva.array.shape
    
    x_idx = int(x + w / 2)
    y_idx = int(h / 2 - y)

    if x_idx < 0 or x_idx >= w or y_idx < 0 or y_idx >= h:
        return

    if z_inv <= 0:
        return

    if(float(z_inv) > canva.z_inv_buf[y_idx][x_idx]):
        canva.z_inv_buf[y_idx][x_idx] = float(z_inv)
        clamped = np.clip(np.round(color).astype(np.int32), 0, 255)
        canva.array[y_idx][x_idx][0] = clamped[0]
        canva.array[y_idx][x_idx][1] = clamped[1]
        canva.array[y_idx][x_idx][2] = clamped[2]

def toInt(val):
    if val > 0:
        return math.floor(val)
    else:
        return math.ceil(val)

def DrawFlatShadedTriangle_fp32(p0, p1, p2, canva, color):
    points = [p0, p1, p2]

    # Sort points depended on y
    # sort to
    #        0
    #       /|
    #      / |
    #     1  |
    #      \ |
    #       \|
    #        2
    # y based
    if(points[1][1] > points[0][1]):
        temp = points[1]
        points[1] = points[0]
        points[0] = temp
    
    if(points[2][1] > points[0][1]):
        temp = points[2]
        points[2] = points[0]
        points[0] = temp
    
    if(points[2][1] > points[1][1]):
        temp = points[2]
        points[2] = points[1]
        points[1] = temp
    
    if points[0][1] != points[1][1]:
        toP1_p = copy.deepcopy(points[0])
        toP2_p = copy.deepcopy(points[0])
    else:
        toP1_p = copy.deepcopy(points[1]) # 0 ________1          1 ________0
        toP2_p = copy.deepcopy(points[0]) #   \      /             \      /
                                          #    \    /               \    /
                                          #     \  /                 \  /
                                          #      \/                   \/
                                          #       2         or         2
    
    # vector [x, y]
    toP1_v_U = np.array([points[1][0], points[1][1]], dtype=np.int32) - np.array([points[0][0], points[0][1]], dtype=np.int32)
    toP1_v_U = toP1_v_U.astype(np.float32) / np.float32(abs(toP1_v_U[1])) if not np.isclose(toP1_v_U[1], 0.) else toP1_v_U.astype(np.float32)
    toP2_v_U = np.array([points[2][0], points[2][1]], dtype=np.int32) - np.array([points[0][0], points[0][1]], dtype=np.int32)
    toP2_v_U = toP2_v_U.astype(np.float32) / np.float32(abs(toP2_v_U[1])) if not np.isclose(toP2_v_U[1], 0.) else toP2_v_U.astype(np.float32)

    toP1_v_D = np.array([points[2][0], points[2][1]], dtype=np.int32) - np.array([points[1][0], points[1][1]], dtype=np.int32)
    toP1_v_D = toP1_v_D.astype(np.float32) / np.float32(abs(toP1_v_D[1])) if not np.isclose(toP1_v_D[1], 0.) else toP1_v_D.astype(np.float32)
    toP2_v_D = np.array([points[2][0], points[2][1]], dtype=np.int32) - np.array([points[0][0], points[0][1]], dtype=np.int32)
    toP2_v_D = toP2_v_D.astype(np.float32) / np.float32(abs(toP2_v_D[1])) if not np.isclose(toP2_v_D[1], 0.) else toP2_v_D.astype(np.float32)
    
    det = (points[1][0] - points[0][0]) * (points[2][1] - points[0][1]) - (points[2][0] - points[0][0]) * (points[1][1] - points[0][1])
    if det == 0:
        return
    
    a_coff_for_brig = (points[1][3] - points[0][3])*(points[2][1] - points[0][1]) - (points[2][3] - points[0][3])*(points[1][1] - points[0][1])
    a_coff_for_brig /= det

    b_coff_for_brig = (points[2][3] - points[0][3])*(points[1][0] - points[0][0]) - (points[1][3] - points[0][3])*(points[2][0] - points[0][0])
    b_coff_for_brig /= det

    a_coff_for_z = (points[1][2] - points[0][2])*(points[2][1] - points[0][1]) - (points[2][2] - points[0][2])*(points[1][1] - points[0][1])
    a_coff_for_z /= det

    b_coff_for_z = (points[2][2] - points[0][2])*(points[1][0] - points[0][0]) - (points[1][2] - points[0][2])*(points[2][0] - points[0][0])
    b_coff_for_z /= det

    while(toP1_p[1] > points[2][1]):
        if toP1_p[0] > toP2_p[0]:
            left_p, right_p = toP2_p, toP1_p
        else:
            left_p, right_p = toP1_p, toP2_p
        
        z_px_left = a_coff_for_z * (left_p[0] - points[0][0]) + \
                b_coff_for_z * (left_p[1] - points[0][1]) + points[0][2]
        z_px = z_px_left

        b_px_left = a_coff_for_brig * (left_p[0] - points[0][0]) + \
                b_coff_for_brig * (left_p[1] - points[0][1]) + points[0][3]
        b_px = b_px_left
        
        for x in range(int(left_p[0]), int(right_p[0])+1):
            draw_color = list(color)
            draw_color[0] = color[0] * b_px
            draw_color[1] = color[1] * b_px
            draw_color[2] = color[2] * b_px
            PutPixel(x, toInt(left_p[1]), z_px, canva, draw_color)
            b_px += a_coff_for_brig
            z_px += a_coff_for_z

        if toP1_p[1] > points[1][1]:
            # Above P1
            #print("U")
            toP1_v = toP1_v_U
            toP2_v = toP2_v_U
        else:
            # below P1
            #print("D")
            toP1_v = toP1_v_D
            toP2_v = toP2_v_D
        toP1_p, toP2_p = (toP1_p[0] + toP1_v[0], toP1_p[1] + toP1_v[1], 0, 0), (toP2_p[0] + toP2_v[0], toP2_p[1] + toP2_v[1], 0, 0)

def BresenhamFlatTriangle(p0, p1, p2, canva, color, fixedpoint=False, n_word=32, n_frac=16):
    points = [copy.deepcopy(p0), copy.deepcopy(p1), copy.deepcopy(p2)]
    # Sort points depended on y
    # sort to
    #        0
    #       /|
    #      / |
    #     1  |
    #      \ |
    #       \|
    #        2
    # y based
    if(points[1][1] > points[0][1]): points[1], points[0] = points[0], points[1]
    if(points[2][1] > points[0][1]): points[2], points[0] = points[0], points[2]
    if(points[2][1] > points[1][1]): points[2], points[1] = points[1], points[2]

    # P0 -> P1 short1
    # P1 -> P2 short2
    # P0 -> P2 long
    # short1
    s1points = [copy.deepcopy(points[0]), copy.deepcopy(points[1])]

    s1deltax = abs(s1points[1][0] - s1points[0][0])
    s1deltay = s1points[0][1] - s1points[1][1]

    s1err = s1deltay // 2
    s1x = s1points[0][0]
    s1xstep = 1 if s1points[0][0] < s1points[1][0] else -1
    
    # short2
    s2points = [copy.deepcopy(points[1]), copy.deepcopy(points[2])]

    s2deltax = abs(s2points[1][0] - s2points[0][0])
    s2deltay = s2points[0][1] - s2points[1][1]

    s2err = s2deltay // 2
    s2x = s2points[0][0]
    s2xstep = 1 if s2points[0][0] < s2points[1][0] else -1

    # long
    lpoints = [copy.deepcopy(points[0]), copy.deepcopy(points[2])]

    ldeltax = abs(lpoints[1][0] - lpoints[0][0])
    ldeltay = lpoints[0][1] - lpoints[1][1]

    lerr = ldeltay // 2
    lx = lpoints[0][0]
    lxstep = 1 if lpoints[0][0] < lpoints[1][0] else -1

    det = (points[1][0] - points[0][0]) * (points[2][1] - points[0][1]) - (points[2][0] - points[0][0]) * (points[1][1] - points[0][1])
    if det == 0:
        return
    a_coff_for_brig = (points[1][3] - points[0][3])*(points[2][1] - points[0][1]) - (points[2][3] - points[0][3])*(points[1][1] - points[0][1])
    a_coff_for_brig /= det

    b_coff_for_brig = (points[2][3] - points[0][3])*(points[1][0] - points[0][0]) - (points[1][3] - points[0][3])*(points[2][0] - points[0][0])
    b_coff_for_brig /= det

    a_coff_for_z = (points[1][2] - points[0][2])*(points[2][1] - points[0][1]) - (points[2][2] - points[0][2])*(points[1][1] - points[0][1])
    a_coff_for_z /= det

    b_coff_for_z = (points[2][2] - points[0][2])*(points[1][0] - points[0][0]) - (points[1][2] - points[0][2])*(points[2][0] - points[0][0])
    b_coff_for_z /= det

    if fixedpoint:
        a_coff_for_brig = Fxp(a_coff_for_brig, signed=True, n_word=n_word, n_frac=n_frac)
        a_coff_for_brig.config.op_sizing = 'same'
        b_coff_for_brig = Fxp(b_coff_for_brig, signed=True, n_word=n_word, n_frac=n_frac)
        b_coff_for_brig.config.op_sizing = 'same'
        a_coff_for_z = Fxp(a_coff_for_z, signed=True, n_word=n_word, n_frac=n_frac)
        a_coff_for_z.config.op_sizing = 'same'
        b_coff_for_z = Fxp(b_coff_for_z, signed=True, n_word=n_word, n_frac=n_frac)
        b_coff_for_z.config.op_sizing = 'same'
    for y in tqdm(range(points[0][1], points[2][1] - 1, -1)):
        if y > points[1][1]:
            # above y1
            if lx < s1x:
                left_x, right_x = lx, s1x
            else:
                left_x, right_x = s1x, lx
        else: 
            # lower than y1
            if lx < s2x:
                left_x, right_x = lx, s2x
            else:
                left_x, right_x = s2x, lx

        if fixedpoint:
            z_px_left = a_coff_for_z * Fxp(float(left_x - points[0][0]), signed=True, n_word=n_word, n_frac=n_frac) + \
                b_coff_for_z * Fxp(float(y - points[0][1]), signed=True, n_word=n_word, n_frac=n_frac) + Fxp(float(points[0][2]), signed=True, n_word=n_word, n_frac=n_frac)
            z_px = z_px_left

            b_px_left = a_coff_for_brig * Fxp(float(left_x - points[0][0]), signed=True, n_word=n_word, n_frac=n_frac) + \
                b_coff_for_brig * Fxp(float(y - points[0][1]), signed=True, n_word=n_word, n_frac=n_frac) + Fxp(float(points[0][3]), signed=True, n_word=n_word, n_frac=n_frac)
            b_px = b_px_left
        else: 
            z_px_left = a_coff_for_z * (left_x - points[0][0]) + \
                b_coff_for_z * (y - points[0][1]) + points[0][2]
            z_px = z_px_left

            b_px_left = a_coff_for_brig * (left_x - points[0][0]) + \
                b_coff_for_brig * (y - points[0][1]) + points[0][3]
            b_px = b_px_left
        
        # Update edges for next scanline
        # Long edge
        lerr -= ldeltax
        while lerr < 0:
            lx += lxstep
            lerr += ldeltay
        
        if y > points[1][1]:
            # Short edge 1
            s1err -= s1deltax
            if s1deltay != 0:
                while s1err < 0:
                    s1x += s1xstep
                    s1err += s1deltay
        else:
            # Short edge 2
            s2err -= s2deltax
            if s2deltay != 0:
                while s2err < 0:
                    s2x += s2xstep
                    s2err += s2deltay
        
        for x in range(left_x, right_x+1, 1):
            draw_color = list(color)
            draw_color[0] = color[0] * b_px
            draw_color[1] = color[1] * b_px
            draw_color[2] = color[2] * b_px
            
            if fixedpoint:
                PutPixel(x, y, float(z_px), canva, [float(c) for c in draw_color])
            else:
                PutPixel(x, y, z_px, canva, draw_color)

            b_px += a_coff_for_brig
            z_px += a_coff_for_z


def line(p0, p1, canva):
    # P0
    #   \
    #    \
    #     \
    #      \
    #       P1
    # Draw line from P0 to P1
    points = [copy.deepcopy(p0), copy.deepcopy(p1)]

    # To prevent steep slop, if delta x > delta y, need swap
    # If abs(dx) > abs(dy), we are "shallow" (x-major). 
    # To iterate over y (y-major), we swap x and y.
    steep = abs(points[1][0] - points[0][0]) > abs(points[1][1] - points[0][1])
    if steep:
        points[0][0], points[0][1] = points[0][1], points[0][0]
        points[1][0], points[1][1] = points[1][1], points[1][0]

    # Ensure we iterate from top (max y) to bottom (min y)
    if points[0][1] < points[1][1]:
        points[0], points[1] = points[1], points[0]
    
    deltax = abs(points[1][0] - points[0][0])
    deltay = points[0][1] - points[1][1]

    error = deltay // 2
    print(type(error))

    x = points[0][0]

    xstep = 1 if points[0][0] < points[1][0] else -1

    for y in range(points[0][1], points[1][1]-1, -1):
        if steep:
            PutPixel(y, x, 1/1500, canva, COLOR)
        else:
            PutPixel(x, y, 1/1500, canva, COLOR)
        error -= deltax
        if error < 0:
            x += xstep
            error += deltay



def calculate_snr(img1_path, img2_path):
    try:
        img1 = np.array(Image.open(img1_path)).astype(np.float64)
        img2 = np.array(Image.open(img2_path)).astype(np.float64)
    except Exception as e:
        print(f"Error opening images: {e}")
        return 0

    if img1.shape != img2.shape:
        print("Images have different dimensions")
        return 0

    signal_power = np.sum(img1 ** 2)
    noise_power = np.sum((img1 - img2) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def calculate_z_buffer_snr(z_buf1, z_buf2):
    if z_buf1.shape != z_buf2.shape:
        print("Z-buffers have different dimensions")
        return 0

    # Only compare pixels that have been drawn on (z > 0) in either buffer
    mask = (z_buf1 > 0) | (z_buf2 > 0)
    
    if not np.any(mask):
        return float('inf') # Both empty

    z1 = z_buf1[mask].astype(np.float64)
    z2 = z_buf2[mask].astype(np.float64)

    signal_power = np.sum(z1 ** 2)
    noise_power = np.sum((z1 - z2) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def main():
    os.makedirs("./images", exist_ok=True)
    canva = Canva(CANVA_WIDTH, CANVA_HEIGHT, DPI, DIS_CEMERA_CANVA)

    points = []
    for _ in range(3):
        x = np.random.randint(-CANVA_WIDTH//2, CANVA_WIDTH//2)
        y = np.random.randint(-CANVA_HEIGHT//2, CANVA_HEIGHT//2)
        z = np.float32(np.random.uniform(1000, 10000))
        brightness = np.float32(np.random.random())
        points.append([x, y, np.float32(1.0)/z, brightness])
    # fp32
    BresenhamFlatTriangle(points[0], points[1], points[2], canva, COLOR, fixedpoint=False)
    img = Image.fromarray(canva.array, mode="RGB")
    img.save(f"./images/fp32Flat.png")
    z_buf_fp32 = canva.z_inv_buf.copy()
    
    canva.reset()

    # Q16_16
    n_word = 32
    n_frac = 20
    BresenhamFlatTriangle(points[0], points[1], points[2], canva, COLOR, fixedpoint=True, n_word=n_word , n_frac=n_frac)
    img = Image.fromarray(canva.array, mode="RGB")
    img.save(f"./images/Q16_16Flat.png")
    z_buf_q16 = canva.z_inv_buf.copy()

    snr_img = calculate_snr("./images/fp32Flat.png", "./images/Q16_16Flat.png")
    print(f"Image SNR: {snr_img} dB")
    
    snr_z = calculate_z_buffer_snr(z_buf_fp32, z_buf_q16)
    print(f"Z-Buffer SNR: {snr_z} dB")

    print("Saving Z-buffers to text files...")
    np.savetxt("./images/z_buf_fp32.txt", z_buf_fp32, fmt='%.8f')
    np.savetxt("./images/z_buf_q16.txt", z_buf_q16, fmt='%.8f')

if __name__ == "__main__":
    main()
