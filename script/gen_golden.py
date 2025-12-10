#!/usr/bin/env python3
import numpy as np
import copy
import math
import os
from PIL import Image
from fixedpoint import FixedPoint

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
    if 1/z_inv < canva.d:
        return
    
    
    h, w, _ = canva.array.shape
    
    x_idx = int(x + w / 2)
    y_idx = int(h / 2 - y)

    if x_idx < 0 or x_idx >= w or y_idx < 0 or y_idx >= h:
        return

    if z_inv <= 0:
        return

    if(z_inv > canva.z_inv_buf[y_idx][x_idx]):
        canva.z_inv_buf[y_idx][x_idx] = z_inv
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

def DrawFlatShadedTriangle_Q16_16(p0:list|FixedPoint, p1:list|FixedPoint, p2:list|FixedPoint, canva:Canva, color:tuple):
    points = [p0, p1, p2]

    # Sort points depended on y (P0 top/max Y, P2 bottom/min Y)
    if(points[1][1] > points[0][1]): points[1], points[0] = points[0], points[1]
    if(points[2][1] > points[0][1]): points[2], points[0] = points[0], points[2]
    if(points[2][1] > points[1][1]): points[2], points[1] = points[1], points[2]

    # Coordinates (Integer for Bresenham)
    x0, y0 = int(float(points[0][0])), int(float(points[0][1]))
    x1, y1 = int(float(points[1][0])), int(float(points[1][1]))
    x2, y2 = int(float(points[2][0])), int(float(points[2][1]))

    # Calculate gradients for Z and Brightness (FixedPoint division)
    # det = (x1-x0)(y2-y0) - (x2-x0)(y1-y0)
    fp_x0, fp_y0 = points[0][0], points[0][1]
    fp_x1, fp_y1 = points[1][0], points[1][1]
    fp_x2, fp_y2 = points[2][0], points[2][1]
    
    # Hardware optimization: Pre-calculate deltas
    dx1 = fp_x1 - fp_x0
    dy1 = fp_y1 - fp_y0
    dx2 = fp_x2 - fp_x0
    dy2 = fp_y2 - fp_y0

    dz1 = points[1][2] - points[0][2]
    dz2 = points[2][2] - points[0][2]

    db1 = points[1][3] - points[0][3]
    db2 = points[2][3] - points[0][3]
    
    det = dx1 * dy2 - dx2 * dy1
    
    if float(det) == 0: return

    # Hardware optimization: Calculate inverse determinant once to replace division with multiplication
    inv_det = 1.0 / det

    a_coff_for_z = (dz1 * dy2 - dz2 * dy1) * inv_det
    b_coff_for_z = (dz2 * dx1 - dz1 * dx2) * inv_det
    
    a_coff_for_brig = (db1 * dy2 - db2 * dy1) * inv_det
    b_coff_for_brig = (db2 * dx1 - db1 * dx2) * inv_det

    # Initialize Long Edge (P0 -> P2)
    lx, ly = x0, y0
    ldx = abs(x2 - x0)
    ldy = abs(y2 - y0)
    lsx = 1 if x0 < x2 else -1
    lsy = 1 if y0 < y2 else -1
    lerr = ldx - ldy

    # Initialize Short Edge 1 (P0 -> P1)
    sdx1 = abs(x1 - x0)
    sdy1 = abs(y1 - y0)
    ssx1 = 1 if x0 < x1 else -1
    ssy1 = 1 if y0 < y1 else -1

    # Initialize Short Edge 2 (P1 -> P2)
    sdx2 = abs(x2 - x1)
    sdy2 = abs(y2 - y1)
    ssx2 = 1 if x1 < x2 else -1
    ssy2 = 1 if y1 < y2 else -1

    # Initialize Current Short Edge (Start with Edge 1)
    sx, sy_curr = x0, y0
    sdx, sdy, ssx, ssy = sdx1, sdy1, ssx1, ssy1
    serr = sdx - sdy
    
    short_target_x, short_target_y = x1, y1
    using_second_short = False

    # Iterate Scanlines from P0.y down to P2.y
    for y in range(y0, y2 - 1, -1):
        
        # Update Long Edge (lx, ly) to match y
        while ly != y:
            if lx == x2 and ly == y2: break
            e2 = 2 * lerr
            if e2 > -ldy:
                lerr -= ldy
                lx += lsx
            if e2 < ldx:
                lerr += ldx
                ly += lsy
        
        # Switch Short Edge if needed
        if not using_second_short and y <= y1:
            using_second_short = True
            sx, sy_curr = x1, y1
            short_target_x, short_target_y = x2, y2
            sdx, sdy, ssx, ssy = sdx2, sdy2, ssx2, ssy2
            serr = sdx - sdy
        
        # Update Short Edge (sx, sy_curr) to match y
        while sy_curr != y:
            if sx == short_target_x and sy_curr == short_target_y: break
            e2 = 2 * serr
            if e2 > -sdy:
                serr -= sdy
                sx += ssx
            if e2 < sdx:
                serr += sdx
                sy_curr += ssy

        # Determine Left/Right
        if lx < sx:
            x_left, x_right = lx, sx
        else:
            x_left, x_right = sx, lx
            
        # Calculate Z and Brightness at start (x_left)
        dy_val = y - y0
        dx_val = x_left - x0
        
        z_px = points[0][2] + a_coff_for_z * dx_val + b_coff_for_z * dy_val
        b_px = points[0][3] + a_coff_for_brig * dx_val + b_coff_for_brig * dy_val
        
        # Draw Scanline
        for x in range(x_left, x_right + 1):
            draw_color = list(color)
            b_val = float(b_px)
            draw_color[0] *= b_val
            draw_color[1] *= b_val
            draw_color[2] *= b_val
            
            PutPixel(x, y, float(z_px), canva, draw_color)
            
            z_px += a_coff_for_z
            b_px += a_coff_for_brig



def main():
    os.makedirs("./images", exist_ok=True)
    canva = Canva(CANVA_WIDTH, CANVA_HEIGHT, DPI, DIS_CEMERA_CANVA)

    points = []
    for _ in range(3):
        x = np.random.randint(-CANVA_WIDTH//2, CANVA_WIDTH//2)
        y = np.random.randint(-CANVA_HEIGHT//2, CANVA_HEIGHT//2)
        z = np.float32(np.random.uniform(1000, 10000))
        brightness = np.float32(np.random.random())
        points.append((x, y, np.float32(1.0)/z, brightness))
    # fp32
    DrawFlatShadedTriangle_fp32(points[0], points[1], points[2], canva, COLOR)
    img = Image.fromarray(canva.array, mode="RGB")
    img.save(f"./images/fp32.png")
    canva.reset()

    # fixed point Q16_16
    fixedfloat_points = []
    for p in points:
        p_fp = p
        # p_fp = [FixedPoint(float(p[i]) , signed=True, m=16, n=16) for i, _ in enumerate(p)]
        fixedfloat_points.append(p_fp)
    DrawFlatShadedTriangle_Q16_16(fixedfloat_points[0], fixedfloat_points[1], fixedfloat_points[2], canva, COLOR)
    img = Image.fromarray(canva.array, mode="RGB")
    img.save(f"./images/Q16_16.png")

if __name__ == "__main__":
    main()

# TODO
# 1. Recognize Bresenham's line algorithm
# 2. Emlimitate # of division as more as possible
# 3. Turn it to Fixed point float
# 