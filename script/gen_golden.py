#!/usr/bin/env python3
import numpy as np
import copy
import math
import os
from PIL import Image
import fixedpoint

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

def DrawFlatShadedTriangle_Q16_16(p0, p1, p2, canva, color):
    pass

def demo_fixed_point_usage():
    print("\n=== Fixed Point Usage Demo ===")
    # 1. 小數 (Float) 轉 FixedPoint
    val_float = 3.14159
    fp_val = fixedpoint.FixedPoint(val_float)
    print(f"Original Float: {val_float}")
    print(f"FixedPoint Object: {fp_val}")
    
    # 2. 獲得 FixedPoint 字串 (Hex 格式，常用於硬體驗證)
    hex_str = fp_val.to_hex()
    print(f"To Hex String: {hex_str}") # 例如 '0x3243f' (3.14159 * 65536)
    
    # 3. 從 Hex 字串轉回 FixedPoint 與 Float
    fp_restored = fixedpoint.FixedPoint.from_hex(hex_str)
    print(f"From Hex to Float: {fp_restored.to_float()}")
    
    # 4. 陣列操作範例
    arr_float = [0.5, -1.5, 2.0]
    fp_arr = fixedpoint.FixedPoint(arr_float)
    print(f"Array Hex: {fp_arr.to_hex()}")
    print("==============================\n")

def main():
    demo_fixed_point_usage()
    os.makedirs("./images", exist_ok=True)
    canva = Canva(CANVA_WIDTH, CANVA_HEIGHT, DPI, DIS_CEMERA_CANVA)

    points = []
    for _ in range(3):
        x = np.random.randint(-CANVA_WIDTH//2, CANVA_WIDTH//2)
        y = np.random.randint(-CANVA_HEIGHT//2, CANVA_HEIGHT//2)
        z = np.float32(np.random.uniform(1000, 10000))
        brightness = np.float32(np.random.random())
        points.append((x, y, np.float32(1.0)/z, brightness))
    DrawFlatShadedTriangle_fp32(points[0], points[1], points[2], canva, COLOR)
    img = Image.fromarray(canva.array, mode="RGB")
    img.save(f"./images/fp32.png")
    canva.reset()


if __name__ == "__main__":
    main()