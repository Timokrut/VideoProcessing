# STEP 1

# Generate video
# Amount of frames = 150
# Video length = 5 seconds
# FPS = 25
# Resolution = 320x240

# Background
# Background generates only ones and doesn't change
# Black-purple gradient from (25. 10, 40) to (60, 15, 70)

# Algorithm for creating a texture
# - create an array of gaussian noise with size of frame (seed = 13)
# - apply gaussian filter (sigma = 16)
# - normalize blured frame into [-1, 1] and multiply to 15
# - clip each chanel to [0, 255]

# Draw geometric figures at created background according to variant

# Move them according analytical traectory (coordinates of figure's center must be stored separately)

# Add noise according to variant

# Change RGB to YCbCr and modify

# Save video as one raw file (YUV420p - planar), fps=25

# Save file with exact coordinates/vectors of movement of objects by frames

import cv2
import math
import numpy as np
from pathlib import Path

def normalize(arr, a, b):
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    scaled_arr = a + (((arr - arr_min) * (b - a)) / (arr_max - arr_min))
    return scaled_arr

WIDTH = 320
HEIGHT = 240
FPS = 25
VIDEO_LENGHT = 5

SIGMA = 16
AMPLITUDE = 15
VARIANT = 13

FIRST_COLOR = np.array([25, 10, 40])
SECOND_COLOR = np.array([60, 15, 70])

AMOUNT_OF_FRAMES = FPS * VIDEO_LENGHT
FRAME_SIZE = WIDTH * HEIGHT

result_path = Path("results")
result_path.mkdir(parents=True, exist_ok=True)

background = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

generator = np.random.default_rng(VARIANT)

noise = generator.normal(0, 1, (HEIGHT, WIDTH)).astype(np.float32)
blurred = cv2.GaussianBlur(noise, (0, 0), sigmaX=SIGMA, sigmaY=SIGMA)
normalized = normalize(blurred, -1, 1) * AMPLITUDE

img1 = normalize(noise, 0, 255).astype(np.uint8)
img2 = normalize(blurred, 0, 255).astype(np.uint8)
cv2.imwrite("results/noise.png", img1)
cv2.imwrite("results/blurred.png", img2)

y = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None, None]
gradient = (FIRST_COLOR * (1 - y) + SECOND_COLOR * y)
gradient = np.broadcast_to(gradient, (HEIGHT, WIDTH, 3)).copy()
cv2.imwrite("results/gradient.png", cv2.cvtColor(gradient.astype(np.uint8), cv2.COLOR_RGB2BGR))

background = np.clip(gradient + normalized[:, :, None], 0, 255).astype(np.uint8)
cv2.imwrite("results/background.png", background)


CIRCLE_ONE_RADIUS = 18
CIRCLE_ONE_COLOR = (140, 230, 60)[::-1]

CIRCLE_TWO_RADIUS = 22
CIRCLE_TWO_COLOR = (40, 220, 210)[::-1]

SQUARE_SIZE = 26
SQUARE_COLOR = (255, 210, 60)[::-1]

def trace_circle_one(t):
    return (40 + 195 * t % WIDTH, 130)

def trace_circle_two(t):
    return (280 - 195 * t % WIDTH, 130)

def trace_square(t, size=SQUARE_SIZE):
    return (160, int(60 + 30 * math.sin(2 * math.pi * t)) % HEIGHT) , (160 + size, int(60 + 30 * math.sin(2 * math.pi * t) + size) % HEIGHT)

def add_snp_noise(img, amount, salt_vs_pepper=0.5):
    noisy = img.copy()

    random_values = generator.random((HEIGHT, WIDTH))

    pepper = random_values < amount * (1 - salt_vs_pepper)
    salt = random_values >= 1 - amount * salt_vs_pepper

    noisy[pepper] = 0
    noisy[salt] = 255

    return noisy
    
vf_path = Path("videoframes")
vf_path.mkdir(parents=True, exist_ok=True)

coordinates_circle_one = []
coordinates_circle_two = []
coordinates_square = []

with open("results/video_yuv420p.yuv", "wb") as yuv_file:
    for i in range(AMOUNT_OF_FRAMES):
        background_copy = background.copy()

        noise_levels = [0.002, 0.005, 0.01, 0.02, 0.04]
        noisy_background = add_snp_noise(background_copy, noise_levels[i % len(noise_levels)])

        cords_c1 = trace_circle_one(i)
        cords_c2 = trace_circle_two(i)
        cords_sq = trace_square(i)

        cv2.circle(noisy_background, center=cords_c1, radius=CIRCLE_ONE_RADIUS, color=CIRCLE_ONE_COLOR)
        cv2.circle(noisy_background, center=cords_c2, radius=CIRCLE_TWO_RADIUS, color=CIRCLE_TWO_COLOR)
        cv2.rectangle(noisy_background, pt1=cords_sq[0], pt2=cords_sq[1], color=SQUARE_COLOR)

        coordinates_circle_one.append(cords_c1)
        coordinates_circle_two.append(cords_c2)
        coordinates_square.append(cords_sq)

        ycrcb_background = cv2.cvtColor(noisy_background, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb_background[:, :, 0]
        Cr = ycrcb_background[:, :, 1]
        Cb = ycrcb_background[:, :, 2]

        Cr_420 = Cr[::2, ::2]
        Cb_420 = Cb[::2, ::2]

        yuv_file.write(Y.tobytes())
        yuv_file.write(Cb_420.tobytes())
        yuv_file.write(Cr_420.tobytes())

        cv2.imwrite(f"videoframes/{i}_background_with_figures.png", noisy_background)

with open("results/coordinates_circle_one.txt", "w") as file:
    for i in coordinates_circle_one:
        file.write(str(i) + "\n")

with open("results/coordinates_circle_two.txt", "w") as file:
    for i in coordinates_circle_two:
        file.write(str(i) + "\n")

with open("results/coordinates_sqare.txt", "w") as file:
    for i in coordinates_square:
        file.write(str(i) + "\n")