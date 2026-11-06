import math
import os
import re
import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def normalize(arr, a, b):
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    scaled_arr = a + (((arr - arr_min) * (b - a)) / (arr_max - arr_min))
    return scaled_arr


# константы
WIDTH = 320
HEIGHT = 240
FPS = 25
VIDEO_LENGHT = 5

SIGMA = 16
AMPLITUDE = 15
VARIANT = 13

FIRST_COLOR = np.array([40, 10, 25])
SECOND_COLOR = np.array([70, 15, 60])

AMOUNT_OF_FRAMES = FPS * VIDEO_LENGHT
FRAME_SIZE = WIDTH * HEIGHT

# дериктория для сохранения результатов
result_path = Path("results")
result_path.mkdir(parents=True, exist_ok=True)

background = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

# создаем шум
generator = np.random.default_rng(VARIANT)

noise = generator.normal(0, 1, (HEIGHT, WIDTH)).astype(np.float32)
blurred = cv2.GaussianBlur(noise, (0, 0), sigmaX=SIGMA, sigmaY=SIGMA)
normalized = normalize(blurred, -1, 1) * AMPLITUDE

img1 = normalize(noise, 0, 255).astype(np.uint8)
img2 = normalize(blurred, 0, 255).astype(np.uint8)
cv2.imwrite("results/noise.png", img1)
cv2.imwrite("results/blurred.png", img2)

# создаем background градиент
y = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None, None]
gradient = FIRST_COLOR * (1 - y) + SECOND_COLOR * y
gradient = np.broadcast_to(gradient, (HEIGHT, WIDTH, 3)).copy()
cv2.imwrite("results/gradient.png", gradient.astype(np.uint8))

background = np.clip(gradient + normalized[:, :, None], 0, 255).astype(np.uint8)
cv2.imwrite("results/background.png", background)


# константы для фигур
CIRCLE_ONE_RADIUS = 18
CIRCLE_ONE_COLOR = (140, 230, 60)[::-1]

CIRCLE_TWO_RADIUS = 22
CIRCLE_TWO_COLOR = (40, 220, 210)[::-1]

SQUARE_SIZE = 26
SQUARE_COLOR = (255, 210, 60)[::-1]


# траектории движения фигур
def trace_circle_one(t):
    t /= AMOUNT_OF_FRAMES
    return (int(40 + 195 * t), 130)


def trace_circle_two(t):
    t /= AMOUNT_OF_FRAMES
    return (int(280 - 195 * t), 130)


def trace_square(t, size=SQUARE_SIZE):
    t /= AMOUNT_OF_FRAMES
    return (160, int(60 + 30 * math.sin(2 * math.pi * t))), (
        160 + size,
        int(60 + 30 * math.sin(2 * math.pi * t) + size),
    )


# S&P шум
def add_snp_noise(img, p, gen):
    noisy = img.copy()

    random_values = gen.random((HEIGHT, WIDTH))

    pepper = random_values < p / 2
    salt = (random_values >= p / 2) & (random_values < p)
    noisy[pepper] = 0
    noisy[salt] = 255

    return noisy


# директория для кадров
vf_path = Path("videoframes")
vf_path.mkdir(parents=True, exist_ok=True)

coordinates_circle_one = []
coordinates_circle_two = []
coordinates_square = []

# константы для обработки видео
noise_levels = [0.002, 0.005, 0.01, 0.02, 0.04]
QP = [16, 22, 28, 34, 40]
GOP = 8


# прореживание
def downsample(channel):
    channel = channel.astype(np.float32)
    return (
        channel.reshape(HEIGHT // 2, 2, WIDTH // 2, 2)
        .mean(axis=(1, 3))
        .round()
        .astype(np.uint8)
    )


# сохраняем координаты только 1 раз
first_time = True

# создаем видео для каждого значения S&P
for j in range(len(noise_levels)):
    vfj_path = Path(f"results/NOISE_{noise_levels[j]}")
    vfj_path.mkdir(parents=True, exist_ok=True)

    # сохраняем .yuv файл и кадры отдельно
    with open(f"results/NOISE_{noise_levels[j]}/video_yuv420p.yuv", "wb") as yuv_file:
        for i in range(AMOUNT_OF_FRAMES):
            background_copy = background.copy()

            cords_c1 = trace_circle_one(i)
            cords_c2 = trace_circle_two(i)
            cords_sq = trace_square(i)

            cv2.circle(
                background_copy,
                center=cords_c1,
                radius=CIRCLE_ONE_RADIUS,
                color=CIRCLE_ONE_COLOR,
                thickness=-1,
            )
            cv2.circle(
                background_copy,
                center=cords_c2,
                radius=CIRCLE_TWO_RADIUS,
                color=CIRCLE_TWO_COLOR,
                thickness=-1,
            )
            cv2.rectangle(
                background_copy,
                pt1=cords_sq[0],
                pt2=cords_sq[1],
                color=SQUARE_COLOR,
                thickness=-1,
            )

            noisy_image = add_snp_noise(background_copy, noise_levels[j], generator)

            if first_time:
                coordinates_circle_one.append(cords_c1)
                coordinates_circle_two.append(cords_c2)
                coordinates_square.append(cords_sq)

            ycrcb_background = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2YCrCb)
            Y = ycrcb_background[:, :, 0]
            Cr = ycrcb_background[:, :, 1]
            Cb = ycrcb_background[:, :, 2]

            Cr_420 = downsample(Cr)
            Cb_420 = downsample(Cb)

            yuv_file.write(Y.tobytes())
            yuv_file.write(Cb_420.tobytes())
            yuv_file.write(Cr_420.tobytes())

            vfi_path = Path(f"videoframes/NOISE_{noise_levels[j]}")
            vfi_path.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(
                f"videoframes/NOISE_{noise_levels[j]}/{i}_background_with_figures.png",
                noisy_image,
            )

    # Сохраняем координаты для каждой из фигур
    if first_time:
        with open("results/coordinates_circle_one.txt", "w") as file:
            for i in coordinates_circle_one:
                file.write(str(i) + "\n")

        with open("results/coordinates_circle_two.txt", "w") as file:
            for i in coordinates_circle_two:
                file.write(str(i) + "\n")

        with open("results/coordinates_sqare.txt", "w") as file:
            for i in coordinates_square:
                file.write(str(i) + "\n")
    first_time = False

# Кодируем видео при S&P=1% со всеми значениями QP, GOP=8
for q in QP:
    qp_path = Path(f"results/QP/QP_{q}")
    qp_path.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [
            "ffmpeg",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            f"{FPS}",
            "-i",
            "results/NOISE_0.01/video_yuv420p.yuv",
            "-c:v",
            "libx264",
            "-qp",
            f"{q}",
            "-bf",
            "0",
            "-g",
            f"{GOP}",
            "-sc_threshold",
            "0",
            f"results/QP/QP_{q}/output.mp4",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    Path(f"results/QP/QP_{q}/enco.txt").write_text(result.stdout, encoding="utf-8")

bitrates = []
psnr_ffmpeg = []

# Считаем битрейт и PSNR
for q in QP:
    # битрейт
    file_path = f"results/QP/QP_{q}/output.mp4"
    file_size = os.path.getsize(file_path)

    bitrates.append((file_size * 8 * FPS) / (1000 * AMOUNT_OF_FRAMES))

    # PSNR
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            f"results/QP/QP_{q}/output.mp4",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            f"{FPS}",
            "-i",
            "results/NOISE_0.01/video_yuv420p.yuv",
            "-lavfi",
            "psnr",
            "-f",
            "null",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(f"encoding failure, QP={q}: {result.stdout}")

    Path(f"results/QP/QP_{q}/psnr.txt").write_text(result.stdout, encoding="utf-8")

    if result.returncode != 0:
        print(f"ffmpeg failure: {result.stdout}")
    else:
        matches = re.findall(
            r"\baverage:([0-9.]+|inf|nan)", result.stdout, re.IGNORECASE
        )
        if matches:
            psnr_ffmpeg.append(float(matches[-1]))
            print(f"PSNR average ffmpeg: {matches[-1]}")

        else:
            print("Average value wasn't find")

plt.plot(psnr_ffmpeg, bitrates)
for i in range(len(QP)):
    plt.annotate(f"QP={QP[i]}", (psnr_ffmpeg[i], bitrates[i]))
plt.title("Кривая зависимости PSNR от битрейта по всем значениям QP")
plt.xlabel("PSNR")
plt.ylabel("Bitrate")

plt.savefig("results/Bitrate2PSNR.png")
plt.close()
