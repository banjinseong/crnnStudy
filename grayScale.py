import os
import random
import time
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import cv2

# Hangul Classification Used for Vehicle License Plates
korean = '가나다라마' \
        '거너더러머버서어저' \
        '고노도로모보소오조' \
        '구누두루무부수우주' \
        '아바사자하허호배'

#
new_img_path = './elc_plate_new.png'
old_img_path = './elc_plate_old.png'

# User default path through environment variables
user_profile = os.environ['USERPROFILE']

# Hangul font download path
# https://www.juso.go.kr/notice/NoticeBoardDetail.do?mgtSn=44&currentPage=11&searchType=&keyword=
ko_font = ImageFont.truetype(f'{user_profile}/AppData/Local/Microsoft/Windows/Fonts/한길체.ttf',
                            100, encoding='unic')
# Numeric font information
# https://fonts.google.com/noto/specimen/Noto+Sans+KR
font = ImageFont.truetype(f'{user_profile}/AppData/Local/Microsoft/Windows/Fonts/NotoSansKR-Medium.ttf',
                        120, encoding='unic')


def run():
    count, save_path = opt.count, opt.save_path

    start = time.time()

    # Make folder to saving outputs
    os.makedirs(f'{save_path}/newElc8', exist_ok=True)
    os.makedirs(f'{save_path}/oldElc8', exist_ok=True)
    os.makedirs(f'{save_path}/oldElc7', exist_ok=True)

    # 8-digit license plate with holographic
    for i in tqdm(range(count), desc='8-digit license plate(holographic)'):
        front = f'{random.randint(10,69)}'
        middle = random.choice(korean)
        back = f' {random.randint(1000, 9999)}'
        full_name = front + middle + back

        image_pil = Image.open(new_img_path)
        draw = ImageDraw.Draw(image_pil)
        # draw.text( (x,y), License plate string, Font color, Font )
        draw.text((80, -8), front, 'black', font)
        draw.text((215, 45), middle, 'black', ko_font)
        draw.text((315, -8), back, 'black', font)

        # Convert the PIL image to OpenCV format
        img_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # Save the image with grayscale applied
        cv2.imwrite(f'{save_path}/newElc8/' + full_name + '.png', img_cv2)

    # 8-digit license plate
    for i in tqdm(range(count), desc='8-digit license plate'):
        front = f'{random.randint(100, 999)}'
        middle = random.choice(korean)
        back = f' {random.randint(1000, 9999)}'
        full_name = front + middle + back

        image_pil = Image.open(old_img_path)
        draw = ImageDraw.Draw(image_pil)
        draw.text((40, -20), front, 'black', font)
        draw.text((245, 35), middle, 'black', ko_font)
        draw.text((340, -20), back, 'black', font)

        # Convert the PIL image to OpenCV format
        img_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # Save the image with grayscale applied
        cv2.imwrite(f'{save_path}/oldElc8/' + full_name + '.png', img_cv2)

    # 7-digit license plate
    for i in tqdm(range(count), desc='7-digit license plate'):
        front = f'{random.randint(10, 99)}'
        middle = random.choice(korean)
        back = f' {random.randint(1000, 9999)}'
        full_name = front + middle + back

        image_pil = Image.open(old_img_path)
        draw = ImageDraw.Draw(image_pil)
        draw.text((65, -20), front, 'black', font)
        draw.text((205, 30), middle, 'black', ko_font)
        draw.text((315, -20), back, 'black', font)

        # Convert the PIL image to OpenCV format
        img_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        # Save the image with grayscale applied
        cv2.imwrite(f'{save_path}/oldElc7/' + full_name + '.png', img_cv2)

    print(f'Done. ({round(time.time() - start, 3)}s)')  # Spending time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=30, help='Number of image to save')
    parser.add_argument('--save-path', type=str, default='result', help='Output path')
    opt = parser.parse_args()

    run()
