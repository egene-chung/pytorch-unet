import os
import numpy as np
from PIL import Image

def save_frames(image, indices, save_dir, prefix):
    for i, idx in enumerate(indices):
        image.seek(idx)
        frame = np.asarray(image)
        np.save(os.path.join(save_dir, f'{prefix}_{i:03d}.npy'), frame)

# 데이터 디렉토리 경로 및 파일 이름 설정
data_directory = './datasets'
label_filename = 'train-labels.tif'
input_filename = 'train-volume.tif'

# 이미지 불러오기
label_image = Image.open(os.path.join(data_directory, label_filename))
input_image = Image.open(os.path.join(data_directory, input_filename))

# 이미지 크기 및 프레임 수
num_frames = label_image.n_frames

# 프레임 수 설정
num_train_frames = 24
num_val_frames = 3
num_test_frames = 3

# 저장 디렉토리 설정
train_dir = os.path.join(data_directory, 'train')
val_dir = os.path.join(data_directory, 'val')
test_dir = os.path.join(data_directory, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# frame에 대하여 random index 생성
frame_indices = np.arange(num_frames)
np.random.shuffle(frame_indices)

# 프레임 인덱스 나누기
train_indices = frame_indices[:num_train_frames]
val_indices = frame_indices[num_train_frames:num_train_frames + num_val_frames]
test_indices = frame_indices[num_train_frames + num_val_frames:]

# 프레임 저장
save_frames(label_image, train_indices, train_dir, 'label')
save_frames(input_image, train_indices, train_dir, 'input')

save_frames(label_image, val_indices, val_dir, 'label')
save_frames(input_image, val_indices, val_dir, 'input')

save_frames(label_image, test_indices, test_dir, 'label')
save_frames(input_image, test_indices, test_dir, 'input')