import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(image_path):
    # Đọc ảnh từ đường dẫn
    image = cv2.imread(image_path)

    # Resize ảnh về kích thước mong muốn (640x640)
    image = cv2.resize(image, (640, 640))

    # Chuẩn hóa dữ liệu bằng cách chia cho 255 (đưa giá trị pixel về khoảng [0, 1])
    image = image / 255.0

    # Augmentation - tăng cường dữ liệu
    datagen = ImageDataGenerator(
        rotation_range=20,  # Góc xoay tối đa là 20 độ
        width_shift_range=0.2,  # Dịch chuyển theo chiều rộng tối đa là 20% chiều rộng ảnh
        height_shift_range=0.2,  # Dịch chuyển theo chiều cao tối đa là 20% chiều cao ảnh
        shear_range=0.2,  # Góc nghiêng tối đa là 20 độ
        zoom_range=0.2,  # Tăng kích thước tối đa là 20%
        horizontal_flip=True,  # Lật ngang ảnh (ngẫu nhiên)
        fill_mode='nearest'  # Điền giá trị pixel bị mất sau khi áp dụng biến đổi
    )

    # Mở rộng chiều batch với một ảnh
    image = np.expand_dims(image, axis=0)

    # Tạo generator cho ảnh
    aug_iter = datagen.flow(image, batch_size=1)

    # Lấy ảnh sau khi tăng cường
    augmented_image = next(aug_iter)[0]

    # Chuyển đổi về định dạng uint8 và đưa giá trị về đúng phạm vi (0-255)
    augmented_image = (augmented_image * 255).astype(np.uint8)

    return augmented_image

# Đường dẫn của ảnh cần tiền xử lý
image_path = '/path/to/your/image'

# Tiền xử lý ảnh
preprocessed_image = preprocess_image(image_path)

# Lưu ảnh sau khi tiền xử lý
cv2.imwrite('/path/to/your/file', preprocessed_image)