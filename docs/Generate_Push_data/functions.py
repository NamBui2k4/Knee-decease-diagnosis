import os

# Các định dạng file ảnh cần đếm
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# Hàm đếm số lượng file ảnh trong thư mục con và trả về dictionary
def count_images_in_subfolders(folder):
    image_count_dict = {}
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            image_count = len([f for f in os.listdir(subfolder_path) if f.lower().endswith(image_extensions)])
            image_count_dict[subfolder] = image_count
    return image_count_dict