import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw


class SyntheticShapesDataset(Dataset):
    def __init__(self, num_images, image_size=(128, 128), colors=["red", "green", "blue", "orange", "purple", "brown"], shape_types=["circle", "square", "rectangle", "triangle"]):
        self.num_images = num_images
        self.image_size = image_size
        self.colors = colors
        self.shape_types = shape_types

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        shape_type = np.random.choice(self.shape_types)
        color = np.random.choice(self.colors)
        image, mask = self.generate_image_and_mask(shape_type, color)
        return torch.from_numpy(image).permute(2, 0, 1).float(), torch.from_numpy(mask).unsqueeze(0).float()

    def generate_image_and_mask(self, shape_type, color):
        background_colors = [c for c in self.colors if c != color]
        background_color = np.random.choice(background_colors) if background_colors else "white"

        image = Image.new("RGB", self.image_size, color=background_color)
        mask = Image.new("L", self.image_size, color=0)

        draw_image = ImageDraw.Draw(image)
        draw_mask = ImageDraw.Draw(mask)

        if shape_type == "circle":
            radius = np.random.randint(20, min(self.image_size) // 4)
            upper_left = (np.random.randint(0, self.image_size[0] - radius),
                          np.random.randint(0, self.image_size[1] - radius))
            lower_right = (upper_left[0] + radius, upper_left[1] + radius)
            draw_image.ellipse([upper_left, lower_right], fill=color)
            draw_mask.ellipse([upper_left, lower_right], fill=255)

        elif shape_type == "square":
            side = np.random.randint(20, min(self.image_size) // 4)
            upper_left = (np.random.randint(0, self.image_size[0] - side),
                          np.random.randint(0, self.image_size[1] - side))
            lower_right = (upper_left[0] + side, upper_left[1] + side)
            draw_image.rectangle([upper_left, lower_right], fill=color)
            draw_mask.rectangle([upper_left, lower_right], fill=255)

        elif shape_type == "rectangle":
            width = np.random.randint(20, self.image_size[0] // 4)
            height = np.random.randint(20, self.image_size[1] // 4)
            upper_left = (np.random.randint(0, self.image_size[0] - width),
                          np.random.randint(0, self.image_size[1] - height))
            lower_right = (upper_left[0] + width, upper_left[1] + height)
            draw_image.rectangle([upper_left, lower_right], fill=color)
            draw_mask.rectangle([upper_left, lower_right], fill=255)

        elif shape_type == "triangle":
            point1 = (np.random.randint(0, self.image_size[0]),
                      np.random.randint(0, self.image_size[1]))
            point2 = (np.random.randint(0, self.image_size[0]),
                      np.random.randint(0, self.image_size[1]))
            point3 = (np.random.randint(0, self.image_size[0]),
                      np.random.randint(0, self.image_size[1]))
            draw_image.polygon([point1, point2, point3], fill=color)
            draw_mask.polygon([point1, point2, point3], fill=255)

        return np.array(image), np.array(mask)
