import cv2
import numpy as np
import torch

from DenoisingModel import DenoisingNetwork

# Загрузка изображения
image = cv2.imread("image.jpg")

# Создание экземпляра модели и загрузка весов
model = DenoisingNetwork(3, 3, 2)
model.load_state_dict(torch.load("./saveModel/Denoising_RRG_model_epoch_3_blocks.pth"))
model.eval()

# Преобразование входного изображения в тензор PyTorch
image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()

# Применение модели к входному шумному изображению
with torch.no_grad():
    lnp = model(image_tensor)

# Преобразование тензора обратно в массив NumPy
lnp_array = lnp.squeeze(0).permute(1, 2, 0).cpu().numpy()

# Вычисление изображения с шумовым паттерном
L = image - lnp_array

# Отображение изображений
cv2.imshow("Original", image)
cv2.imshow("Denoised", lnp_array)
cv2.imshow("Noise Pattern", L)
cv2.waitKey(0)

b, g, r = cv2.split(L)

# Применение преобразования Фурье к каждому каналу
dft_b = cv2.dft(np.float32(b), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_g = cv2.dft(np.float32(g), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_r = cv2.dft(np.float32(r), flags=cv2.DFT_COMPLEX_OUTPUT)

# Обратное преобразование Фурье для каждого канала
idft_b = cv2.idft(dft_b)
idft_g = cv2.idft(dft_g)
idft_r = cv2.idft(dft_r)

# Вычисление модуля и нормализация изображений
magnitude_b = cv2.magnitude(idft_b[:, :, 0], idft_b[:, :, 1])
magnitude_g = cv2.magnitude(idft_g[:, :, 0], idft_g[:, :, 1])
magnitude_r = cv2.magnitude(idft_r[:, :, 0], idft_r[:, :, 1])

# Нормализация изображений
magnitude_b = cv2.normalize(magnitude_b, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
magnitude_g = cv2.normalize(magnitude_g, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
magnitude_r = cv2.normalize(magnitude_r, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Объединение каналов обратно в RGB изображение
result_image = cv2.merge((magnitude_b, magnitude_g, magnitude_r))

# Отображение результатов
cv2.imshow("Original", image)
cv2.imshow("DFT Blue Channel", magnitude_b)
cv2.imshow("DFT Green Channel", magnitude_g)
cv2.imshow("DFT Red Channel", magnitude_r)
cv2.imshow("Result Image", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
