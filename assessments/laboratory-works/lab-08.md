# МЕТОДИЧЕСКИЕ УКАЗАНИЯ ПО ВЫПОЛНЕНИЮ ЛАБОРАТОРНОЙ РАБОТЫ № 8

**Дисциплина:** Методы компьютерного зрения  
**Тема:** Диффузионные модели для генерации изображений.  
**Курс:** 3 курс, 6 семестр  
**Трудоемкость:** 4 акад. часа

## 1. Цели работы:

- изучить теоретические основы и математические принципы работы диффузионных моделей (DDPM, SDE-based);
- освоить практические навыки работы с современными библиотеками генеративного компьютерного зрения;
- научиться решать прикладные задачи генерации, модификации и анализа изображений с помощью латентных диффузионных моделей;
- научиться настраивать параметры генерации, реализовывать управляемую генерацию и комбинировать предобученные генеративные модели в прикладном пайплайне (`DL-2.1`, средний).

### Задачи работы:

- Развернуть рабочее окружение с поддержкой GPU-ускорения для запуска предобученных моделей.
- Реализовать программные скрипты для генерации изображений по текстовому описанию (Text-to-Image).
- Исследовать влияние гиперпараметров генерации на качество, структуру и соответствие итогового изображения промпту.
- Применить продвинутые техники кондиционирования: редактирование по маске, перенос стиля и оптимизацию промптов.
- Проанализировать метрики качества генерации и составить структурированный отчет.

### Необходимое ПО:

**Операционная система:** Linux (Ubuntu 22.04+) или Windows 10/11 с установленным WSL2.

**Среда выполнения:** Python 3.10+, Jupyter Lab / Google Colab (с GPU-ускорением T4/A100).

**Основные библиотеки:**

- torch, torchvision (фреймворк глубокого обучения).
- diffusers (библиотека Hugging Face для работы с диффузионными моделями).
- transformers, accelerate (для работы с текстовыми энкодерами и для оптимизации памяти).
- PIL, matplotlib, numpy (для обработки и визуализации изображений).

**Базовая модель:** Stable Diffusion (версии 1.5, 2.1 или SDXL-Turbo / Lightning для ускорения генерации).

### Настройка базового окружения

```python
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Проверка доступности вычислительного устройства
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используемое устройство: {device}")
if device == "cpu":
    print("ВНИМАНИЕ: используйте согласованную с преподавателем облегчённую модель или предоставленную GPU-среду.")
```

## ТЕОРИЯ

### Математические основы диффузионных моделей

В основе диффузионных моделей (в частности, DDPM — Denoising Diffusion Probabilistic Models) лежит концепция марковских цепей, которая разделяет процесс работы с изображением на два этапа: прямой (внесение шума) и обратный (генерация).

**Прямой процесс (Forward Process):** Марковская цепь добавляет гауссов шум

$$
[\,x_{0}\ (\text{Изображение})\,]\ --->\ [\,x_{1}\,]\ --->\ ...\ --->\ [\,x_{t}\,]\ --->\ ...\ --->\ [\,x_{T}\ (\text{Чистый шум})\,]
$$

**Обратный процесс (Reverse Process):** Нейросеть (U-Net) предсказывает и удаляет шум

$$
[\,x_{0}\ (\text{Результат})\,]\ <---\ [\,x_{1}\,]\ <---\ ...\ <---\ [\,x_{t}\,]\ <---\ ...\ <---\ [\,x_{T}\ (\text{Гауссов шум})\,]
$$

### 1. Прямой процесс диффузии (Forward Process)

Прямой процесс постепенно разрушает структуру исходного изображения $x_{0}$, добавляя к нему гауссов шум на каждом шаге $t\in\{1,\ldots,T\}$ в соответствии с заданным расписанием дисперсии (variance schedule) $\beta_{t}\in(0,1)$:

$$
q(x_{t}\mid x_{t-1})=\mathcal{N}(x_{t};\sqrt{1-\beta_{t}}\mathbin{\cdot}x_{t-1};\beta_{t}I).
$$

Благодаря свойствам гауссовского распределения зашумлённое изображение на любом шаге \(t\) можно получить напрямую из \(x_0\), не итерируя по всей цепи. Введём обозначения:

$$
\alpha_{t}=1-\beta_{t},\qquad \bar{\alpha}_{t}=\prod_{i=1}^{t}\alpha_{t}.
$$

Тогда:

$$
x_{t}=\sqrt{\bar{\alpha}_{t}}\mathbin{\cdot}x_{0}+\sqrt{1-\bar{\alpha}_{t}}\mathbin{\cdot}\varepsilon,\qquad \varepsilon\sim\mathcal{N}(0,1).
$$

При достаточно большом $T\ (T\approx1000)$ распределение $x_{T}$ практически полностью переходит в чистый белый шум.

### 2. Обратный процесс диффузии (Reverse Process)

Задача генерации — пройти эту цепь в обратном направлении: из случайного шума $x_{T}$ восстановить чистое изображение $x_{0}$. Распределение $p(x_{t-1}\mid x_{t})$ не имеет явного аналитического выражения (невозможно вычислить в радикалах), поэтому его аппроксимируют с помощью нейросети (обычно архитектуры U-Net) с параметрами $\theta$:

$$
p_\theta(x_{t-1}\mid x_t)=\mathcal{N}\!\left(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t)\right).
$$

Вместо прямого предсказания среднего значения $\mu_{\theta}$, сеть обучают предсказывать случайный шум $\varepsilon_{\theta}(x_{t},t)$, который был добавлен на шаге $t$. Функция потерь $(L_{simple})$ представляет собой среднеквадратичное отклонение (MSE) между реальным шумом и предсказанным:

$$
L_{simple}(\theta)=\mathbb{E}_{t,x_{0},\varepsilon}\left[\left\lVert\varepsilon-\varepsilon_{\theta}(x_{t},t)\right\rVert^{2}\right]
$$

### 3. Переход к латентной диффузии (Latent Diffusion / Stable Diffusion)

Классическая диффузия в пространстве пикселей требует колоссальных вычислительных ресурсов. В Latent Diffusion Models (LDM) процесс диффузии перенесен в компактное латентное пространство:

- **Энкодер** автокодировщика (VAE) сжимает изображение $x$ в латентный код $z=\mathcal{E}$.
- **Диффузия** (прямая и обратная) происходит внутри этого пространства $z$.
- **Декодер** $x_{gen}=\mathcal{D}(z_{denoised})$ восстанавливает финальное пиксельное изображение.

Для управления генерацией (текст, маски, контуры) в блоки U-Net интегрирован механизм Cross-Attention (перекрестного внимания), который сопоставляет признаки латентного шума с эмбеддингами условий (например, текста, полученного из модели CLIP).

## Задания для выполнения

Все задания представляют собой единый сквозной проект по созданию и модификации цифрового контента.

### Задание 1. Базовая генерация (Text-to-Image) и исследование шума.

**Инструкция:** загрузите модель Stable Diffusion через StableDiffusionPipeline. Сгенерируйте серию изображений по фиксированному промпту (например, "A futuristic city in the style of cyberpunk, highly detailed").

**Суть задачи:** Зафиксируйте случайное зерно (seed). Проведите генерацию, изменяя параметр количества шагов денойзинга (num_inference_steps) от 5 до 50 с шагом 10. Визуализируйте промежуточные этапы генерации (извлекая латенты на разной стадии), чтобы показать, как изображение "проявляется" из шума.

```python
# Загрузка предобученной модели Stable Diffusion v1.5
model_id = "runwayml/stable-diffusion-v1.5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
pipe = pipe.to(device)

prompt = "A futuristic city in the style of cyberpunk, highly detailed, 8k resolution"
generator = torch.Generator(device=device).manual_seed(42) # Фиксация seed для воспроизводимости

# Эксперимент: изменение количества шагов денойзинга
steps_list = [5, 15, 30, 50]
images = []

for steps in steps_list:
    print(f"Генерация для num_inference_steps = {steps}...")
    image = pipe(
        prompt=prompt, 
        num_inference_steps=steps, 
        generator=generator,
        guidance_scale=7.5
    ).images[0]
    images.append(image)

# Визуализация результатов задания 1
fig, axes = plt.subplots(1, len(steps_list), figsize=(20, 5))
for ax, img, steps in zip(axes, images, steps_list):
    ax.imshow(img)
    ax.set_title(f"Steps: {steps}")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

### Задание 2. Анализ влияния параметров кондиционирования.

**Инструкция:** используйте тот же prompt и seed.

**Суть задачи:** исследуйте параметр шкалы соответствия тексту (guidance_scale / CFG scale). Проведите генерацию для значений CFG = 1.0 (без текста), 3.0, 7.5 (стандарт), 15.0 и 30.0. Постройте матрицу изображений (Grid), объясните физический и математический смысл перенасыщения (over-saturation) картинки при слишком высоких значениях CFG.

Дополнительно замените планировщик на `DDIMScheduler` или `EulerAncestralDiscreteScheduler` через `from_config`, повторите один эксперимент с тем же seed и сравните результат и время генерации.

```python
# Используем ту же модель pipe и prompt
cfg_values = [1.0, 3.0, 7.5, 15.0, 30.0]
cfg_images = []

for cfg in cfg_values:
    print(f"Генерация для guidance_scale = {cfg}...")

    # Сбрасываем генератор на тот же seed для честного сравнения
    generator = torch.Generator(device=device).manual_seed(42)
    image = pipe(
        prompt=prompt, 
        num_inference_steps=30, 
        guidance_scale=cfg, 
        generator=generator
    ).images[0]
    cfg_images.append(image)

# Визуализация матрицы результатов (Grid)
fig, axes = plt.subplots(1, len(cfg_values), figsize=(20, 5))
for ax, img, cfg in zip(axes, cfg_images, cfg_values):
    ax.imshow(img)
    ax.set_title(f"CFG Scale: {cfg}")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

### Задание 3. Локальное редактирование (Inpainting) с маскированием.

**Инструкция:** возьмите одно из успешно сгенерированных изображений из Задания 1.

**Суть задачи:** напишите скрипт, который программно или интерактивно накладывает маску на определенный объект (например, на здание или небо). Используя StableDiffusionInpaintPipeline, замените замаскированную область новым объектом (например, "а flying steampunk airship"). Добейтесь бесшовного встраивания объекта, варьируя параметр strength (степень изменения исходного изображения под маской).

```python
# Переключаем пайплайн на инпейнтинг
inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Выбираем базовую картинку (можно взять первую сгенерированную из Задания 1)
init_image = images[2].resize((512, 512))

# Программное создание бинарной маски (квадрат по центру кадра)
# Студенты могут написать интерактивный инструмент или использовать более сложную маску
mask_np = np.zeros((512, 512), dtype=np.uint8)
mask_np[150:350, 150:350] = 255 # Закрашиваем белым область для замены
mask_image = Image.fromarray(mask_np)

inpaint_prompt = "A flying steampunk airship, detailed, realistic"

# Эксперимент с силой изменения оригинального изображения (strength)
strength_values = [0.1, 0.5, 0.9]
inpainted_images = []

for str_val in strength_values:
    generator = torch.Generator(device=device).manual_seed(100)
    out_img = inpaint_pipe(
        prompt=inpaint_prompt,
        image=init_image,
        mask_image=mask_image,
        strength=str_val,
        num_inference_steps=30,
        generator=generator
    ).images[0]
    inpainted_images.append(out_img)

# Отрисовка результатов: Оригинал, Маска, Результаты при разных значениях strength
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
axes[0].imshow(init_image); axes[0].set_title("Original Image"); axes[0].axis("off")
axes[1].imshow(mask_image, cmap='gray'); axes[1].set_title("Mask"); axes[1].axis("off")
for i, str_val in enumerate(strength_values):
    axes[i+2].imshow(inpainted_images[i])
    axes[i+2].set_title(f"Strength: {str_val}")
    axes[i+2].axis("off")
plt.show()
```

### Задание 4. Управление геометрией через Контрольные сети (ControlNet).

**Инструкция:** подключите предобученный модуль ControlNet (например, Canny edge или OpenPose).

**Суть задачи:** выделите контуры (Canny edges) из любого реального изображения. Используя эти контуры в качестве жесткого геометрического условия, сгенерируйте абсолютно новое изображение в принципиально другом стиле (например, превратите фотографию комнаты в чертеж или в заросшие джунгли). Проанализируйте, насколько точно модель сохраняет пространственную композицию оригинала.

```python
import cv2

# Инициализируем модель ControlNet для детектирования границ Canny
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16 if device == "cuda" else torch.float32)
control_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1.5", controlnet=controlnet, torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Подготовка управляющего изображения (контуров)
# Студенты загружают любое свое изображение. Ниже — пример генерации контуров из заготовки Задания 1
original_np = np.array(init_image)
low_threshold = 100
high_threshold = 200
canny_edges = cv2.Canny(original_np, low_threshold, high_threshold)
canny_edges = np.concatenate([canny_edges[:, :, None]] * 3, axis=2) # Перевод в 3 канала
control_image = Image.fromarray(canny_edges)

# Новый стиль для старой геометрии
new_style_prompt = "A lush overgrown jungle ruins city, photorealistic, abandoned cyberpunk, moss and trees, 4k"

# Запуск генерации через ControlNet
controlnet_image = control_pipe(
    prompt=new_style_prompt,
    image=control_image,
    num_inference_steps=30,
    guidance_scale=9.0
).images[0]

# Отрисовка
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(init_image); axes[0].set_title("Original"); axes[0].axis("off")
axes[1].imshow(control_image); axes[1].set_title("Canny Map (Control)"); axes[1].axis("off")
axes[2].imshow(controlnet_image); axes[2].set_title("ControlNet Result"); axes[2].axis("off")
plt.show()
```

### Задание 5. Оптимизация промптов и негативный промпт (Продвинутый уровень).

**Инструкция:** реализуйте генерацию сложного антропоморфного объекта или анатомически точного персонажа.

**Суть задачи:** столкнитесь с проблемой генерации артефактов (лишние пальцы, размытые лица). Настройте систему генерации с использованием структуры позитивного и негативного промптов (negative_prompt). Экспериментально подберите ключевые слова (например, "bad anatomy, blurry, deformed hands"), чтобы минимизировать дефекты. Опишите, как текстовый энкодер CLIP обрабатывает отрицательные веса в пространстве латентов.

```python
# Позитивный промпт, заведомо провоцирующий дефекты анатомии без уточнения стилистики
bad_anatomy_prompt = "A close up portrait of a medieval blacksmith showing his hands, holding a hammer, detailed human hands"

# Генерация БЕЗ негативного промпта
gen_without_neg = pipe(
    prompt=bad_anatomy_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=torch.Generator(device=device).manual_seed(777)
).images[0]

# Настройка жесткого негативного промпта для исправления дефектов
negative_prompt = "deformed, bad anatomy, extra fingers, mutated hands, blurry, low quality, bad proportions, missing limbs"

# Генерация С негативным промптом
gen_with_neg = pipe(
    prompt=bad_anatomy_prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
    generator=torch.Generator(device=device).manual_seed(777) # Идентичный seed
).images[0]

# Сравнение результатов
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(gen_without_neg); axes[0].set_title("Without Negative Prompt"); axes[0].axis("off")
axes[1].imshow(gen_with_neg); axes[1].set_title("With Negative Prompt"); axes[1].axis("off")
plt.show()
```

## Контрольные вопросы

- В чем заключается принципиальное различие между прямой диффузией (Forward Process) и обратной диффузией (Reverse Process)?
- Зачем в модели Stable Diffusion используется латентное пространство (Latent Space) вместо пространства исходных пикселей? Какой блок архитектуры за это отвечает?
- Какую роль играет нейросеть U-Net на каждом шаге обратного процесса диффузии? Что именно она предсказывает?
- Как механизм перекрестного внимания (Cross-Attention) связывает текстовый промпт и генерируемые латентные признаки?
- Что такое Classifier-Free Guidance (CFG)? Как математически и логически изменяется генерация при выходе параметра guidance_scale за стандартные рамки?
- В чем различие между планировщиками (Schedulers/Samplers), такими как DDIM, Euler a, и DPM++? Как они влияют на скорость сходимости?

## Структура отчета и критерии оценивания

### Структура отчета

- Титульный лист (вуз, кафедра, дисциплина, тема, ФИО студента, группа, дата).
- Цели и задачи лабораторной работы.
- Описание архитектуры модели (краткая теоретическая справка о выбранной версии Stable Diffusion, используемых текстовых энкодерах и U-Net).
- Ход выполнения заданий:
  - Листинги ключевых фрагментов кода с комментариями.
  - Сгенерированные матрицы изображений (Grids) для каждого задания.
  - Графики/таблицы зависимостей (например, время генерации от количества шагов).
- Анализ результатов и выводы: детальное описание того, как гиперпараметры (CFG, Steps, Seed, Strength) влияют на визуальное качество.

### Критерии оценивания

Итоговая оценка за лабораторную работу формируется на основе процентного соотношения набранных баллов (из возможных за работу) и переводится в стандартные вербальные оценки.

#### 20%: настройка окружения и воспроизводимость

- 20% — Окружение развёрнуто без ошибок, устройство выбирается явно, seed фиксируется, а доступная память используется корректно. Для полного варианта применяется GPU; при его отсутствии использована заранее согласованная облегчённая модель или предоставленная среда.
- 10% — Окружение настроено, но сравнение невоспроизводимо либо выбранная конфигурация регулярно вызывает нехватку памяти.
- 0% — Программный код не запускается.

#### 30%: базовый уровень (Задания 1–2)

- 30% — Реализована базовая генерация. Построена корректная матрица изображений (Grid), наглядно демонстрирующая процесс денойзинга на разных шагах и влияние шкалы CFG. Дан аргументированный анализ артефактов перенасыщения.
- 15% — Задания выполнены частично. Изображения сгенерированы, но отсутствуют сравнительные матрицы или нарушена фиксация seed (из-за чего сравнение гиперпараметров некорректно).
- 0% — Базовые задания не выполнены.

#### 30%: продвинутый уровень (Задания 3–5)

- 30% — Успешно реализован инпейнтинг с варьированием параметра strength. Настроена геометрия через ControlNet (перенос стиля с сохранением контуров). Продемонстрирован эффект работы с негативным промптом.
- 15% — Выполнено только одно продвинутое задание, либо в работе ControlNet/Inpainting присутствуют грубые логические ошибки (например, маска накладывается со смещением).
- 0% — Задания повышенной сложности не затронуты.

Если ни одно из заданий 3–5 не выполнено на работоспособном уровне, итоговый результат за лабораторную работу не может превышать 69 баллов независимо от суммы по остальным критериям.

#### 20%: оформление отчета и защита работы

- 20% — Отчет полностью структурирован. Описана архитектура Stable Diffusion и роль ее блоков. Студент уверенно отвечает на контрольные вопросы, демонстрируя понимание математики диффузионных процессов.
- 10% — Отчет содержит только листинги кода без аналитических выводов. При ответе на контрольные вопросы студент испытывает затруднения.
- 0% — Отчет не предоставлен или полностью скопирован у другого студента.

### Перевод баллов в итоговую оценку

| Процент набранных баллов | Оценка | Требования к демонстрации результатов |
|---|---|---|
| от 90% до 100% | Отлично | Все задания выполнены, включая фактическую замену scheduler и воспроизводимое сравнение. Студент аргументированно анализирует результаты. |
| от 70% до 89% | Хорошо | Выполнены все базовые и большая часть усложнённых заданий; имеются незначительные неточности. |
| от 50% до 69% | Удовлетворительно | Выполнена базовая генерация и исследование основных параметров, но часть техник реализована неполно. |
| от 0% до 49% | Неудовлетворительно | Базовые задания не выполнены или результат не воспроизводится. |

## Рекомендуемая литература

- Ho J., Jain A., Abbeel P. Denoising Diffusion Probabilistic Models // Advances in Neural Information Processing Systems (NeurIPS). – 2020. [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)
- Rombach R., Blattmann A., Lorenz D. et al. High-Resolution Image Synthesis with Latent Diffusion Models // Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). – 2022. [https://arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752)
- Zhang L., Agrawala M. Adding Conditional Control to Text-to-Image Diffusion Models // International Conference on Computer Vision (ICCV). – 2023. [https://arxiv.org/abs/2302.05543](https://arxiv.org/abs/2302.05543) или [https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html)
- Документация Hugging Face Diffusers: [https://huggingface.co/docs/diffusers/en/index](https://huggingface.co/docs/diffusers/en/index) — практические руководства по оптимизации пайплайнов и работе с предобученными весами.
- Фостер, Д. Генеративное глубокое обучение : как не мы рисуем картины, пишем романы и музыку / Д. Фостер; перевод с английского Л. Киселева. — 2-е междун. изд. — Астана: Спринт Бук, 2024. — 448 с. — (Бестселлеры O'Reilly). — Перевод издания: Generative deep learning: teaching machines to Paint, write, compose, and play. — ISBN 978-601-08-3729-4.
- Song Y. et al. «Score-Based Generative Modeling through Stochastic Differential Equations» (ICLR, 2021). [https://arxiv.org/abs/2011.13456](https://arxiv.org/abs/2011.13456)
- SOTA-фреймворк для работы с диффузионными моделями: [https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers)
- Реализация классической модели DDPM (из статьи Ho et al., 2020) на PyTorch: [https://github.com/lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
- Структурированный репозиторий со ссылками на статьи по диффузии в CV, 3D, аудио, медицине, а также со ссылками на туториалы и Jupyter Ноутбуки: [https://github.com/diff-usion/Awesome-Diffusion-Models](https://github.com/diff-usion/Awesome-Diffusion-Models)
- Репозиторий, сфокусированный на методах оптимизации, ускорения сэмплирования и эффективного обучения диффузионных сетей: [https://github.com/TsinghuaC3I/Efficient-Diffusion-Models](https://github.com/TsinghuaC3I/Efficient-Diffusion-Models)
- Браузерный интерфейс для локального запуска Stable Diffusion: [https://github.com/AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- Интерфейс, где пайплайн генерации собирается в виде графа из узлов (Nodes). Дает наглядное визуальное представление о том, как латенты переходят из VAE в U-Net, как подмешивается шум и работают сэмплеры: [https://github.com/Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI)
- Hugging Face Diffusion Models Course (Официальный интерактивный курс): [https://huggingface.co/learn/diffusion-course/unit0/1](https://huggingface.co/learn/diffusion-course/unit0/1)
- Раздел «Диффузионные модели» в Учебнике по ML от Яндекса: [https://education.yandex.ru/handbook/ml/article/diffuzionnye-modeli](https://education.yandex.ru/handbook/ml/article/diffuzionnye-modeli)
