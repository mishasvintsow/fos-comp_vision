<!-- Источник: Методические указания к лабораторной работе 2 с компетенциями.docx -->

# МЕТОДИЧЕСКИЕ УКАЗАНИЯ ПО ВЫПОЛНЕНИЮ ЛАБОРАТОРНОЙ РАБОТЫ

**Дисциплина:** Методы компьютерного зрения

**Тема:** реализация «с нуля» простого сверточного слоя и слоя пулинга на Python (NumPy) для глубокого понимания их математической сути. Обучение простого классификатора на наборе данных (например, MNIST) с использованием библиотеки PyTorch/TensorFlow.

**Курс:** 3 курс, 6 семестр

**Трудоемкость:** 4 академических часа

## 1. ВВЕДЕНИЕ

**Цели работы:**

- освоить математические принципы работы сверточных нейронных сетей (CNN): через реализацию базовых компонентов (сверточный слой, слой пулинга) с использованием библиотеки NumPy, без применения готовых высокоуровневых функций глубокого обучения;
- освоить навыки работы с фреймворками глубокого обучения (PyTorch, TensorFlow) путем обучения простого классификатора на наборе данных MNIST;
- освоить выбор и использование инструментов разработки на Python, приемлемых для создания прикладной системы обработки научных данных, машинного обучения и визуализации с заданными требованиями (`PL-1.2`, базовый);
- научиться выбирать параметры свёртки и пулинга и применять готовую архитектуру CNN (`DL-1.4`, базовый).

**Задачи работы:**

- Реализовать "с нуля" операцию двумерной свертки с использованием библиотеки NumPy.
- Реализовать "с нуля" операцию пулинга (максимальный и средний) с использованием NumPy.
- Сравнить результаты собственных реализаций с встроенными функциями библиотек глубокого обучения.
- Изучить архитектуру и принципы обучения нейронных сетей на примере классификатора MNIST.
- Построить и обучить простую сверточную нейронную сеть с использованием PyTorch/TensorFlow.

**Необходимое ПО:**

- Python 3.8+
- Jupyter Notebook / Google Colab
- Библиотеки: `numpy`, `matplotlib`, `torch` или `tensorflow`, `torchvision` или `tensorflow-datasets`.

## 2. ТЕОРЕТИЧЕСКАЯ ЧАСТЬ

### 2.1. Математические основы сверточного слоя

Сверточный слой является ключевым компонентом CNN и выполняет операцию кросс-корреляции (часто называемую сверткой) между входным тензором и набором обучаемых фильтров (ядер).

Математическая формула для двумерной свертки:

Для входного изображения \(I\) размера \(H×W\) и ядра \(K\) размера \(h × w\), результат свертки \(S\) (карта признаков) вычисляется следующим образом:

\[
S(i,j)=(I∙K)(i,j)=\sum_{m=0}^{h-1}\sum_{n=0}^{w-1}I(i+m,j+n)∙K(m,n)
\]

**Ключевые параметры сверточного слоя:**

- Ядро свертки (Kernel/Filter): матрица весов, которая "скользит" по входному изображению. Размер ядра в большинстве случаев равен 3x3 или 5x5.
- Шаг (Stride): количество пикселей, на которое сдвигается ядро при переходе к следующей позиции. Шаг 1 означает, что ядро сдвигается на 1 пиксель, что дает выходную карту признаков того же размера (с учетом отступов). Шаг 2 уменьшает размер выходной карты вдвое.
- Отступ (Padding): добавление нулевых пикселей по краям входного изображения. Отступ позволяет контролировать размер выходной карты признаков. Так, `padding='same'` сохраняет пространственные размеры, а `padding='valid'` не добавляет отступы, уменьшая размер.

Размер выходной карты признаков вычисляется по формуле:

\[
W_{out}=\left[\frac{W_{in}-K+2P}{S}\right]+1
\]

где \(W_{out}\) — размер выходной карты, \(W_{in}\) — размер входной карты, \(K\) — размер ядра, \(P\) — размер отступа (padding), \(S\) — шаг (stride).

### 2.2. Слой пулинга (Pooling Layer)

Слой пулинга используется для уменьшения пространственных размеров карт признаков, что снижает вычислительную сложность и обеспечивает некоторую инвариантность к малым смещениям.

**Максимальный пулинг (Max Pooling):**

Для входной карты признаков \(F\) и окна пулинга размера \(h×w\), выходное значение пулинга \(P(i,j)\) вычисляется следующим образом:

[image1.png](attachments/lab-02/image1.png)

\[
P(i,j)=\max_{\substack{0\le m<h\\0\le n<w}}F(iS+m,jS+n),
\]

где \(S\) — шаг пулинга.

**Средний пулинг (Average Pooling):**

В этом случае вместо максимума берется среднее арифметическое значение в окне:

\[
P(i,j)=\frac{1}{h∙w}\sum_{m=0}^{h-1}\sum_{n=0}^{w-1}F(i∙S+m,j∙S+n)
\]

### 2.3. Набор данных MNIST

MNIST — это эталонная база данных из 70 000 изображений рукописных цифр (от 0 до 9).

- Количество изображений: 60000 для обучения, 10000 для тестирования.
- Размер изображения: 28x28 пикселей, градации серого.
- Количество классов: 10 (цифры 0-9).

## 3. ПОРЯДОК ВЫПОЛНЕНИЯ РАБОТЫ

### 3.1. Подготовка рабочего окружения

Установите необходимые библиотеки и импортируйте их.

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report
import time

# Проверка наличия GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")
```

### 3.2. Реализация сверточного слоя "с нуля" на NumPy

#### Задание 1: Реализация операции свертки

Реализуйте функцию `conv2d_numpy`, которая выполняет двумерную свертку без использования встроенных функций глубокого обучения.

**Требования к функции:**

**Входные параметры:**

- `input_image` — двумерный массив NumPy (H, W) или трехмерный (H, W, C) для нескольких каналов.
- `kernel` — двумерный массив NumPy (h, w).
- `stride` — шаг свертки (по умолчанию 1).
- `padding` — размер отступа (по умолчанию 0).

**Выходные данные:** двумерный массив NumPy — карта признаков после свертки.

**Методические указания:**

- Добавьте отступы к входному изображению с помощью `np.pad()`.
- Вычислите размеры выходной карты признаков.
- Организуйте вложенные циклы для прохода по изображению и применения ядра.
- Для многоканального входа выполните свертку для каждого канала и суммируйте результаты.

```python
def conv2d_numpy (image, kernel, stride=1, padding=0):
# Реализация двумерной свертки с помощью NumPy.
    
    Args:
        image: входное изображение (H, W) или (H, W, C)
        kernel: ядро свертки (h, w)
        stride: шаг свертки
        padding: размер отступа
    
    Returns:
        output: карта признаков после свертки
   
    # Добавление отступов
    if len(image.shape) == 2:
        H, W = image.shape
        C = 1
        image = image.reshape(H, W, 1)
    else:
        H, W, C = image.shape
    
    h, w = kernel.shape
    
    # Добавление отступов
    padded_image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    
    # Вычисление размеров выходной карты
    H_out = ((H + 2*padding - h) // stride) + 1
    W_out = ((W + 2*padding - w) // stride) + 1
    
    # Инициализация выходной карты
    output = np.zeros((H_out, W_out))
    
    # Выполнение свертки
    for i in range(0, H_out):
        for j in range(0, W_out):
            # Извлечение области изображения
            h_start = i * stride
            h_end = h_start + h
            w_start = j * stride
            w_end = w_start + w
            
            region = padded_image[h_start:h_end, w_start:w_end, :]
            
            # Свертка: сумма по элементам (включая все каналы)
            output[i, j] = np.sum(region * kernel)
    
    return output
```

#### Задание 2: Реализация слоя пулинга

Реализуйте функции `max_pool2d_numpy` и `avg_pool2d_numpy`.

**Требования к функциям:**

- Входные параметры:
  - `input_image` — двумерный массив NumPy (H, W).
  - `pool_size` — размер окна пулинга (обычно 2).
  - `stride` — шаг пулинга (по умолчанию равен `pool_size`).
- Выходные данные: двумерный массив NumPy после пулинга.

```python
def max_pool2d_numpy (image, pool_size=2, stride=None):

#Реализация максимального пулинга с помощью NumPy.
       if stride is None:
        stride = pool_size
    
    H, W = image.shape
    h, w = pool_size, pool_size
    
    H_out = ((H - h) // stride) + 1
    W_out = ((W - w) // stride) + 1
    
    output = np.zeros((H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + h
            w_start = j * stride
            w_end = w_start + w
            
            region = image [h_start:h_end, w_start:w_end]
            output [i, j] = np.max(region)
    
    return output

def avg_pool2d_numpy (image, pool_size=2, stride=None):
   
   # Реализация среднего пулинга с помощью NumPy.
    if stride is None:
        stride = pool_size
    
    H, W = image.shape
    h, w = pool_size, pool_size
    
    H_out = ((H - h) // stride) + 1
    W_out = ((W - w) // stride) + 1
    
    output = np.zeros((H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + h
            w_start = j * stride
            w_end = w_start + w
            
            region = image [h_start:h_end, w_start:w_end]
            output [i, j] = np.mean(region)
    
    return output
```

#### Задание 3: Тестирование и сравнение реализаций

- Создайте тестовое изображение с помощью `np.random.rand(10, 10)`.
- Создайте тестовое ядро: `kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])`.
- Примените вашу функцию `conv2d_numpy` и сравните результат с `scipy.signal.convolve2d` или `torch.nn.functional.conv2d` (предварительно преобразовав данные).
- Проведите аналогичное сравнение для функций пулинга.
- Визуализируйте результаты, демонстрируя корректность ваших реализаций.
- Измерьте и сравните время выполнения ваших реализаций и встроенных функций. Сделайте выводы о производительности.

### 3.3. Обучение простого классификатора на MNIST с использованием PyTorch

#### Задание 4: Загрузка и подготовка данных MNIST

- Загрузите набор данных MNIST с помощью `torchvision.datasets.MNIST`.
- Примените необходимые трансформации: преобразование в тензор, нормализация.
- Разделите данные на тренировочный и валидационный наборы.
- Создайте DataLoader-ы для пакетной загрузки.

```python
# Загрузка данных MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Разделение на train/validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
```

#### Задание 5: Построение простой сверточной нейронной сети

Создайте класс `SimpleCNN` с архитектурой, включающей:

- Сверточный слой (входной канал 1, выходной 32, ядро 3x3).
- Слой ReLU.
- Слой пулинга (MaxPool2d, размер 2x2).
- Еще один сверточный слой (32 -> 64, ядро 3x3).
- ReLU.
- Еще один слой пулинга (2x2).
- Полносвязный слой (входной размер 64 \* 7 \* 7, выходной 128).
- ReLU.
- Выходной полносвязный слой (128 -> 10).

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

#### Задание 6: Обучение модели

- Инициализируйте модель, функцию потерь (CrossEntropyLoss) и оптимизатор (например, Adam).
- Реализуйте цикл обучения на нескольких эпохах (5-10 эпох).
- Для каждой эпохи выводите значения функции потерь и точности на тренировочном и валидационном наборах.
- Визуализируйте кривые обучения (графики потерь и точности).

```python
def train_model(model, train_loader, val_loader, epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        # Обучение
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_losses, train_accs, val_accs
```

#### Задание 7: Оценка модели на тестовом наборе

- Выполните инференс на тестовом наборе данных.
- Рассчитайте точность (accuracy) и другие метрики (precision, recall, f1-score).
- Постройте матрицу ошибок (confusion matrix) для визуализации результатов.
- Визуализируйте несколько примеров с правильными и неправильными предсказаниями.

```python
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Test Accuracy: {accuracy:.4f}')
    print(classification_report(all_labels, all_preds))
    
    return all_preds, all_labels
```

### 3.4. Вариативная часть (по выбору студента)

#### Вариант А: Сравнение с реализацией на TensorFlow/Keras

Реализуйте аналогичную модель на TensorFlow/Keras и сравните результаты обучения (скорость, точность). Проанализируйте различия в синтаксисе и подходах.

#### Вариант Б: Исследование влияния архитектуры на качество

Проведите эксперименты с различными архитектурами (добавление слоев, изменение количества фильтров, использование BatchNorm, Dropout). Сравните результаты и сделайте выводы.

#### Вариант В: Визуализация карт признаков

Визуализируйте карты признаков, полученные после первого и второго сверточных слоев для нескольких тестовых изображений. Проанализируйте, какие признаки изучает каждый слой.

## 4. ОТЧЕТ И КРИТЕРИИ ОЦЕНКИ

### 4.1. Структура отчета

Отчет должен быть оформлен в виде Jupyter Notebook и содержать:

- Титульная страница: Название работы, дисциплина, группа, ФИО студента.
- Введение: Цель и задачи работы.
- Часть 1: Реализация свертки и пулинга "с нуля"
  - Код функций `conv2d_numpy`, `max_pool2d_numpy`, `avg_pool2d_numpy`.
  - Тестирование функций на тестовых данных.
  - Сравнение с встроенными функциями (визуализация результатов, сравнение времени выполнения).
  - Выводы.
- Часть 2: Обучение классификатора на MNIST
  - Архитектура модели `SimpleCNN`.
  - Графики кривых обучения (потери и точность на train/val).
  - Результаты на тестовом наборе (метрики, матрица ошибок).
  - Визуализация примеров предсказаний.
  - Выводы.
- Вариативная часть: (если выполнена) описание эксперимента и его результаты.
- Общие выводы: Что было изучено, какие навыки приобретены. Сопоставление теоретических знаний с практической реализацией.

### 4.2. Критерии оценки

Оценка за лабораторную работу выставляется по следующим критериям:

- "Отлично" (90-100%): Все задания выполнены в полном объеме. Реализации на NumPy корректны и эффективны. Проведено глубокое сравнение со встроенными функциями. Модель на MNIST обучена с высокой точностью (>98%). Выполнена вариативная часть. Выводы глубокие и обоснованные.
- "Хорошо" (70-89%): Основные задания выполнены. Реализации на NumPy работают корректно. Модель обучена с хорошей точностью. Анализ результатов присутствует, но может быть неполным.
- "Удовлетворительно" (50-69%): Задания выполнены частично. Реализации работают с ошибками. Модель обучена с низкой точностью. Анализ результатов минимален.
- "Неудовлетворительно" (0-49%): Задания не выполнены.

## 5. КОНТРОЛЬНЫЕ ВОПРОСЫ

- Что такое операция свертки в контексте CNN? Напишите ее математическую формулу.
- Какие параметры определяют размер выходной карты признаков? Как они влияют?
- В чем отличие операции свертки от кросс-корреляции?
- Для чего используется слой пулинга? В чем разница между Max Pooling и Average Pooling?
- Почему важно реализовать свертку и пулинг "с нуля" на NumPy, если есть готовые реализации в PyTorch/TensorFlow?
- Как работает обратное распространение ошибки через сверточный слой?
- Какой размер карт признаков после первого сверточного слоя (32 фильтра, ядро 3x3) для изображения 28x28? (с учетом padding=1)
- Что такое ReLU и почему он используется после сверточных слоев?
- Какой размер входного вектора для полносвязного слоя после двух слоев MaxPooling (2x2) для изображения 28x28?
- Какие методы регуляризации (Dropout, Batch Normalization) можно применить для улучшения качества модели и почему они работают?

## 6. РЕКОМЕНДУЕМАЯ ЛИТЕРАТУРА

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. (Chapter 9). [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
- PyTorch Documentation: [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)
- NumPy Documentation: [https://numpy.org/doc/stable/](https://numpy.org/doc/stable/)
- Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
- A technical report on convolution arithmetic in the context of deep learning. [https://github.com/vdumoulin/conv_arithmetic?tab=readme-ov-file](https://github.com/vdumoulin/conv_arithmetic?tab=readme-ov-file)
- Шолле, Ф. (2022). Глубокое обучение на Python. (Глава 8: Введение в компьютерное зрение).
- Стивенс, Э., Антига, Л., Фиман, Т. (2022). Глубокое обучение на PyTorch. (Глава 8: Использование сверток для распознавания изображений).
- CNN Explainer. [https://poloclub.github.io/cnn-explainer/](https://poloclub.github.io/cnn-explainer/)
