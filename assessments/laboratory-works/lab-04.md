# МЕТОДИЧЕСКИЕ УКАЗАНИЯ ПО ВЫПОЛНЕНИЮ ЛАБОРАТОРНОЙ РАБОТЫ

**Дисциплина:** Методы компьютерного зрения.

**Тема:** реализация пайплайна дообучения (fine-tuning) предобученной модели для решения задачи классификации на пользовательском наборе данных. Эксперименты с разными стратегиями Transfer Learning (заморозка слоев, изменение последнего слоя).

**Курс:** 3 курс, 6 семестр

**Трудоемкость:** 4 академических часа

## 1. ВВЕДЕНИЕ

**Цели работы:**

- освоить методологию Transfer Learning и реализовать полный пайплайн дообучения предобученных моделей для решения задачи классификации на пользовательском наборе данных;
- экспериментально исследовать различные стратегии переноса обучения (заморозка слоев, изменение классификатора, fine-tuning) и оценить их эффективность;
- освоить выбор и использование инструментов разработки на Python, приемлемых для создания прикладной системы обработки научных данных, машинного обучения и визуализации с заданными требованиями (PL-1.2, средний);
- освоить применение и экспериментальное сравнение известных алгоритмов, библиотек и предобученных моделей компьютерного зрения с дообучением и валидацией на собственных данных (`DL-3.1`, средний);
- научиться изменять классификационную часть CNN и разрабатывать решение на основе сложной конфигурации свёрточной сети (`DL-1.4`, средний).

**Задачи работы:**

- Изучить методологию Transfer Learning и различные стратегии дообучения моделей.
- Научиться загружать и модифицировать предобученные модели (замена классификатора, заморозка слоев).
- Освоить техники аугментации данных для повышения обобщающей способности модели.
- Реализовать полноценный пайплайн обучения с использованием PyTorch/TensorFlow.
- Провести сравнительный анализ различных стратегий Transfer Learning.
- Оценить влияние гиперпараметров на качество дообучения.
- Проанализировать и визуализировать результаты обучения.

**Необходимое ПО:**

- Python 3.8+
- Jupyter Notebook / Google Colab
- Библиотеки: `torch`, `torchvision`, `matplotlib`, `numpy`, `Pillow`, `scikit-learn`, `tqdm` (отображение прогресса)

## 2. ТЕОРЕТИЧЕСКАЯ ЧАСТЬ

### 2.1. Transfer Learning: методология и стратегии

Transfer Learning (перенос обучения) — это метод машинного обучения, при котором знания, полученные при решении одной задачи, используются для решения другой, связанной задачи. В контексте компьютерного зрения это означает использование моделей, предварительно обученных на больших наборах данных (ImageNet, COCO), для решения задач на меньших, специализированных датасетах.

**Основные стратегии Transfer Learning:**

1. **Извлечение признаков (Feature Extraction):**
   - Принцип работы: предобученная модель используется как фиксированный экстрактор признаков. Все слои модели замораживаются, и на выходе добавляется новый классификатор, который обучается с нуля.
   - Условия применения: применяется в случае, если целевой датасет достаточно мал (сотни изображений) и подобен исходному датасету ImageNet.
   - Преимущества: быстрое обучение, требуется немного данных для обучения, низкий риск переобучения.
   - Недостатки: ограниченная адаптивность к новым доменам.
2. **Дообучение (Fine-Tuning):**
   - Принцип работы: вся модель или часть ее слоев "размораживается" и дообучается на новых данных. Обычно начинают с низкой скорости обучения.
   - Условия применения: в случае, если целевой датасет достаточно большой (тысячи и более изображений) или данные значительно отличаются от ImageNet.
   - Преимущества: высокая адаптивность к новым данным, часто дает лучшую точность.
   - Недостатки: требует больше данных, выше риск переобучения, дольше обучается.
3. **Прогрессивное размораживание (Progressive Unfreezing):**
   - Принцип работы: постепенное размораживание слоев модели от последних к первым. Сначала обучается только новый классификатор, затем размораживается последний блок, потом предыдущий и т.д.
   - Условия применения: применяется в случае необходимости достижения максимальной точности.
   - Преимущества: баланс между адаптивностью и предотвращением катастрофического забывания.
   - Недостатки: риск переобучения.
4. **Parameter-Efficient Fine-Tuning (PEFT):**
   - Принцип работы: вводятся небольшие дополнительные параметры (адаптеры, LoRA), которые обучаются, в то время как основные веса модели остаются замороженными.
   - Условия применения: при работе с большими языковыми моделями (LLM, большие ViT). В этом случае полное дообучение требует огромных ресурсов.
   - Преимущества: минимальные вычислительные затраты, эффективно для больших языковых моделей.

### 2.2. Подготовка данных и аугментация

**Предобработка данных:**

- Ресайз (Resize): приведение изображений к единому размеру (например, 224x224 для ResNet).
- Нормализация: нормализация пикселей с использованием средних и стандартных отклонений из ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

**Аугментация данных:**

- Случайные повороты (RandomRotation).
- Случайные отражения (RandomHorizontalFlip).
- Случайное изменение цвета, контраста, яркости (ColorJitter).
- Случайное изменение размера и кропа (RandomResizedCrop).
- Применение аугментаций повышает обобщающую способность модели и помогает избежать переобучения.

### 2.3. Гиперпараметры дообучения

**Ключевые гиперпараметры:**

- Скорость обучения (Learning Rate): критически важный параметр. При дообучении обычно используется значительно меньшая скорость обучения (`lr=1e-4` или `lr=1e-5`), чем при обучении с нуля.
- Оптимизатор: Adam, SGD с моментом (SGD with momentum), AdamW.
- Размер батча (Batch Size): зависит от размера модели и доступной памяти GPU.
- Количество эпох (Number of Epochs): определяется по валидационной метрике (ранняя остановка).
- Регуляризация: Dropout, Weight Decay (L2 регуляризация).

**Планировщик скорости обучения (Learning Rate Scheduler):**

- StepLR: уменьшение скорости обучения через определенное количество эпох.
- ReduceLROnPlateau: уменьшение скорости обучения при остановке улучшения валидационной метрики.
- CosineAnnealingLR: плавное уменьшение скорости обучения по косинусоидальному закону.

## 3. ПОРЯДОК ВЫПОЛНЕНИЯ РАБОТЫ

### 3.1. Подготовка рабочего окружения

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import os
from tqdm import tqdm
import copy
import time

# Проверка наличия GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используемое устройство: {device}")

# Установка seed для воспроизводимости
torch.manual_seed(42)
np.random.seed(42)
```

### 3.2. Загрузка и подготовка пользовательского датасета

#### Задание 1: Подготовка структуры данных

- Создайте структуру папок для пользовательского датасета:

```text
data/
├── train/
│   ├── class_1/
│   │   ├── img1.jpg
│   │   └── ...
│   └── class_2/
│       ├── img1.jpg
│       └── ...
└── test/
    ├── class_1/
    │   └── ...
    └── class_2/
        └── ...
```

- Определите классы вашего датасета и количество классов.

#### Задание 2: Определение трансформаций для данных

- Определите трансформации для тренировочной выборки (с аугментацией).
- Определите трансформации для валидационной и тестовой выборок (без аугментации).

```python
# Трансформации для тренировочных данных (с аугментацией)
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Трансформации для валидационных/тестовых данных
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

#### Задание 3: Загрузка данных

- Загрузите данные с помощью `torchvision.datasets.ImageFolder`.
- Разделите тренировочный датасет на train и validation (80/20 или 70/30).
- Создайте DataLoader-ы для пакетной загрузки.

```python
def load_datasets(data_dir, batch_size=32):
   #    Загрузка и подготовка датасетов.
    # Загрузка тренировочного датасета
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'), 
        transform=train_transforms
    )
    
    # Загрузка тестового датасета
    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'test'), 
        transform=val_transforms
    )
    
    # Разделение train на train/validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Создание DataLoader-ов
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Получение имен классов
    class_names = train_dataset.dataset.classes
    
    return train_loader, val_loader, test_loader, class_names
```

### 3.3. Реализация стратегий Transfer Learning

#### Задание 4: Функция обучения модели

Создайте универсальную функцию для обучения модели, которая будет принимать конфигурацию обучения и возвращать историю обучения.

```python
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, epochs=25, device='cuda'):
#    Универсальная функция обучения модели.
    model.to(device)
    
    # История обучения
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'best_val_acc': 0
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 30)
        
        # Обучение
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = 100 * train_correct / train_total
        
        # Валидация
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
        
        # Обновление истории
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            history['best_val_acc'] = best_val_acc
        
        # Шаг планировщика
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print()
    
    # Загрузка лучшей модели
    model.load_state_dict(best_model_wts)
    
    return model, history
```

#### Задание 5: Стратегия 1 - Извлечение признаков (Feature Extraction)

- Загрузите предобученную модель (например, ResNet50 или EfficientNet-B0).
- Заморозьте все слои модели.
- Замените последний полносвязный слой на новый классификатор.
- Обучите только новый классификатор.

```python
def create_feature_extraction_model(num_classes, model_name='resnet50'):
#    Создание модели для извлечения признаков.
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        # Заморозка всех слоев
        for param in model.parameters():
            param.requires_grad = False
        # Замена классификатора
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model
```

#### Задание 6: Стратегия 2 - Дообучение (Fine-Tuning)

- Загрузите предобученную модель.
- Замените последний классификатор (не замораживая остальные слои).
- Настройте разные скорости обучения для классификатора и остальных слоев.
- Обучите всю модель с низкой скоростью обучения.

```python
def create_finetuning_model(num_classes, model_name='resnet50'):
#    Создание модели для дообучения.
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        # Замена классификатора
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return model
```

#### Задание 7: Стратегия 3 - Прогрессивное размораживание

- Начните с замороженной модели и тренируйте только новый классификатор (как в Feature Extraction).
- Через несколько эпох разморозьте последний блок модели.
- Продолжите обучение с более низкой скоростью обучения.
- Постепенно размораживайте все больше слоев.

```python
def progressive_unfreezing(model, stages):
#    Реализация прогрессивного размораживания.
    # stage 1: только классификатор
    # stage 2: разморозить последний блок
    # stage 3: разморозить все
    pass
```

### 3.4. Сравнительный анализ стратегий

#### Задание 8: Проведение экспериментов

- Для каждой стратегии (Feature Extraction, Fine-Tuning):
  - Обучите модель с различными гиперпараметрами.
  - Записывайте историю обучения и итоговые метрики.
  - Сохраняйте лучшие модели.
- Создайте таблицу сравнения стратегий с указанием:
  - Время обучения.
  - Точность на валидации.
  - Точность на тесте.
  - Количество обучаемых параметров.

```python
def evaluate_model(model, test_loader, class_names, device='cuda'):
#    Оценка модели на тестовом наборе.
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Метрики
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, report, cm
```

#### Задание 9: Визуализация результатов

- Постройте графики кривых обучения (потери и точность) для разных стратегий.
- Визуализируйте матрицы ошибок.
- Отобразите примеры изображений с правильными и неправильными предсказаниями.

```python
def plot_training_history(histories, title='Training History'):
#    Визуализация истории обучения для разных стратегий.
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Потери
    for name, history in histories.items():
        axes[0].plot(history['train_loss'], label=f'{name} - Train')
        axes[0].plot(history['val_loss'], '--', label=f'{name} - Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    # Точность
    for name, history in histories.items():
        axes[1].plot(history['train_acc'], label=f'{name} - Train')
        axes[1].plot(history['val_acc'], '--', label=f'{name} - Val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
```

### 3.5. Исследование влияния гиперпараметров

#### Задание 10: Эксперименты с гиперпараметрами

- Скорость обучения: Протестируйте разные значения (1e-2, 1e-3, 1e-4, 1e-5).
- Оптимизаторы: Сравните Adam и SGD с моментом.
- Регуляризация: Протестируйте различные значения Weight Decay.
- Размер батча: Сравните батчи размером 16, 32, 64.

```python
def hyperparameter_experiment():
#    Проведение экспериментов с гиперпараметрами.
    results = []
    
    # Конфигурации для экспериментов
    configs = [
        {'lr': 1e-3, 'optimizer': 'adam', 'wd': 1e-4, 'name': 'Adam_1e-3'},
        {'lr': 1e-4, 'optimizer': 'adam', 'wd': 1e-4, 'name': 'Adam_1e-4'},
        {'lr': 1e-5, 'optimizer': 'adam', 'wd': 1e-4, 'name': 'Adam_1e-5'},
        {'lr': 1e-3, 'optimizer': 'sgd', 'wd': 1e-4, 'momentum': 0.9, 'name': 'SGD_1e-3'},
    ]
    
    for config in configs:
        print(f"Testing configuration: {config['name']}")
        # Обучение модели с данной конфигурацией
        # Сохранение результатов
        # Очистка памяти
        pass
    
    return results
```

### 3.6. Вариативная часть (по выбору студента)

#### Вариант А: Использование Learning Rate Scheduler

Реализуйте различные планировщики скорости обучения (StepLR, ReduceLROnPlateau, CosineAnnealingLR) и сравните их влияние на процесс обучения.

#### Вариант Б: Работа с дисбалансом классов

Модифицируйте датасет или функцию потерь для работы с несбалансированными классами:

- Используйте `WeightedRandomSampler` для балансировки выборки.
- Используйте взвешенную функцию потерь `nn.CrossEntropyLoss(weight=class_weights)`.

#### Вариант В: PEFT с использованием LoRA

Реализуйте Parameter-Efficient Fine-Tuning с использованием библиотеки `peft`:

```python
from peft import LoraConfig, get_peft_model, TaskType

def create_lora_model(model, rank=4):
#    Применение LoRA к модели.
    lora_config = LoraConfig(
        task_type=TaskType.IMAGE_CLASSIFICATION,
        r=rank,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q", "v"]  # Для ViT/трансформеров
    )
    model = get_peft_model(model, lora_config)
    return model
```

## 4. ОТЧЕТ И КРИТЕРИИ ОЦЕНКИ

### 4.1. Структура отчета

Отчет должен быть оформлен в виде Jupyter Notebook и содержать:

- Титульная страница: Название работы, дисциплина, группа, ФИО студента.
- Введение: Цель и задачи работы.
- Ход работы:
  - Подготовка данных (трансформации, аугментация, DataLoader-ы).
  - Реализация стратегий Transfer Learning (Feature Extraction, Fine-Tuning, Progressive Unfreezing).
  - Графики кривых обучения для каждой стратегии.
  - Сравнительная таблица результатов (время обучения, точность, количество параметров).
  - Матрицы ошибок и примеры предсказаний.
  - Исследование влияния гиперпараметров.
- Выводы: Сравнительный анализ стратегий, рекомендации по выбору стратегии для разных задач, выводы о влиянии гиперпараметров.

### 4.2. Критерии оценки

- "Отлично" (90-100%): Все стратегии реализованы корректно. Проведен детальный сравнительный анализ. Выполнены дополнительные эксперименты с гиперпараметрами. Выводы глубокие и обоснованные.
- "Хорошо" (70-89%): Основные стратегии (Feature Extraction и Fine-Tuning) реализованы. Анализ результатов присутствует. Выводы сформулированы.
- "Удовлетворительно" (50-69%): Задания выполнены частично. Код работает с ошибками.
- "Неудовлетворительно" (0-49%): Задания не выполнены.

## 5. КОНТРОЛЬНЫЕ ВОПРОСЫ

- Что такое Transfer Learning и в каких случаях он применяется?
- В чем разница между Feature Extraction и Fine-Tuning?
- Почему при Fine-Tuning используется низкая скорость обучения?
- Что такое "катастрофическое забывание" (catastrophic forgetting) и как его избежать?
- Как выбрать, какие слои замораживать, а какие дообучать?
- Что такое Progressive Unfreezing и в чем его преимущества?
- Какие методы аугментации данных наиболее эффективны для вашей задачи?
- Какой планировщик скорости обучения лучше всего подходит для Fine-Tuning?
- Как оценить, достаточно ли данных для эффективного Fine-Tuning?
- Что такое PEFT (Parameter-Efficient Fine-Tuning) и когда его используют?

## 6. РЕКОМЕНДУЕМАЯ ЛИТЕРАТУРА

- PyTorch Transfer Learning Tutorial: [https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
- Tan, M., & Le, Q. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks. In International conference on machine learning (pp. 6105-6114).
- Howard, A. G., et al. (2017). Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861.
- Hu, E. J., et al. (2021). LoRA: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685
- Шолле, Ф. (2022). Глубокое обучение на Python. (Разделы про перенос обучения / Transfer Learning). [https://github.com/fchollet/deep-learning-with-python-notebooks](https://github.com/fchollet/deep-learning-with-python-notebooks)
- Transfer learning & fine-tuning. [https://keras.io/guides/transfer_learning/](https://keras.io/guides/transfer_learning/)
- Стивенс, Э., Антига, Л. (2022). Глубокое обучение на PyTorch.
- LoRA: Low-Rank Adaptation of Large Language Models. [https://iclr.cc/virtual/2022/poster/6319](https://iclr.cc/virtual/2022/poster/6319)
- LoRA: Low-Rank Adaptation of Large Language Models. [https://github.com/microsoft/LoRA](https://github.com/microsoft/LoRA)
- Xiang Lisa Li, Percy Liang (2021). Prefix-Tuning: Optimizing Continuous Prompts for Generation. [https://arxiv.org/abs/2101.00190](https://arxiv.org/abs/2101.00190)
- Hu, E. J., et al. (2021). LoRA: Low-rank adaptation of large language models. [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
- Meng, Zi-An, et al. (2024/2025). DoRA: Weight-Decomposed Low-Rank Adaptation. [https://arxiv.org/abs/2402.09353](https://arxiv.org/abs/2402.09353)
