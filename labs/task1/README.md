## Описание
Программа вычисляет сумму значений синуса для массива из 10⁷ элементов. Поддерживаются два типа данных: float  и double. Сборка происходит через CMake

## Сборка
```bash
# Сборка на FLOAT
cmake -DUSE_DOUBLE=OFF -S . -B build

# Сборка на DOUBLE
cmake -DUSE_DOUBLE=ON -S . -B build
cmake --build build
./build/task1
```

# Результаты

##### Float
```text
Float is using.
Sum = 0.349212
```

##### Double
```text
Double is using!
Sum = -6.76917e-10
```