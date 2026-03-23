## Сборка
#### 1 задача
```bash
cd 1
cmake -S . -B build
cmake --build build
./build/task1
```
#### 2 задача
```bash
cd 2
cmake -S . -B build
cmake --build build
./build/task2
```
#### 3 задача
##### Вариант 0: Компиляция всех задач
```bash
cd 3
cmake -DSUM=1 -B build -S .
cmake --build build
```

##### Вариант 1: Для каждого распараллеливаемого цикла создается отдельная параллельная секция #pragma omp parallel for
```bash
cd 3
cmake -DSUM=1 -B build -S .
cmake --build build
./build/variant1
```

##### Вариант 2: Cоздается одна параллельная секция #pragma omp parallel, охватывающая весь итерационный алгоритм.
```bash
cd 3
cmake -DSUM=2 -B build -S .
cmake --build build
./build/variant2
```

##### Вариант 3: Провести исследование на определение оптимальных параметров #pragma omp for schedule(...) при некотором фиксированном размере задачи и количестве потоков.
```bash
cd 3
cmake -DSUM=3 -B build -S .
cmake --build build
./build/variant3
```
