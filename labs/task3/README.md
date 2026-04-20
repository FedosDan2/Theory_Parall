## Сборка
#### 1 задача

##### Решение через std::thread
```bash
cd 1
cmake -S . -B build
cmake --build build
./build/thr_runner
```

##### Решение через std::jthread
```bash
cd 1
cmake -S . -B build
cmake --build build
./build/jthr_runner
```

#### 2 задача
```bash
cd 2
./start.sh
```