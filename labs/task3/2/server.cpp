#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <tuple>
#include <functional>
#include <fstream> 
#include <random>

template<typename T>
T func_pow(T x, T y){
    return static_cast<T>(std::pow(x, y));
}

template<typename T>
T func_sin(T x){
    return static_cast<T>(std::sin(x));
}

template<typename T>
T func_add(T x, T y, T z){
    return static_cast<T>(x + y + z);
}


template<typename T>
class Server {
private:
    std::atomic<bool> stop{false};
    std::atomic<size_t> count_tasks_id{0};
    
    using TaskWrapper = std::function<T()>;
    std::queue<std::pair<size_t, TaskWrapper>> tasks;
    std::mutex tasks_mutex;
    std::condition_variable tasks_cv;
    
    std::unordered_map<size_t, T> results;
    std::mutex res_mutex;
    std::condition_variable res_cv;
    
    std::thread worker_thread;
    
    void worker_loop() {        
        while (!stop) {
            std::unique_lock<std::mutex> lock(tasks_mutex);
            tasks_cv.wait(lock, [this] {return !tasks.empty() || stop;});
            
            if (stop && tasks.empty()) break;
            
            if (!tasks.empty()) {
                auto [task_id, task] = std::move(tasks.front());
                tasks.pop();
                lock.unlock();
                
                T result = task();
                
                {
                    std::lock_guard<std::mutex> res_lock(res_mutex);
                    results[task_id] = result;
                }
                res_cv.notify_all();  
                
            }
        }
    }
    
public:
    Server() = default;
    
    ~Server() = default;
    
    void start_server() {
        if (worker_thread.joinable()) {
            return;
        }
        stop = false;
        worker_thread = std::thread(&Server::worker_loop, this);
        std::cout << "[Server] Started\n";
    }
    
    void stop_server() {
        stop = true;
        tasks_cv.notify_all();
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
        std::cout << "[Server] Stopped\n";
    }
    
    template<typename Func, typename... Args>
    size_t add_task(size_t client_id, Func&& f, Args&&... args) {
        auto args_tuple = std::make_tuple(std::forward<Args>(args)...);
        
        TaskWrapper task_wrapper = [f, args_tuple]() -> T {
            return std::apply(f, args_tuple);
        };
        
        size_t task_id = ++count_tasks_id;
    
        {
            std::lock_guard<std::mutex> lock(tasks_mutex);
            tasks.emplace(task_id, std::move(task_wrapper));
        }
        tasks_cv.notify_one();
        return task_id;
    }

    T request_result(size_t id) {
        std::unique_lock<std::mutex> lock(res_mutex);
        res_cv.wait(lock, [this, id] { return results.find(id) != results.end() || stop;});        
        T res = results[id];
        return res;
    }
    
    void save_client_results(size_t client_id, const std::string& filename, std::vector<T>& expected_results,  std::vector<size_t>& ids) {        
        std::ofstream out(filename);
        size_t wrong_done = 0;
        size_t len = ids.size();
        
        for (size_t i = 0; i < len; i++) {
            T result = request_result(ids[i]);
            if (std::abs(result - expected_results[i]) > 1e-9){
                ++wrong_done;
                out << ids[i] << " - WRONG: expected: " << expected_results[i] << ", received: " <<  result << std::endl;
                continue;
            }
            out << ids[i] << " - " <<  result << std::endl;
        }
        double acc = static_cast<double>(len - wrong_done) / len;
        out << "ACCURACY: " << acc << std::endl;
        std::cout << "Client " << client_id << " finished, ACCURACY: " << acc << ",  results saved to " << filename << std::endl;
    }
};

template<typename T, typename Func, typename ArgGen>
void client_thread(Server<T>& server, int client_id, Func func, ArgGen arg_generator, const std::string& filename, int N) {
    std::vector<size_t> ids(N);
    std::vector<T> expected_results(N);
    
    for (int i = 0; i < N; ++i) {
        auto args = arg_generator();  // генерируем аргументы
        T expected = std::apply(func, args);
        expected_results[i] = expected;
        
        ids[i] = server.add_task(client_id, [func, args]() -> T {
            return std::apply(func, args);
        });
    }
    
    server.save_client_results(client_id, filename, expected_results, ids);
}

double random_double(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

int main() {
    const int N = 1000; 
    Server<double> server;
    server.start_server();
    
    auto sin_gen = []() { return std::make_tuple(random_double(0, 2*M_PI)); };
    std::thread client1(client_thread<double, decltype(func_sin<double>), decltype(sin_gen)>, std::ref(server), 1, func_sin<double>, sin_gen, "results/client1_sin.txt", N);

    auto pow_gen = []() { return std::make_tuple(random_double(1, 5), random_double(1, 3)); };
    std::thread client2(client_thread<double, decltype(func_pow<double>), decltype(pow_gen)>, std::ref(server), 2, func_pow<double>, pow_gen, "results/client2_pow.txt", N);

    auto add_gen = []() { return std::make_tuple(random_double(0, 100), random_double(0, 100), random_double(0, 100)); };
    std::thread client3(client_thread<double, decltype(func_add<double>), decltype(add_gen)>, std::ref(server), 3, func_add<double>, add_gen, "results/client3_add.txt", N);

    client1.join();
    client2.join();
    client3.join();

    server.stop_server();
    return 0;
}