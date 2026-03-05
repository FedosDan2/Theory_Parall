#include <iostream>
#include <vector>
#include <cmath>

#define MAX_EL 10
#define TWO_PI static_cast<TYPE>(2.0 * M_PI)

int main() {
#if defined(USE_DOUBLE) && USE_DOUBLE
    using TYPE = double;
    std::cout << "Double is using!\n";
#else
    using TYPE = float;
    std::cout << "Float is using.\n";
#endif

    const TYPE STEP = TWO_PI / static_cast<TYPE>(MAX_EL-1);
    std::vector<TYPE> arr;
    arr.reserve(MAX_EL);
    
    for (size_t i = 0; i < MAX_EL; ++i) {
        arr.push_back(std::sin(STEP * static_cast<TYPE>(i))); 
    }

    TYPE sum = 0;
    for (size_t i = 0; i < MAX_EL; i++) {
        sum += static_cast<TYPE>(arr[i]);
    }

    std::cout << "Sum = " << sum << std::endl;
    
    return 0;
}