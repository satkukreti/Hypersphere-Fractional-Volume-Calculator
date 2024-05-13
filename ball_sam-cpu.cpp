#include <iostream>
#include <cmath>
#include <random>
#include <vector>

using namespace std;

int main(){
    default_random_engine eng;
    uniform_real_distribution<float> dist(-1, 1);

    cout << dist(eng) << " " << dist(eng) << endl;
}