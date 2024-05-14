#include <iostream>
#include <cmath>
#include <random>
#include <vector>

using namespace std;

float dist(const std::vector<float>& point) {
    float sum = 0.0;
    for(float coord : point){
        sum += coord * coord;
    }
    return sqrt(sum);
}

int main(){
    for(int dim = 2; dim <= 16; dim++){

        unsigned long long sam_points = 1000000;
        unsigned long long true_points = 0;
        vector<unsigned int> hist(100, 0);

        default_random_engine eng;
        uniform_real_distribution<float> unif(-1, 1);

        for(unsigned long long i = 0; i < sam_points; i++){
            vector<float> points(dim);
            for(int j = 0; j < dim; j++){
                points[j] = unif(eng);
            }

            float distance = dist(points);
            int bin = static_cast<int>(distance*100);
            if(bin < 100){
                hist[bin]++;
                true_points++;
            }
        }
        cout << "Dimension: " << dim << "\n";
        for(int i = 0; i < 100; i++){
            cout << static_cast<float>(hist[i]*100)/true_points << " ";
        }
        cout << "\n\n";
    }

    return 0;
}