#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace std;

//bins and samples
const int n_bins = 100;
const int n_points = 3000;

int main(){
    //random unif num generator
    default_random_engine eng;
    uniform_real_distribution<double> unif(-1.0, 1.0);

    for(int dim = 2; dim <= 16; dim++){
        vector<int> hist(n_bins, 0);

        for(int i = 0; i < n_points; i++){
            vector<double> points(dim);
            double sum_squares = 0.0;

            //point validation
            do {
                sum_squares = 0.0;
                for(double& p : points){
                    p = unif(eng);
                    sum_squares += p * p;
                    if(sum_squares > 1.0) //early break, helps with speed
                        break;
                }
            } while(sum_squares > 1.0); //redo if invalid

            double distance = sqrt(sum_squares);
            int bin = min((int)(distance * n_bins), n_bins - 1);
            hist[bin]++;
        }
        //print results
        cout << "Dimension: " << dim << "\n";
        for(int i = 0; i < n_bins; i++){
            cout << (double)(hist[i]) / n_points << " ";
        }
        cout << "\n\n";
    }

    return 0;
}
