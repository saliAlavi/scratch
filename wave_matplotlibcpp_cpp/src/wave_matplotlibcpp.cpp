#include <matplotlibcpp.h>
#include <vector>
#include <cmath>

namespace plt = matplotlibcpp;

int main() {
    const int N = 100;                // Number of spatial points
    const double L = 10.0;            // Length of the domain
    const double dx = L / N;          // Spatial resolution
    const double dt = 0.01;           // Time step
    const double c = 1.0;             // Wave speed
    const int steps = 200;            // Number of time steps

    std::vector<double> x(N), u(N), u_prev(N), u_next(N);

    // Initialize the wave function and spatial points
    for (int i = 0; i < N; ++i) {
        x[i] = i * dx;
        u[i] = std::sin(2.0 * M_PI * x[i] / L); // Initial wave profile
        u_prev[i] = u[i];
    }

    // Time evolution
    for (int t = 0; t < steps; ++t) {
        // Apply the wave equation discretization
        for (int i = 1; i < N - 1; ++i) {
            u_next[i] = 2 * u[i] - u_prev[i] + (c * c * dt * dt / (dx * dx)) * (u[i+1] - 2 * u[i] + u[i-1]);
        }

        // Reflecting boundary conditions
        u_next[0] = 0;
        u_next[N-1] = 0;

        // Update
        u_prev = u;
        u = u_next;

        // Plot the wave at this time step
        plt::clf();            // Clear the previous plot
        plt::plot(x, u, "b-"); // Plot wave profile
        plt::xlabel("x");
        plt::ylabel("Amplitude");
        plt::title("Wave Propagation");
        plt::pause(0.01);      // Pause for animation
    }

    plt::show(); // Show the final plot
    return 0;
}
g++ wave_matplotlibcpp.cpp -I~/libraries/matplotlib-cpp/ -lpython3.8 