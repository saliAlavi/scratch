#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

// Simulation parameters
const int grid_size = 100;
float c = 1.0f;      // Wave speed
const float dt = 0.1f;     // Time step
const float dx = 1.0f;     // Spatial step

std::vector<std::vector<float>> u(grid_size, std::vector<float>(grid_size, 0.0f));
std::vector<std::vector<float>> u_prev(grid_size, std::vector<float>(grid_size, 0.0f));
std::vector<std::vector<float>> u_next(grid_size, std::vector<float>(grid_size, 0.0f));

// Initialize the grid with a central disturbance
void initialize_wave() {
    int center = grid_size / 2;
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            float dist = std::sqrt((i - center) * (i - center) + (j - center) * (j - center));
            u_prev[i][j] = std::exp(-dist * dist / 50.0f); // Gaussian distribution
        }
    }

    center = grid_size / 4;
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            float dist = std::sqrt((i - center) * (i - center) + (j - center) * (j - center));
            u_prev[i][j] = std::exp(-dist * dist / 50.0f); // Gaussian distribution
        }
    }
}

// // Update the wave equation
// void update_wave() {
//     for (int i = 1; i < grid_size - 1; ++i) {
//         for (int j = 1; j < grid_size - 1; ++j) {
//             float laplacian = (u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1] - 4 * u[i][j]) / (dx * dx);
//             u_next[i][j] = 2 * u[i][j] - u_prev[i][j] + c * c * dt * dt * laplacian;
//         }
//     }
//     u_prev = u;
//     u = u_next;
// }

extern void update_wave_cuda(float *u, float *u_prev, float *u_next, float c, float dt, float dx, int grid_size);

void update_wave() {
    update_wave_cuda(u[0].data(), u_prev[0].data(), u_next[0].data(), c, dt, dx, grid_size);
    u_prev.swap(u);
    u.swap(u_next);
}

// Render the grid
void render_wave() {
    glBegin(GL_POINTS);
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            float intensity = (u[i][j] + 1) / 2; // Normalize to [0, 1]
            glColor3f(intensity, intensity, intensity);
            glVertex2f(i / float(grid_size) * 2 - 1, j / float(grid_size) * 2 - 1);
        }
    }
    glEnd();
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 800, "Wave Equation Simulation", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Initialize the wave
    initialize_wave();

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Update wave simulation
        update_wave();

        // Rendering
        glClear(GL_COLOR_BUFFER_BIT);
        render_wave();

        // Render ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Wave Parameters");
        ImGui::Text("Wave simulation using the wave equation");
        ImGui::SliderFloat("Wave Speed", &c, 0.1f, 5.0f);
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
