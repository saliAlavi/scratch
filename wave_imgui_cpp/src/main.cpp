#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <deque>
#include <numeric> // For std::accumulate

const int average_frames = 100; // Number of frames to average
std::deque<float> frame_times;

// Update FPS calculation
void calculate_fps(double delta_time, float& smoothed_fps) {
    frame_times.push_back(static_cast<float>(delta_time)); // Add the new frame time
    if (frame_times.size() > average_frames) {
        frame_times.pop_front(); // Remove the oldest frame time if the queue is too long
    }

    float total_time = std::accumulate(frame_times.begin(), frame_times.end(), 0.0f);
    if (total_time > 0) {
        smoothed_fps = frame_times.size() / total_time; // Calculate the moving average FPS
    }
}

const int grid_size = 1000;
float c = 1.0f;      // Wave speed
const float dt = 0.1f;     // Time step
const float dx = 1.5f;     // Spatial step
float wave_speed = 1.0f;

float* d_u = nullptr;
float* d_u_prev = nullptr;
float* d_u_next = nullptr;

// CUDA-related declarations
extern void initializeWaveCuda(float* d_u, float* d_u_prev, int grid_size);
extern void updateWaveCuda(float* d_u, float* d_u_prev, float* d_u_next, int grid_size, float c, float dt, float dx);

void allocateCudaMemory(float** d_u, float** d_u_prev, float** d_u_next, int grid_size) {
    cudaMalloc((void**)d_u, grid_size * grid_size * sizeof(float));
    cudaMalloc((void**)d_u_prev, grid_size * grid_size * sizeof(float));
    cudaMalloc((void**)d_u_next, grid_size * grid_size * sizeof(float));
}

// Free CUDA memory
void freeCudaMemory(float* d_u, float* d_u_prev, float* d_u_next) {
    if (d_u) cudaFree(d_u);
    if (d_u_prev) cudaFree(d_u_prev);
    if (d_u_next) cudaFree(d_u_next);
}

// Render the grid
void renderWave(float* d_u, int grid_size) {
    std::vector<float> u_host(grid_size * grid_size);
    cudaMemcpy(u_host.data(), d_u, grid_size * grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    // std::cout << u_host[grid_size * grid_size/2] << std::endl;
    glBegin(GL_POINTS);
    for (int i = 0; i < grid_size; ++i) {
        for (int j = 0; j < grid_size; ++j) {
            // float intensity = (u_host[i * grid_size + j] + 1.0f) / 2.0f; // Normalize to [0, 1]
            float intensity = u_host[i * grid_size + j]*2 ;
            // std::cout << intensity << std::endl;
            glColor3f(intensity, intensity, intensity);
            glVertex2f(i / float(grid_size) * 2.0f - 1.0f, j / float(grid_size) * 2.0f - 1.0f);
        }
    }
    glEnd();
}

int main() {
    
    // Simulation parameters
    // const int grid_size = 200;
    // float c = 1.0f;      // Wave speed
    // const float dt = 0.1f;     // Time step
    // const float dx = 1.0f;     // Spatial step
    // float wave_speed = 1.0f;

    // float* d_u = nullptr;
    // float* d_u_prev = nullptr;
    // float* d_u_next = nullptr;

    allocateCudaMemory(&d_u, &d_u_prev, &d_u_next, grid_size);
    initializeWaveCuda(d_u, d_u_prev, grid_size);
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        freeCudaMemory(d_u, d_u_prev, d_u_next);
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 800, "Wave Equation Simulation", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        freeCudaMemory(d_u, d_u_prev, d_u_next);
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        freeCudaMemory(d_u, d_u_prev, d_u_next);
        return -1;
    }

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Set up OpenGL fps vars
    double last_time = glfwGetTime();
    float smoothed_fps = 0.0f;
    
    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Update FPS calculation
        double current_time = glfwGetTime();
        double delta_time = current_time - last_time;
        last_time = current_time;
        calculate_fps(delta_time, smoothed_fps);

        // Update wave simulation
        updateWaveCuda(d_u, d_u_prev, d_u_next, grid_size, wave_speed, dt, dx);

        // Render the wave
        glClear(GL_COLOR_BUFFER_BIT);
        // renderWave(d_u);
        renderWave(d_u, grid_size);

        // Rendering    
        glfwPollEvents();

        // Check for errors in CUDA
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
            freeCudaMemory(d_u, d_u_prev, d_u_next);
            return -1;
        }

        // Render ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Display FPS at the top right corner
        // Render ImGui FPS window
        ImGui::SetNextWindowPos(ImVec2(10, 10));
        ImGui::Begin("Performance", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("FPS (avg): %.1f", smoothed_fps);
        ImGui::End();
        
        ImGui::Begin("Wave Parameters");
        ImGui::SliderFloat("Wave Speed", &wave_speed, 0.1f, 5.0f);
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    freeCudaMemory(d_u, d_u_prev, d_u_next);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
    }