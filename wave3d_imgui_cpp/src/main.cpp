#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <cmath>

// Simulation parameters
const int grid_size = 50;
float c = 1.0f;       // Wave speed
float dt = 0.1f;      // Time step
float dx = 1.0f;      // Spatial step

std::vector<std::vector<std::vector<float>>> u(grid_size, std::vector<std::vector<float>>(grid_size, std::vector<float>(grid_size, 0.0f)));
std::vector<std::vector<std::vector<float>>> u_prev(grid_size, std::vector<std::vector<float>>(grid_size, std::vector<float>(grid_size, 0.0f)));
std::vector<std::vector<std::vector<float>>> u_next(grid_size, std::vector<std::vector<float>>(grid_size, std::vector<float>(grid_size, 0.0f)));

// Camera controls
float cam_angle_x = 0.0f;
float cam_angle_y = 0.0f;
float cam_distance = 2.0f;

// Initialize the grid with a central disturbance
void initialize_wave() {
    int center = grid_size / 2;
    for (int x = 0; x < grid_size; ++x) {
        for (int y = 0; y < grid_size; ++y) {
            for (int z = 0; z < grid_size; ++z) {
                float dist = std::sqrt((x - center) * (x - center) + 
                                       (y - center) * (y - center) + 
                                       (z - center) * (z - center));
                u_prev[x][y][z] = std::exp(-dist * dist / 50.0f); // Gaussian distribution
            }
        }
    }
}

// Update the wave equation
void update_wave() {
    for (int x = 1; x < grid_size - 1; ++x) {
        for (int y = 1; y < grid_size - 1; ++y) {
            for (int z = 1; z < grid_size - 1; ++z) {
                float laplacian = (u[x + 1][y][z] + u[x - 1][y][z] +
                                   u[x][y + 1][z] + u[x][y - 1][z] +
                                   u[x][y][z + 1] + u[x][y][z - 1] -
                                   6 * u[x][y][z]) / (dx * dx);
                u_next[x][y][z] = 2 * u[x][y][z] - u_prev[x][y][z] + c * c * dt * dt * laplacian;
            }
        }
    }

    u_prev.swap(u);
    u.swap(u_next);
}

// OpenGL rendering for the grid
void render_grid() {
    glBegin(GL_POINTS);
    for (int x = 0; x < grid_size; ++x) {
        for (int y = 0; y < grid_size; ++y) {
            for (int z = 0; z < grid_size; ++z) {
                float intensity = (u[x][y][z] + 1.0f) / 2.0f; // Normalize to [0, 1]
                glColor3f(intensity, intensity, intensity);
                glVertex3f(x / float(grid_size), y / float(grid_size), z / float(grid_size));
            }
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

    // Create window
    GLFWwindow* window = glfwCreateWindow(800, 800, "3D Wave Simulation", nullptr, nullptr);
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

    // Setup OpenGL state
    glEnable(GL_DEPTH_TEST);
    glPointSize(2.0f);

    // Initialize simulation
    initialize_wave();

    // Main rendering loop
    while (!glfwWindowShouldClose(window)) {
        std::cout << "Running" << std::endl;        
        glfwPollEvents();

        // Update simulation
        update_wave();

        // Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Setup camera
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45.0, 1.0, 0.1, 10.0);
        gluLookAt(cam_distance * sin(cam_angle_y) * cos(cam_angle_x),
                  cam_distance * cos(cam_angle_y),
                  cam_distance * sin(cam_angle_y) * sin(cam_angle_x),
                  0.5, 0.5, 0.5,
                  0.0, 1.0, 0.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // Render wave grid
        render_grid();

        // Render ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Controls");
        ImGui::SliderFloat("Wave Speed", &c, 0.1f, 5.0f);
        ImGui::SliderAngle("Camera X", &cam_angle_x);
        ImGui::SliderAngle("Camera Y", &cam_angle_y);
        ImGui::SliderFloat("Camera Distance", &cam_distance, 1.0f, 5.0f);
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap buffers
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
