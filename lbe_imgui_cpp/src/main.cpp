#include <iostream>
#include <vector>
#include <cmath>
#include <imgui.h>
#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

// Grid parameters
const int NX = 100;     // Grid width
const int NY = 100;     // Grid height
const double tau = 0.6; // Relaxation time

// LBM parameters
const int Q = 9; // Lattice directions
const double weights[Q] = {4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
                           1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};
const int dx[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int dy[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};

// Functions for LBM
inline int index(int x, int y, int dir) {
    return (y * NX + x) * Q + dir;
}

void initialize(std::vector<double>& f, std::vector<double>& rho, std::vector<double>& ux, std::vector<double>& uy) {
    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            rho[y * NX + x] = 1.0;
            ux[y * NX + x] = 0.0;
            uy[y * NX + x] = 0.0;
            for (int d = 0; d < Q; d++) {
                f[index(x, y, d)] = weights[d];
            }
        }
    }
}

void collision(std::vector<double>& f, const std::vector<double>& rho, const std::vector<double>& ux, const std::vector<double>& uy) {
    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            double u2 = ux[y * NX + x] * ux[y * NX + x] + uy[y * NX + x] * uy[y * NX + x];
            for (int d = 0; d < Q; d++) {
                double cu = dx[d] * ux[y * NX + x] + dy[d] * uy[y * NX + x];
                double feq = weights[d] * rho[y * NX + x] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
                f[index(x, y, d)] += -1.0 / tau * (f[index(x, y, d)] - feq);
            }
        }
    }
}

void streaming(std::vector<double>& f) {
    std::vector<double> temp = f;
    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            for (int d = 0; d < Q; d++) {
                int xNew = (x + dx[d] + NX) % NX;
                int yNew = (y + dy[d] + NY) % NY;
                f[index(xNew, yNew, d)] = temp[index(x, y, d)];
            }
        }
    }
}

void updateMacros(const std::vector<double>& f, std::vector<double>& rho, std::vector<double>& ux, std::vector<double>& uy) {
    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            rho[y * NX + x] = 0.0;
            ux[y * NX + x] = 0.0;
            uy[y * NX + x] = 0.0;
            for (int d = 0; d < Q; d++) {
                double fVal = f[index(x, y, d)];
                rho[y * NX + x] += fVal;
                ux[y * NX + x] += fVal * dx[d];
                uy[y * NX + x] += fVal * dy[d];
            }
            ux[y * NX + x] /= rho[y * NX + x];
            uy[y * NX + x] /= rho[y * NX + x];
        }
    }
}

void renderDensity(const std::vector<double>& rho) {
    for (int y = 0; y < NY; y++) {
        for (int x = 0; x < NX; x++) {
            float color = static_cast<float>(rho[y * NX + x]);
            std::cout << color << std::endl;
            ImGui::GetForegroundDrawList()->AddCircleFilled(ImVec2(x * 5, y * 5), 2.0f, ImColor(color, color, color));
        }
    }
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW!" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(800, 800, "LBM Visualization", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window!" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Initialize ImGui
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // LBM state
    std::vector<double> f(NX * NY * Q, 0.0);
    std::vector<double> rho(NX * NY, 0.0);
    std::vector<double> ux(NX * NY, 0.0);
    std::vector<double> uy(NX * NY, 0.0);

    initialize(f, rho, ux, uy);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // LBM step
        collision(f, rho, ux, uy);
        streaming(f);
        updateMacros(f, rho, ux, uy);

        // Visualization
        renderDensity(rho);

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
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
