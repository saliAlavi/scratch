#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>

// Random number generator for branching and pruning
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

struct Dendrite {
    float x, y;            // Start point
    float angle;           // Growth angle in radians
    float length;          // Length of the branch
    std::vector<Dendrite> children; // Child branches

    // Constructor
    Dendrite(float startX, float startY, float angle, float length)
        : x(startX), y(startY), angle(angle), length(length) {}

    // Grow function with branching and pruning
    void grow(float growthRate = 2.0f, float branchProb = 0.1f, float pruneProb = 0.02f) {
        length += growthRate;

        // Branching logic
        if (dis(gen) < branchProb) {
            float branchAngleVariation = M_PI / 4; // ±45 degrees
            children.emplace_back(x + length * cos(angle), 
                                  y + length * sin(angle), 
                                  angle + branchAngleVariation, 
                                  length / 2);
            children.emplace_back(x + length * cos(angle), 
                                  y + length * sin(angle), 
                                  angle - branchAngleVariation, 
                                  length / 2);
        }

        // Pruning logic
        if (dis(gen) < pruneProb) {
            children.clear(); // Remove all children
        }
    }

    // Render the branch and its children
    void render() {
        float endX = x + length * cos(angle);
        float endY = y + length * sin(angle);
        ImGui::GetWindowDrawList()->AddLine(
            ImVec2(x + 400, -y + 400),  // Adjusting to screen center
            ImVec2(endX + 400, -endY + 400),
            IM_COL32(0, 0, 0, 255), 2.0f
        );

        // Render children recursively
        for (auto& child : children) {
            child.render();
        }
    }
};

// Main function
int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    // Create GLFW window
    GLFWwindow* window = glfwCreateWindow(800, 800, "Purkinje Dendritic Growth", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Root dendrite
    Dendrite root(0.0f, 0.0f, M_PI / 2, 5.0f);
    std::vector<Dendrite*> dendrites = { &root };

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Create ImGui window
        ImGui::Begin("Purkinje Dendritic Growth");

        // Add controls for tuning
        static float growthRate = 2.0f;
        static float branchProbability = 0.1f;
        static float pruneProbability = 0.02f;
        ImGui::SliderFloat("Growth Rate", &growthRate, 0.5f, 5.0f);
        ImGui::SliderFloat("Branch Probability", &branchProbability, 0.0f, 0.3f);
        ImGui::SliderFloat("Prune Probability", &pruneProbability, 0.0f, 0.1f);

        // Simulate growth
        std::vector<Dendrite*> newDendrites;
        for (auto dendrite : dendrites) {
            dendrite->grow(growthRate, branchProbability, pruneProbability);
            newDendrites.push_back(dendrite);
            for (auto& child : dendrite->children) {
                newDendrites.push_back(&child);
            }
        }
        dendrites = newDendrites;

        // Render the dendritic tree
        root.render();
        ImGui::End();

        // Render ImGui
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
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