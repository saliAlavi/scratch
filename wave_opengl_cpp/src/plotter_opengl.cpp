#include <GL/glut.h>
#include <cmath>
#include <vector>

const int WIDTH = 50;   // Grid width
const int HEIGHT = 50;  // Grid height
const float C = 0.1f;   // Wave speed
const float DT = 0.1f;  // Time step
const float DX = 0.1f;  // Spatial step in X direction
const float DY = 0.1f;  // Spatial step in Y direction

// Variables to store the current and previous wave states
std::vector<std::vector<float>> u_curr(HEIGHT, std::vector<float>(WIDTH, 0.0f));
std::vector<std::vector<float>> u_prev(HEIGHT, std::vector<float>(WIDTH, 0.0f));
std::vector<std::vector<float>> u_next(HEIGHT, std::vector<float>(WIDTH, 0.0f));

// Function to update the wave equation using finite differences
void updateWave() {
    for (int i = 1; i < HEIGHT - 1; i++) {
        for (int j = 1; j < WIDTH - 1; j++) {
            // 2D wave equation using finite differences
            u_next[i][j] = 2 * u_curr[i][j] - u_prev[i][j] +
                           (C * C) * ((u_curr[i-1][j] + u_curr[i+1][j] - 2 * u_curr[i][j]) / (DX * DX) +
                                      (u_curr[i][j-1] + u_curr[i][j+1] - 2 * u_curr[i][j]) / (DY * DY));
        }
    }

    // Swap previous and current states for the next update
    u_prev = u_curr;
    u_curr = u_next;
}

// Function to visualize the wave in OpenGL
void drawWave() {
    glClear(GL_COLOR_BUFFER_BIT);

    // Set up the color mapping based on the wave displacement
    glBegin(GL_QUADS);
    for (int i = 0; i < HEIGHT - 1; i++) {
        for (int j = 0; j < WIDTH - 1; j++) {
            float x = j * DX;
            float y = i * DY;
            float z = u_curr[i][j];
            float z_next = u_curr[i][j + 1];
            float z_down = u_curr[i + 1][j];
            float z_down_next = u_curr[i + 1][j + 1];

            glColor3f((z + 1) / 2, (z_next + 1) / 2, (z_down + 1) / 2); // Color based on height

            // Drawing a quad for the grid cell
            glVertex3f(x, y, z);
            glVertex3f(x + DX, y, z_next);
            glVertex3f(x + DX, y + DY, z_down_next);
            glVertex3f(x, y + DY, z_down);
        }
    }
    glEnd();

    glutSwapBuffers();
}

// Function to initialize OpenGL
void myInit() {
    glClearColor(0.0, 0.0, 0.0, 1.0);   // Set background color to black
    glEnable(GL_DEPTH_TEST);              // Enable depth testing
    glMatrixMode(GL_PROJECTION);          // Set projection matrix
    glLoadIdentity();
    gluPerspective(45.0, 1.0, 0.1, 100.0); // Set perspective projection
    glTranslatef(-WIDTH / 2.0f, -HEIGHT / 2.0f, -200.0f); // Translate for better view
}

// Function to update the simulation at each frame
void update(int value) {
    updateWave();    // Update wave state
    glutPostRedisplay();  // Redraw the screen
    glutTimerFunc(30, update, 0);  // Call update function every 30 ms
}

// Main program to set up the GLUT window and OpenGL context
int main(int argc, char **argv) {
    // Initialize wave with initial displacement
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            u_curr[i][j] = sin(M_PI * i / HEIGHT) * cos(M_PI * j / WIDTH);  // Initial condition
        }
    }

    // Set up GLUT window and OpenGL context
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);  // Double buffer, RGB colors, depth test
    glutInitWindowSize(800, 600);
    glutCreateWindow("2D Wave Equation Simulation");
    glutDisplayFunc(drawWave);  // Register the drawing function
    myInit();                  // Initialize OpenGL settings
    glutTimerFunc(30, update, 0);  // Start updating the wave with a time interval
    glutMainLoop();             // Enter the GLUT main loop
    return 0;
}
