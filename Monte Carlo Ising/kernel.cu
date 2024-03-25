#include <iostream>
#include <random>
#include <ctime>
#include <cuda_runtime.h>
#include <raylib.h>
#include <string>

#include <curand_kernel.h>

const int rows = 128; 
const int cols = 128; 
const int numThreadsPerBlockX = 8;
const int numThreadsPerBlockY = 8;

const int numBlocksX = cols / numThreadsPerBlockX;
const int numBlocksY = rows  / numThreadsPerBlockY;
const float temperature_max = 10.0;
const float temperature_min = 0.01;

// Function to initialize the 2D spin configuration
void initializeSpins(int *spins, int rows, int cols) {
    std::mt19937 rng(std::time(0));
    std::uniform_int_distribution<int> distribution(0, 1);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            spins[i * cols + j] = 2*distribution(rng)-1; 
        }
    }
}

__global__ void updateSpins2D(int *spins, float *randNums, int rows, int cols, float temperature, float J) {
    curandState state;
    curand_init(clock64(), blockIdx.x*blockDim.x+threadIdx.x, threadIdx.x+threadIdx.y, &state);

    int row = curand(&state) % rows;
    int col = curand(&state) % cols;

        // Perform Metropolis algorithm
        int idx = row * cols + col;
        int neighbors = spins[((row + 1) % rows) * cols + col] +
                        spins[((row - 1 + rows) % rows) * cols + col] +
                        spins[row * cols + (col + 1) % cols] +
                        spins[row * cols + (col - 1 + cols) % cols];

        float deltaE = 2.0* J * spins[idx] * neighbors;

        if (deltaE < 0.0f || randNums[idx] < exp(-deltaE / temperature)) 
            spins[idx] *= -1;  // Flip the spin
}

bool reset_button(int X, int Y)
{
    DrawText("Reset", X, Y, 30, BLACK);
    Vector2 mp = GetMousePosition();
    bool hover=false;
    if ((mp.x >= X) && (mp.x <= X + 200) && (mp.y >= Y + 45) && (mp.y <= Y + 45 + 12+30))
         hover = true;
    Color color = (hover) ? GRAY : BLACK;
    DrawRectangleLines(X, Y + 45, 200, 50, color);
    DrawText("(R)", X + 25, Y + 45 + 12, 30, color);

    return (hover && IsMouseButtonReleased(MOUSE_BUTTON_LEFT));
}

bool j_button(double J, int X, int Y)
{
    char temp[50];
    strcpy(temp, (J > 0.0) ? "Ferromagnetic" : "Antiferromagnetic");
    DrawText(temp, X, Y, 30, BLACK);
    Vector2 mp = GetMousePosition();
    bool hover=false;
    if ((mp.x >= X) && (mp.x <= X + 200) && (mp.y >= Y + 45) && (mp.y <= Y + 45 + 12+30))
         hover = true;
    Color color = (hover) ? GRAY : BLACK;
    DrawRectangleLines(X, Y + 45, 200, 50, color);
    DrawText("Toggle (J)", X + 25, Y + 45 + 12, 30, color);

    return (hover && IsMouseButtonReleased(MOUSE_BUTTON_LEFT));
}

double temperature_slider(double temperature, int X, int Y)
{
    char temp[50];
    sprintf(temp, "Temperature: %.*f", 2, temperature);
    DrawText(temp, X, Y, 30, BLACK);
    DrawRectangleLines(X, Y + 65, 200, 10, BLACK);
    Vector2 center = {X + 200 * (temperature - temperature_min) / temperature_max, Y + 70};
    float radius = 20.0;
    Vector2 mp = GetMousePosition();
    bool hover = CheckCollisionPointCircle(mp, center, radius);

    Color color = (hover) ? GRAY : BLACK;
    if (hover && IsMouseButtonDown(MOUSE_BUTTON_LEFT))
    {
        center.x = mp.x;
    }
    DrawCircleV(center, radius, color);
    temperature = (center.x - X)/200.0 * temperature_max + temperature_min;
    temperature = (temperature > temperature_max) ? temperature_max : temperature;
    temperature = (temperature < temperature_min) ? temperature_min : temperature;

    return temperature;
}

int main() {

    dim3 blocks(numBlocksX, numBlocksY);
    dim3 threads(numThreadsPerBlockX, numThreadsPerBlockY);
    float temperature = 1.0 ;
    float J = 1.0; 
    int *spins;
    float *randNums;

    int screenWidth = 1024;
    int screenHeight = 800;
    int displayWidth = 1024;
    int displayHeight = 750;
    int rectWidth = displayWidth / cols;
    int rectHeight = displayHeight / rows;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    
    InitWindow(screenWidth, screenHeight, "2D Ising Model Visualization");
    SetTargetFPS(30);
    // Allocate memory on the CPU
    spins = new int[rows * cols];
    randNums = new float[rows * cols];

    // Allocate memory on the GPU
    int *dev_spins;
    float *dev_randNums;

    cudaMalloc((void**)&dev_spins, rows * cols * sizeof(int));
    cudaMalloc((void**)&dev_randNums, rows * cols * sizeof(float));

    // Initialize 2D spin configuration on the CPU
    initializeSpins(spins, rows, cols);

    // Copy data from CPU to GPU
    cudaMemcpy(dev_spins, spins, rows * cols * sizeof(int), cudaMemcpyHostToDevice);

    // Main simulation loop
    while (!WindowShouldClose()) {
        // Generate random numbers on the CPU and copy to GPU
        for (int i = 0; i < rows * cols; i++) {
            randNums[i] = dis(gen);
        }
        
        cudaMemcpy(dev_randNums, randNums, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

       // for (int i = 0; i < 100; i++) {
            updateSpins2D<<<blocks, threads>>>(dev_spins, dev_randNums, rows, cols, temperature, J);
            cudaDeviceSynchronize();
        //}
        // Copy data from GPU to CPU for visualization
        cudaMemcpy(spins, dev_spins, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

        BeginDrawing();
        ClearBackground(RAYWHITE);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int idx = i * cols + j;
                Color color = ((spins[idx] == 1) ? RED : BLUE);
                DrawRectangle(j * rectWidth, i * rectHeight, rectWidth, rectHeight, color);
            }
        }
        
        if (reset_button(displayWidth / 10, displayHeight - 80) || IsKeyPressed(KEY_R))
        {
            initializeSpins(spins, rows, cols);
            cudaMemcpy(dev_spins, spins, rows * cols * sizeof(int), cudaMemcpyHostToDevice);
        }
        if (j_button(J,4*displayWidth/10, displayHeight - 80 ) || IsKeyPressed(KEY_J)) J *= -1.0;
        temperature = temperature_slider(temperature, 7*displayWidth/10, displayHeight - 80);

        EndDrawing();

        if (IsKeyDown(KEY_RIGHT))
        {
            temperature += 0.1;
            temperature = ((temperature > temperature_max) ? temperature_max : temperature);
        }
        if (IsKeyDown(KEY_LEFT))
        {
            temperature -= 0.1;
            temperature = ((temperature < temperature_min) ? temperature_min : temperature);
        }

    }

    // Clean up
    delete[] spins;
    delete[] randNums;

    cudaFree(dev_spins);
    cudaFree(dev_randNums);

    CloseWindow();

    return 0;
}
