#include <iostream>
#include <random>
#include <ctime>
#include <cuda_runtime.h>
#include <raylib.h>
#include <string>

#include <curand_kernel.h>

const int size = 128; 
const float temperature_max = 10.0;
const float temperature_min = 0.01;

// Function to initialize the 2D spin configuration
void initializeSpins(int *spins, int size) {
    std::mt19937 rng(std::time(0));
    std::uniform_int_distribution<int> distribution(0, 1);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            spins[i * size + j] = 2*distribution(rng)-1; 
        }
    }
}

__global__ void updateSpins2D(int *spins, int size, float temperature, float J) {
    curandState state;
    curand_init(clock64(), blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y, &state);

    int row = (blockIdx.x*blockDim.x+threadIdx.x + curand(&state)) % size;
    int col = (blockIdx.y*blockDim.y+threadIdx.y + curand(&state)) % size;

    float p = (float)curand(&state) / (float)UINT_MAX;
        // Perform Metropolis algorithm
        int idx = row * size + col;
        int neighbors = spins[((row + 1) % size) * size + col] +
                        spins[((row - 1 + size) % size) * size + col] +
                        spins[row * size + (col + 1) % size] +
                        spins[row * size + (col - 1 + size) % size];

        float deltaE = 2.0* J * spins[idx] * neighbors;

        if (deltaE < 0.0f || p < exp(-deltaE / temperature)) 
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
    int max_threads;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // Assumes device 0
    cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerBlock, 0);
    int threads = sqrt(max_threads);
    int blocks = size / threads + (size % threads != 0);
    dim3 gridSize(blocks, blocks);
    dim3 blockSize(threads, threads);

    float temperature = 1.0 ;
    float J = 1.0; 
    int *spins;
    float *randNums;

    int screenWidth = 1024;
    int screenHeight = 800;
    int displayWidth = 1024;
    int displayHeight = 750;
    int rectWidth = displayWidth / size;
    int rectHeight = displayHeight / size;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    
    InitWindow(screenWidth, screenHeight, "2D Ising Model Visualization");
    SetTargetFPS(30);
    // Allocate memory on the CPU
    spins = new int[size* size];
    randNums = new float[size*size];

    // Allocate memory on the GPU
    int *dev_spins;
    float *dev_randNums;

    cudaMalloc((void**)&dev_spins, size * size * sizeof(int));
    cudaMalloc((void**)&dev_randNums, size * size * sizeof(float));

    // Initialize 2D spin configuration on the CPU
    initializeSpins(spins, size);

    // Copy data from CPU to GPU
    cudaMemcpy(dev_spins, spins, size * size * sizeof(int), cudaMemcpyHostToDevice);

    // Main simulation loop
    while (!WindowShouldClose()) {

        updateSpins2D<<<gridSize, blockSize>>>(dev_spins, size, temperature, J);
        cudaDeviceSynchronize();

        // Copy data from GPU to CPU for visualization
        cudaMemcpy(spins, dev_spins, size * size * sizeof(int), cudaMemcpyDeviceToHost);
 
        BeginDrawing();
        ClearBackground(RAYWHITE);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int idx = i * size + j;
                Color color = ((spins[idx] == 1) ? RED : BLUE);
                DrawRectangle(j * rectWidth, i * rectHeight, rectWidth, rectHeight, color);
            }
        }
        
        if (reset_button(displayWidth / 10, displayHeight - 80) || IsKeyPressed(KEY_R))
        {
            initializeSpins(spins, size);
            cudaMemcpy(dev_spins, spins, size * size * sizeof(int), cudaMemcpyHostToDevice);
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
