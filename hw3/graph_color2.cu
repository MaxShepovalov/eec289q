//graph coloring
#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <math.h>


//
//print all verticies. Default maximum 100 verticies
//

//#define PRINTALL

//
//print which steps the program running (Kernel launch, CPU search or check loops)
//

//#define PRINT_DEBUG

//
//print loop info (while loop) to see if the program stalls
//

#define PRINT_LOOP

//file parsers from the example
int ReadColFile(const char filename[], bool** graph, int* V);
int ReadMMFile(const char filename[], bool** graph, int* V);

//////////////////////////////////////////////////////////////////////////////////////////////////

//run neighbor color for each pairs of verticies
__global__ void KernelNeighbourColor(bool* graph, int* colors, bool* output, int V, int* work){
    int work_index = floorf((blockIdx.x * blockDim.x + threadIdx.x)/V); //primary vertex selector from work list
    int near  = (blockIdx.x * blockDim.x + threadIdx.x) % V;           //neighbor vertex index         (col of graph)
    int index = work[work_index];            //primary vertex index;

    //find color for neighbour
    int color_near_r = 0;
    if (graph[index * V + near]){
        color_near_r = colors[near];
    }

    //mark used color
    if (color_near_r != 0){
        output[index * V + color_near_r] = true;
    }
}

//run search per each vertex
__global__ void KernelSearchColor(int* colors, bool* nearcolors, int V, int N, int* work, int work_offset){

    //work index
    int work_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (work_index < N){

        //local vertex index for the given part of graph
        int index = work[work_index];
        
        //real vertex index in colors
        int index_real = index + work_offset;

        //scan near_colors for the first unoccupied color
        for (int clr = 1; clr < V; clr ++){
            if (!nearcolors[index * V + clr]){
                colors[index_real] = clr;
                break;
            }
        }
    }
}

//run check per each vertex
__global__ void KernelCheckColor(bool* graph, int* colors, int V, int* work, int* new_work, int work_offset){
    
    //work index
    int work_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    //local vertex index for the given part of graph
    int index = work[work_index];
    
    //vertex index in colors
    int index_real = index + work_offset;
    
    //default value (-1 = no work)
    int new_work_id = -1;

    //scan if selected vertex has neighbours with the same color
    for (int i = index_real + 1; i < V; i++) {

        //only consider neighbors with higher index
        if (graph[index * V + i] and colors[i]==colors[index_real]) {
            new_work_id = index;
            break;
        }
    }

    //update work
    new_work[work_index] = new_work_id;
}

//clean boolean array with given size
__global__ void KernelBoolClear(bool* array, int size){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
        array[index] = false;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////

void GraphColoringGPU(const char filename[], int** color){
    int V;         //number of verticies
    bool* graph_h; //graph matrix on host
    bool* graph_d; //graph matrix on device

    //read graph file
    int file_error = 0;
    if (std::string(filename).find(".col") != std::string::npos)
        file_error = ReadColFile(filename, &graph_h, &V);
    else if (std::string(filename).find(".mm") != std::string::npos) 
        file_error = ReadMMFile(filename, &graph_h, &V);
    else {
        //exit now, if cannot parse the file
        std::cout << "Cannot parse the file\n";
        return;
    }

    //if files were not ready, exit now
    if (file_error!=0){
        std::cout << "File error\n";
        return;
    }

    //read and report available memory
    size_t free, total;
    cudaMemGetInfo(&free,&total); 
    printf("\nGPU: %d KB free of total %d KB\n",free/1024,total/1024);

////////////////////////////////////ALLOCATION

    //find memory devision
    // number of verticies per one full-memory alocation
    int Nverticies = min(V, int(floor((free -20*1024*1024 - 4 * V)/(2*V + 4))));
    
    //allocate list of colors for GPU
    cudaError malloc_err = cudaMallocManaged(color, V * sizeof(int));
        if (malloc_err != cudaSuccess){
            std::cout << "COLOR_MALLOC cuda malloc ERROR happened: " << cudaGetErrorName(malloc_err) << std::endl;
            exit(malloc_err);
        }
    for (int i=0; i < V; i++){
        (*color)[i] = 0;
    }

    //allocate work list for GPU (indicies of verticies to process)
    int* work;
    bool* near_colors;
    malloc_err = cudaMallocManaged(&work, Nverticies * sizeof(int));
        if (malloc_err != cudaSuccess){
            std::cout << "JOB_MALLOC cuda malloc ERROR happened: " << cudaGetErrorName(malloc_err) << std::endl;
            exit(malloc_err);
        }

    //allocate part of graph for GPU
    malloc_err = cudaMalloc(&graph_d, V * Nverticies * sizeof(bool));
        if (malloc_err != cudaSuccess){
            std::cout << "GRAPH_MALLOC cuda malloc ERROR happened: " << cudaGetErrorName(malloc_err) << std::endl;
            exit(malloc_err);
        }

    //allocate array for saving neigbour color between kernels
    malloc_err = cudaMallocManaged(&near_colors, V * Nverticies * sizeof(bool));
        if (malloc_err != cudaSuccess){
            std::cout << "NEAR_MALLOC cuda malloc ERROR happened: " << cudaGetErrorName(malloc_err) << std::endl;
            exit(malloc_err);
        }

////////////////////////////////////ALGORITHM

    //go from last part of graph to first (higher indicies -> lower indicies)
    for (int V_start_r = V - Nverticies; V_start_r > -Nverticies; V_start_r -= Nverticies){

        //start should not be below 0
        int V_start = max(0, V_start_r);

        //actual amount of verticies that awailable for proccessing
        int Nv = min(Nverticies - V_start + V_start_r, V - V_start);

        //fill worklist
        for (int i = 0; i < Nv; i++){
            work[i] = i;
        }
        bool done = false;

        //move part of the graph to device memory
        cudaMemcpy(graph_d, graph_h + V_start * V, V * Nv * sizeof(bool), cudaMemcpyHostToDevice);
    
        #ifdef PRINT_LOOP
            //count how many loops passed
            int D = 0;
        #endif
        
        //amount of work per iteration
        int N;

        //repeat until find solition for the given part
        while (!done){

            //sort work list and count amount of work
            N = 0;
                //int carry = 0;
            for (int j=0; j < Nv; j++){
                    if (work[j] != -1) {
                        if (j != N){
                            work[N] = work[j];
                            work[j] = -1;
                        }
                        N++;
                    }
            }

            #ifdef PRINT_LOOP
                //visual feedback
                D++;
                printf("====while loop, %d verticies need processing; part from V %d with %d verticies; iteration:%d\n", N, V_start, Nv,D);
            #endif
            
            //if no work should be done, exit loop. Properly, (bool)done should have done that, but just in case...
            if (N == 0) break;

            //clear OR initialize near_color array
            int nthreads = min(512, V*Nverticies);
            int nblocks = ceil(V * Nverticies / nthreads);
            KernelBoolClear<<<nthreads, nblocks>>>(near_colors, Nverticies);

            //scan color of neighbours
            nthreads = min(512, V*N);
            nblocks = ceil(float(V*N)/nthreads);
            #ifdef PRINT_DEBUG
                printf("  NEIGHBOR launching %d threads and %d blocks for %d pairs\n", nthreads, nblocks, V*N);
            #endif
            KernelNeighbourColor<<<nblocks, nthreads>>>(graph_d, *color, near_colors, V, work);      
    
            //find available colors
            if (N != 1) {
                nthreads = min(512, N);
                nblocks = ceil(float(N)/nthreads);
            
                #ifdef PRINT_DEBUG
                printf("  SEARCH launching %d threads and %d blocks for %d items\n", nthreads, nblocks, N);
                #endif
            
                KernelSearchColor<<<nblocks, nthreads>>>(*color, near_colors, V, N, work, V_start);
            } else {
                
                // no need to run GPU for one item
                #ifdef PRINT_DEBUG
                    printf("  SEARCH launching CPU for 1 item\n");
                #endif
                
                cudaError search_synced = cudaDeviceSynchronize();
                if (search_synced != cudaSuccess){
                    std::cout << "SEARCH_with_CPU cuda sync ERROR happened: " << cudaGetErrorName(search_synced) << std::endl;
                    exit(search_synced);
                }
                for (int ji = 0; ji < Nv; ji++){
                    if (work[ji]==-1) continue;
                    int index = work[ji] + V_start;
                    for (int clr = 1; clr < V; clr ++){
                        if (!near_colors[work[ji] * V + clr]){
                            (*color)[index] = clr;
                            break;
                        }
                    }
                }
            }

            //check if there are wrong colors
            if (N != 1) {
                nthreads = min(512, Nverticies);
                nblocks = ceil(float(Nverticies)/nthreads);
                
                #ifdef PRINT_DEBUG
                    printf("  CHECK launching %d threads and %d blocks for %d items\n", nthreads, nblocks, N);
                #endif
                
                KernelCheckColor<<<nblocks, nthreads>>>(graph_d, *color, V, work, work, V_start);
            } else {
                
                //no need to run GPU for one item
                #ifdef PRINT_DEBUG
                    printf("  CHECK launching CPU for 1 item\n");
                #endif
                
                //sync CUDA and CPU
                cudaError check_synced = cudaDeviceSynchronize();
                if (check_synced != cudaSuccess){
                    std::cout << "CHECK_with_CPU cuda sync ERROR happened: " << cudaGetErrorName(check_synced) << std::endl;
                    exit(check_synced);
                }
                for (int ji = 0; ji < Nv; ji++){
                    if (work[ji]==-1) continue;
                    int index = work[ji] + V_start;
                    work[ji] = -1;
                    for (int i = index + 1; i < V; i++) {
                        if (graph_h[index * V + i] and (*color)[i]==(*color)[index]) {
                            work[ji] = index - V_start;
                            break;
                        }
                    }
                }
            }

            //sync GPU with CPU for proper work[] access
            cudaError synced = cudaDeviceSynchronize();
                if (synced != cudaSuccess){
                    std::cout << "CYCLE_END cuda sync ERROR happened: " << cudaGetErrorName(synced) << std::endl;
                    exit(synced);
                }
    
            //check if done
            done = true;
            for(int i=0; i < V; i++){
                if (work[i] != -1){
                    done = false;
                    break;
                }
            }

        } //while not done
    } //for V_start
    cudaFree(near_colors);
    cudaFree(work);
    cudaFree(graph_d);

//analyze result
/////////////////////////////////////////////////
    printf("CUDA part ended, calculating number of colors\n");

    //color counter
    int num_colors = 0;
    bool seen_colors[V+1];
    for (int i = 0; i < V; i++) seen_colors[i] = false;

    std::cout << "Vertex - color" << std::endl;

    for (int i = 0; i < V; i++) {
       if (!seen_colors[(*color)[i]]) {
          seen_colors[(*color)[i]] = true;
          num_colors++;
       }  
    }

    //print result
#ifndef PRINTALL
    if (V < 100){
#endif
        for (int i = 0; i < V; i++) {
            std::cout << i << " - color " << (*color)[i] << std::endl;
        }
#ifndef PRINTALL
    } else {
        std::cout << "Too many verticies to print out, uncomment PRINTALL if need to print " << V << " verticies" << std::endl;
    }
#endif
    std::cout << "Solution has " << num_colors << " colors" << std::endl;
    cudaFree(*color);
}

/////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char const **argv)
{
    int* color;
    GraphColoringGPU(argv[1], &color);
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////
//parsers from the example

// Read MatrixMarket graphs
// Assumes input nodes are numbered starting from 1
int ReadMMFile(const char filename[], bool** graph, int* V) 
{
   std::string line;
   std::ifstream infile(filename);
   if (infile.fail()) {
      printf("Failed to open %s\n", filename);
      return -1;
   }

   // Reading comments
   while (getline(infile, line)) {          
      std::istringstream iss(line);
      if (line.find("%") == std::string::npos)
         break;
   }

   // Reading metadata
   std::istringstream iss(line);
   int num_rows, num_cols, num_edges;
   iss >> num_rows >> num_cols >> num_edges;

   *graph = new bool[num_rows * num_rows];
   memset(*graph, 0, num_rows * num_rows * sizeof(bool));
   *V = num_rows;

   // Reading nodes
   while (getline(infile, line)) {          
      std::istringstream iss(line);
      int node1, node2, weight;
      iss >> node1 >> node2 >> weight;
      
      // Assume node numbering starts at 1
      (*graph)[(node1 - 1) * num_rows + (node2 - 1)] = true;
      (*graph)[(node2 - 1) * num_rows + (node1 - 1)] = true;
   }
   infile.close();
   return 0;
}


// Read DIMACS graphs
// Assumes input nodes are numbered starting from 1
int ReadColFile(const char filename[], bool** graph, int* V) 
{
   std::string line;
   std::ifstream infile(filename);
   if (infile.fail()) {
      printf("Failed to open %s\n", filename);
      return -1;
   }

   int num_rows, num_edges;

   while (getline(infile, line)) {
      std::istringstream iss(line);
      std::string s;
      int node1, node2;
      iss >> s;
      if (s == "p") {
         iss >> s; // read string "edge"
         iss >> num_rows;
         iss >> num_edges;
         *V = num_rows;
         *graph = new bool[num_rows * num_rows];
         memset(*graph, 0, num_rows * num_rows * sizeof(bool));
         continue;
      } else if (s != "e")
         continue;
      
      iss >> node1 >> node2;

      // Assume node numbering starts at 1
      (*graph)[(node1 - 1) * num_rows + (node2 - 1)] = true;
      (*graph)[(node2 - 1) * num_rows + (node1 - 1)] = true;
   }
   infile.close();
   return 0;
}