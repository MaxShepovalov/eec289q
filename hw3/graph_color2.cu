//graph coloring

//CUDA accelerator approach

//color management is happeing on CPU, while for loops are parralellized on GPU

//KernelNeighbourColor(graph_line, colors, output, Vsize);
//Kernel
#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <math.h>

//file parsers from example
void ReadColFile(const char filename[], bool** graph, int* V);
void ReadMMFile(const char filename[], bool** graph, int* V);

////////////////////////////////////////////////////////////////////////////////////

// __device__ int DeviceVertexProcess(bool* graph, int* colors, int Vsize, int V1, int V2){
//     //graph - pointer on graph
//     //colors - pointer on current selected colors
//     //Vsize - amount of vertexes in graph
//     //V1 - main vertex index
//     //V2 - vertex index to compare with
    
//     //filtered color
//     int out = 0;
    
//     //if vertexes are connected
//     if (graph[V1 * V + V2]){
//         out = colors[V2];
//     }
    
//     //retrn result
//     return out;
// }


//Kernel work with pairs of vertexes
__global__ void KernelNeighbourColor(bool* graph, int* colors, bool* output, int V){
    int index = floorf(threadIdx.x/V); // primary vertex index (row of graph)
    int near  = threadIdx.x % V//neighbor vertex index         (col of graph)

    //stage 1. scan neighbour

    //find color for neighbour
    //int color_near_r = DeviceVertexProcess(graph, colors, V, V1, index);
    int color_near_r = 0;
    if (graph[index * V + near]){
        color_near_r = colors[near];
    }

    //write color
    //color_near[index] = color_near_r;
    //__syncthreads();

    //stage 2. mark used colors
    if (color_near_r != 0){
        output[index * V + color_near_r] = true;
    }
}

__global__ void KernelSearchColor(int* colors, int* nearcolors, int V){
    int index = threadIdx.x; //vertex index
    for (int clr = 0; clr < V; clr ++){
        if (!nearcolors[index * V + clr]){
            colors[index] = clr;
            break;
        }
    }
}

// __global__ void GraphKernel(bool* graph, int* color, int V) {
//     int index = blockIdx.x * blockDim.x + threadIdx.x; //thread ID
// //    int stride = blockDim.x * gridDim.x;               //
//     //each thread works with only one vertex

//     //shared memory for final colors
//     extern __shared__ int color_sh[];
//     color_sh[index] = 0;
//     __syncthreads();

//     //decide the color
//     for (int attempt = 0; attempt < V; attempt++) {

//         //scan colors of neighbours
//         bool* near = new bool[V+1];
//         for (int i = 0; i < V; i++) near[i] = false;

//         for (int i = 0; i < V; i++) {
//             if (graph[index * V + i] and i != index) {
//                 //near.insert(color_sh[i]);
//                 near[color_sh[i]] = true;
//             }
//         }

//         //select color
//         for (int color_i = 1; color_i < V; color_i++) {
//             if (!near[color_i]) {
//                 color_sh[index] = color_i;
//                 break;
//             }
//         }

//         //wait for others
//         __syncthreads();
        
//         //check if there is a mistake
//         bool done = true;
//         for (int i = index + 1; i < V; i++) {
//             if (graph[index * V + i] and color_sh[i]==color_sh[index]) {
//                 done = false;
//                 break;
//             }
//         }
//         if (done) {
//             //exit loop
//             break;
//         }
//     }
//
//     //write out result
//     color[index] = color_sh[index];
// }

void GraphColoringGPU(const char filename[], int** color){
    int V;         //number of vertexes
    bool* graph_h; //graph matrix on host
    bool* graph_d; //graph matrix on device
    //int* color_d;  //colors on device

    //read graph file
    if (std::string(filename).find(".col") != std::string::npos)
        ReadColFile(filename, &graph_h, &V);
    else if (std::string(filename).find(".mm") != std::string::npos) 
        ReadMMFile(filename, &graph_h, &V);
    else
        //exit now, if cannot parse the file
        return;

    //allocate list of colors per vector
    cudaMallocManaged(color, V * sizeof(int));

    //move graph to device memory
    cudaMalloc((bool**)&graph_d, V * V * sizeof(bool));
    cudaMemcpy(graph_d, graph_h, V * V * sizeof(bool), cudaMemcpyHostToDevice);
    
    //start kernel
    //int nblocks = 1;
    //int nthreads = V;
    //GraphKernel<<<nblocks, nthreads, V * sizeof(bool)>>>(graph_d, color_d, V);
    //GraphKernel<<<nblocks, nthreads, V * sizeof(bool)>>>(graph_d, *color, V);
    for (int vi = 0; vi < V; vi++){
        bool* near_colors;
        cudaMallocManaged(&near_colors, V * V * sizeof(bool));
        KernelNeighbourColor<<<1, V*V>>>(graph_d, *color, near_colors, V, vi);

        //find colors

        KernelSearchColor<<<1, V>>>(*color, near_colors, V);
        cudaFree(near_colors);
        
        //sync CUDA and CPU
        cudaError synced = cudaDeviceSynchronize();
        if (synced != cudaSuccess){
            std::cout << "cuda sync ERROR happened: " << cudaGetErrorName(synced) << std::endl;
            exit(synced);
        }
        // else {
        //   std::cout << "cuda sync OK" << std::endl;
        // }
    }

    //counter
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
    for (int i = 0; i < V; i++) {
        std::cout << i << " - color " << (*color)[i] << std::endl;
    }
    std::cout << "Solution has " << num_colors << " colors" << std::endl;
    cudaFree(*color);
}

/////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char const **argv)
{
    /* code */
    // std::cout << argc << " items " << std::endl;
    // for (int i = 0; i < argc; i++){
    //     std::cout << i << ": '" << argv[i] << "'" << std::endl;
    // }
    int* color;
    GraphColoringGPU(argv[1], &color);
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////
// Read MatrixMarket graphs
// Assumes input nodes are numbered starting from 1
void ReadMMFile(const char filename[], bool** graph, int* V) 
{
   std::string line;
   std::ifstream infile(filename);
   if (infile.fail()) {
      printf("Failed to open %s\n", filename);
      return;
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
}


// Read DIMACS graphs
// Assumes input nodes are numbered starting from 1
void ReadColFile(const char filename[], bool** graph, int* V) 
{
   std::string line;
   std::ifstream infile(filename);
   if (infile.fail()) {
      printf("Failed to open %s\n", filename);
      return;
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
}