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

//#define PRINTALL

//file parsers from example
void ReadColFile(const char filename[], bool** graph, int* V);
void ReadMMFile(const char filename[], bool** graph, int* V);

//////////////////////////////////////////////////////////////////////////////////////////////////

//run neighbor color for each pairs of verticies
__global__ void KernelNeighbourColor(bool* graph, int* colors, bool* output, int V, int* work){
    int work_index = floorf((blockIdx.x * blockDim.x + threadIdx.x)/V); //primary vertex selector from work list
    int near  = (blockIdx.x * blockDim.x + threadIdx.x) % V;           //neighbor vertex index         (col of graph)
    //int near_real = near + work_offset;
    int index = work[work_index];            //primary vertex index;
    //int index_real = index + work_offset;

    //stage 1. scan neighbour

    //find color for neighbour
    int color_near_r = 0;
    if (graph[index * V + near]){
        color_near_r = colors[near];
    }

    //stage 2. mark used colors
    if (color_near_r != 0){
        output[work_index * V + color_near_r] = true;
    }
/*debug*///if (work_index < 3) printf("NEIBOUR work %d vertex %d with %d, color %d\n", work_index,index, near, color_near_r);
}

//run search per each vertex
__global__ void KernelSearchColor(int* colors, bool* nearcolors, int V, int N, int* work, int work_offset){
    int work_index = blockIdx.x * blockDim.x + threadIdx.x; //work index
    if (work_index < N){
        int index = work[work_index];  //vertex index
        int index_real = index + work_offset;
        for (int clr = 1; clr < V; clr ++){
            if (!nearcolors[work_index * V + clr]){
                colors[index_real] = clr;
                break;
            }
        }
        if (work_index < 3) printf("SEARCH work %d vertex local %d, vertex real %d, color %d\n", work_index, index, index_real, colors[index_real]);
    }
}

//run check per each vertex
__global__ void KernelCheckColor(bool* graph, int* colors, int V, int* work, int* new_work, int work_offset){
    int work_index = blockIdx.x * blockDim.x + threadIdx.x; //work index           //neighbor vertex index         (col of graph)
    int index = work[work_index];            //primary vertex index;
    int index_real = index + work_offset;
    new_work[work_index] = -1; //default value
    for (int i = index + 1; i < V; i++) {
        if (graph[index * V + i] and colors[i + work_offset]==colors[index_real]) {
            new_work[work_index] = index;
            break;
        }
    }
    if (work_index < 3) printf("CHECK work %d vertex local %d, vertex real %d, new work %d\n", work_index, index, index_real,new_work[work_index]);
}

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
    if (std::string(filename).find(".col") != std::string::npos)
        ReadColFile(filename, &graph_h, &V);
    else if (std::string(filename).find(".mm") != std::string::npos) 
        ReadMMFile(filename, &graph_h, &V);
    else {
        //exit now, if cannot parse the file
        std::cout << "Cannot parse the file\n";
        return;
    }

    //read free memory
    size_t free, total;
    cudaMemGetInfo(&free,&total); 
    printf("\nGPU: %d KB free of total %d KB\n",free/1024,total/1024);

    //allocate list of colors per vector
    cudaError malloc_err = cudaMallocManaged(color, V * sizeof(int));
        if (malloc_err != cudaSuccess){
            std::cout << "COLOR_MALLOC cuda malloc ERROR happened: " << cudaGetErrorName(malloc_err) << std::endl;
            exit(malloc_err);
        }
    for (int i=0; i < V; i++){
        (*color)[i] = 0;
    }

    //find memory devision
    //                                      VV leave 20MB free
    int Nverticies = min(V, int(floor((free -20*1024*1024 - 4 * V)/(2*V + 4)))); // number of verticies per one full-memory alocation
/*debug*/ Nverticies = 100;
    int Nparts = ceil(V/Nverticies);

    //work for GPU (indicies of verticies to process)
    int* work;
    bool* near_colors;
    malloc_err = cudaMallocManaged(&work, Nverticies * sizeof(int));
        if (malloc_err != cudaSuccess){
            std::cout << "JOB_MALLOC cuda malloc ERROR happened: " << cudaGetErrorName(malloc_err) << std::endl;
            exit(malloc_err);
        }
    malloc_err = cudaMalloc(&graph_d, V * Nverticies * sizeof(bool));
        if (malloc_err != cudaSuccess){
            std::cout << "GRAPH_MALLOC cuda malloc ERROR happened: " << cudaGetErrorName(malloc_err) << std::endl;
            exit(malloc_err);
        }
    malloc_err = cudaMallocManaged(&near_colors, V * Nverticies * sizeof(bool));
        if (malloc_err != cudaSuccess){
            std::cout << "NEAR_MALLOC cuda malloc ERROR happened: " << cudaGetErrorName(malloc_err) << std::endl;
            exit(malloc_err);
        }

    //SEGMENT graph_d FROM HERE
////////////////////////////////////////////
    //for (int V_start = 0; V_start < V; V_start += Nverticies){
    //go from last part to first
    for (int V_start_r = V - Nverticies; V_start_r > -Nverticies; V_start_r -= Nverticies){
        //start should not be below 0
        int V_start = max(0, V_start_r);
        //actual amount of verticies that awailable for proccessing
        int Nv = min(Nverticies - V_start + V_start_r, V - V_start);

        //fill work
        for (int i = 0; i < Nv; i++){
            work[i] = i;
        }
        bool done = false;

        //move graph to device memory
        cudaMemcpy(graph_d, graph_h + V_start * V, V * Nv * sizeof(bool), cudaMemcpyHostToDevice);
    
        //repeat until find solition
        int N;
        int D = 0;
        while (!done){

            //sort work list and count amount of work
            N = 0;
                //int carry = 0;
            for (int j=0; j < Nv; j++){
                    if (work[j] != -1) {
                        if (j != N){
                            printf("move work for vertex %d from %d to %d\n", work[j], j, N);
                            work[N] = work[j];
                            work[j] = -1;
                        }
                        N++;
                    }
            }
            D++;
            printf("====while loop, %d verticies need processing; part from V %d with %d verticies; iteration:%d\n", N, V_start, Nv,D);
            if (N == 0) break;
    
    /*debug*/// for (int a=0; a<V; a++)
    /*debug*///     if (work[a]!=-1)
    /*debug*///         std::cout << "    work " << a << ": " << work[a] << " color: " << (*color)[work[a]] << "\n";
    /*debug*///     else std::cout << "    work " << a << ": " << work[a] << "\n";
    

            int nthreads = min(512, V*Nverticies);
            int nblocks = ceil(V * Nverticies / nthreads);
            KernelBoolClear<<<nthreads, nblocks>>>(near_colors, Nverticies);

            nthreads = min(512, V*N);
            nblocks = ceil(float(V*N)/nthreads);
/*debug info*/printf("  NEIGHBOR launching %d threads and %d blocks for %d pairs\n", nthreads, nblocks, V*N);
            KernelNeighbourColor<<<nblocks, nthreads>>>(graph_d, *color, near_colors, V, work);
            //sync CUDA and CPU
            //cudaError synced = cudaDeviceSynchronize();
            //    if (synced != cudaSuccess){
            //        std::cout << "COLOR_NEARBY cuda sync ERROR happened: " << cudaGetErrorName(synced) << std::endl;
            //        exit(synced);
            //    }
    
    /*debug*/// for (int r=0; r < N; r++){
    /*debug*///   printf("    near V %d: ",work[r]);
    /*debug*///   for (int c=0; c < V; c++)
    /*debug*///     printf(" %d",near_colors[r*V+c]);
    /*debug*///   printf("\n");
    /*debug*/// }        
    
            //find colors
            if (N != 1) {
                nthreads = min(512, N);
                nblocks = ceil(float(N)/nthreads);
    /*debug info*/printf("  SEARCH launching %d threads and %d blocks for %d items\n", nthreads, nblocks, N);
                KernelSearchColor<<<nblocks, nthreads>>>(*color, near_colors, V, N, work, V_start);
                //sync CUDA and CPU
                cudaError synced = cudaDeviceSynchronize();
                    if (synced != cudaSuccess){
                        std::cout << "SEARCH_COLOR cuda sync ERROR happened: " << cudaGetErrorName(synced) << std::endl;
                        exit(synced);
                    }
            } else {
    /*debug info*/printf("  SEARCH launching CPU for 1 item\n");
                cudaError search_synced = cudaDeviceSynchronize();
                if (search_synced != cudaSuccess){
                    std::cout << "SEARCH_with_CPU cuda sync ERROR happened: " << cudaGetErrorName(search_synced) << std::endl;
                    exit(search_synced);
                }
                for (int ji = 0; ji < Nv; ji++){
                    if (work[ji]==-1) continue;
                    int index = work[ji] + V_start;
                    for (int clr = 1; clr < V; clr ++){
                        if (!near_colors[clr]){
                            (*color)[index] = clr;
                            break;
                        }
                    }
                }
            }
    
    /*debug*/// for (int c=0; c<V; c++){
    /*debug*///     printf("    V %d - color %d\n", c, (*color)[c]);
    /*debug*/// }

//            nthreads = min(512, V*Nverticies);
//            nblocks = ceil(V * Nverticies / nthreads);
//            KernelBoolClear<<<nthreads, nblocks>>>(near_colors, Nverticies);
//    
//            nthreads = min(512, V*N);
//            nblocks = ceil(float(V*N)/nthreads);
///*debug info*/printf("  NEIGHBOR launching %d threads and %d blocks for %d pairs\n", nthreads, nblocks, V*N);
//            KernelNeighbourColor<<<nblocks, nthreads>>>(graph_d, *color, near_colors, V, work);
            //sync CUDA and CPU
            //synced = cudaDeviceSynchronize();
            //    if (synced != cudaSuccess){
            //        std::cout << "COLOR_NEARBY_CHECK cuda sync ERROR happened: " << cudaGetErrorName(synced) << std::endl;
            //        exit(synced);
            //    }
    
    /*debug*/// for (int r=0; r < N; r++){
    /*debug*///   printf("    near V %d: ",work[r]);
    /*debug*///   for (int c=0; c < V; c++)
    /*debug*///     printf(" %d",near_colors[r*V+c]);
    /*debug*///   printf("\n");
    /*debug*/// }

            if (N != 1) {
                nthreads = min(512, Nverticies);
                nblocks = ceil(float(Nverticies)/nthreads);
    /*debug info*/printf("  CHECK launching %d threads and %d blocks for %d items\n", nthreads, nblocks, N);
                KernelCheckColor<<<nblocks, nthreads>>>(graph_d, *color, V, work, work, V_start);
                //sync CUDA and CPU
            } else {
    /*debug info*/printf("  CHECK launching CPU for 1 item\n");
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
                        if (graph_h[index * V + i] and (*color)[i + V_start]==(*color)[index]) {
                            work[ji] = index - V_start;
                            break;
                        }
                    }
                }
            }
            
    
/*debug*/   if (N < 10) {
/*debug*/       cudaError check_synced = cudaDeviceSynchronize();
/*debug*/       if (check_synced != cudaSuccess){
/*debug*/           std::cout << "CHECK_DEBUG cuda sync ERROR happened: " << cudaGetErrorName(check_synced) << std::endl;
/*debug*/           exit(check_synced);
/*debug*/       }
/*debug*/       for (int a=0; a<Nverticies; a++)
/*debug*/           if (work[a]!=-1){
/*debug*/               std::cout << "    work " << a << ": " << work[a] << " color: " << (*color)[work[a]] << " near:";
                        for (int c=0; c < V; c++){
                            std::cout << (int)near_colors[work[a] * V + c];
                        }
                        std::cout << "\n";
                    }
/*debug*/           //else std::cout << "    work " << a << ": " << work[a] << "\n";
            }

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