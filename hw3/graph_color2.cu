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

//////////////////////////////////////////////////////////////////////////////////////////////////

//run neighbor color for each pairs of verticies
__global__ void KernelNeighbourColor(bool* graph, int* colors, bool* output, int V, int* job){
    int job_index = floorf((blockIdx.x * blockDim.x + threadIdx.x)/V); //primary vertex selector from job list
    int near  = (blockIdx.x * blockDim.x + threadIdx.x) % V;           //neighbor vertex index         (col of graph)
    //int near_real = near + job_offset;
    int index = job[job_index];            //primary vertex index;
    //int index_real = index + job_offset;

    //stage 1. scan neighbour

    //find color for neighbour
    int color_near_r = 0;
    if (graph[index * V + near]){
        color_near_r = colors[near];
    }

    //stage 2. mark used colors
    if (color_near_r != 0){
        output[job_index * V + color_near_r] = true;
    }
}

//run search per each vertex
__global__ void KernelSearchColor(int* colors, bool* nearcolors, int V, int N, int* job, int job_offset){
    int job_index = blockIdx.x * blockDim.x + threadIdx.x; //job index
    if (job_index < N){
        int index = job[job_index];  //vertex index
        int index_real = index + job_offset;
        for (int clr = 1; clr < V; clr ++){
            if (!nearcolors[job_index * V + clr]){
                colors[index_real] = clr;
                break;
            }
        }
    }
}

//run check per each vertex
__global__ void KernelCheckColor(bool* graph, int* colors, int V, int* job, int* new_job, int job_offset){
    //int job_index = floorf((blockIdx.x * blockDim.x + threadIdx.x)/V); //primary vertex selector from job list
    int job_index = blockIdx.x * blockDim.x + threadIdx.x; //job index
    //int near  = (blockIdx.x * blockDim.x + threadIdx.x) % V;           //neighbor vertex index         (col of graph)
    int index = job[job_index];            //primary vertex index;
    int index_real = index + job_offset;
    //need scatter to gather
    new_job[job_index] = -1; //default value
    for (int i = index + 1; i < V; i++) {
        if (graph[index * V + i] and colors[i + job_offset]==colors[index_real]) {
            new_job[job_index] = index;
            break;
        }
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

    //job for GPU (indexes of verticies to process)
    int* job;
    malloc_err = cudaMallocManaged(&job, Nverticies * sizeof(int));
        if (malloc_err != cudaSuccess){
            std::cout << "JOB_MALLOC cuda malloc ERROR happened: " << cudaGetErrorName(malloc_err) << std::endl;
            exit(malloc_err);
        }
    malloc_err = cudaMalloc(&graph_d, V * Nverticies * sizeof(bool));
        if (malloc_err != cudaSuccess){
            std::cout << "GRAPH_MALLOC cuda malloc ERROR happened: " << cudaGetErrorName(malloc_err) << std::endl;
            exit(malloc_err);
        }

    //SEGMENT graph_d FROM HERE
////////////////////////////////////////////
    for (int V_start = 0; V_start < V; V_start += Nverticies){
        //actual amount of verticies that awailable for proccessing
        int Nv = min(Nverticies, V - V_start);
    /*debug*/ printf("part from V %d with %d verticies\n", V_start, Nv);

        //fill job
        for (int i = 0; i < Nv; i++){
            job[i] = i;
    /*debug*/// std::cout << "job " << i << " now is " << job[i] << "\n";
        }
        bool done = false;
        
        //start kernel

        //move graph to device memory
        cudaMemcpy(graph_d, graph_h + V_start * V, V * Nv * sizeof(bool), cudaMemcpyHostToDevice);
    
        //repeat until find solition
    /*debug*/// int D = 0;
        int N;
        while (!done){
            //sort job list and count amount of job
            N = 0;
            for (int j=0; j < Nv; j++){
                if (job[j] == -1){
                    for(int jj=j; jj < V; jj++){
                        if (job[jj]!=-1){
                            job[jj-1] = job[jj];
                            job[jj] = -1;
                        }
                    }
                }
                //cannot put `else`, as if job[j] == -1, the array will be different at this point
                if (job[j] != -1) N++;
            }
    
    /*debug*/// for (int a=0; a<V; a++)
    /*debug*///     if (job[a]!=-1)
    /*debug*///         std::cout << "    job " << a << ": " << job[a] << " color: " << (*color)[job[a]] << "\n";
    /*debug*///     else std::cout << "    job " << a << ": " << job[a] << "\n";
    
            //check colors nearby " << N << " verticies
            bool* near_colors;
            cudaError malloc_err = cudaMallocManaged(&near_colors, V * N * sizeof(bool));
                if (malloc_err != cudaSuccess){
                    std::cout << "NEAR_MALLOC cuda malloc ERROR happened: " << cudaGetErrorName(malloc_err) << std::endl;
                    exit(malloc_err);
                }
    
            for (int i=0; i < V*N; i++)
                near_colors[i] = false;
    
            int nthreads = min(512, V*N);
            int nblocks = ceil(float(V*N)/nthreads);
        printf("  NEIGHBOR launching %d threads and %d blocks for %d pairs\n", nthreads, nblocks, V*N);
            KernelNeighbourColor<<<nblocks, nthreads>>>(graph_d, *color, near_colors, V, job);
            //sync CUDA and CPU
            //cudaError synced = cudaDeviceSynchronize();
            //    if (synced != cudaSuccess){
            //        std::cout << "COLOR_NEARBY cuda sync ERROR happened: " << cudaGetErrorName(synced) << std::endl;
            //        exit(synced);
            //    }
    
    /*debug*/// for (int r=0; r < N; r++){
    /*debug*///   printf("    near V %d: ",job[r]);
    /*debug*///   for (int c=0; c < V; c++)
    /*debug*///     printf(" %d",near_colors[r*V+c]);
    /*debug*///   printf("\n");
    /*debug*/// }        
    
            //find colors
            if (N != 1) {
                nthreads = min(512, N);
                nblocks = ceil(float(N)/nthreads);
            printf("  SEARCH launching %d threads and %d blocks for %d items\n", nthreads, nblocks, N);
                KernelSearchColor<<<nblocks, nthreads>>>(*color, near_colors, V, N, job, V_start);
                //sync CUDA and CPU
                cudaError synced = cudaDeviceSynchronize();
                    if (synced != cudaSuccess){
                        std::cout << "SEARCH_COLOR cuda sync ERROR happened: " << cudaGetErrorName(synced) << std::endl;
                        exit(synced);
                    }
            } else {
            printf("  SEARCH launching CPU for 1 item\n");
                cudaError search_synced = cudaDeviceSynchronize();
                if (search_synced != cudaSuccess){
                    std::cout << "SEARCH_with_CPU cuda sync ERROR happened: " << cudaGetErrorName(search_synced) << std::endl;
                    exit(search_synced);
                }
                int index = job[0] + V_start;
                for (int clr = 1; clr < V; clr ++){
                    if (!near_colors[clr]){
                        (*color)[index] = clr;
                        break;
                    }
                }
            }
    
            //cudaFree(near_colors);
    
    /*debug*/// for (int c=0; c<V; c++){
    /*debug*///     printf("    V %d - color %d\n", c, (*color)[c]);
    /*debug*/// }
            
            //check if need to work again (update `near_colors`)
            //malloc_err = cudaMallocManaged(&near_colors, V * N * sizeof(bool));
            //    if (malloc_err != cudaSuccess){
            //        std::cout << "NEAR_COLORS_2 cuda malloc ERROR happened: " << cudaGetErrorName(malloc_err) << std::endl;
            //        exit(malloc_err);
            //    }
    
            for (int i=0; i < V*N; i++)
                near_colors[i] = false;
    
            nthreads = min(512, V*N);
            nblocks = ceil(float(V*N)/nthreads);
        printf("  NEIGHBOR launching %d threads and %d blocks for %d pairs\n", nthreads, nblocks, V*N);
            KernelNeighbourColor<<<nblocks, nthreads>>>(graph_d, *color, near_colors, V, job);
            //sync CUDA and CPU
            //synced = cudaDeviceSynchronize();
            //    if (synced != cudaSuccess){
            //        std::cout << "COLOR_NEARBY_CHECK cuda sync ERROR happened: " << cudaGetErrorName(synced) << std::endl;
            //        exit(synced);
            //    }
    
    /*debug*/// for (int r=0; r < N; r++){
    /*debug*///   printf("    near V %d: ",job[r]);
    /*debug*///   for (int c=0; c < V; c++)
    /*debug*///     printf(" %d",near_colors[r*V+c]);
    /*debug*///   printf("\n");
    /*debug*/// }
    
            //update job
            //int* new_job;
            //cudaMallocManaged(&new_job, V * sizeof(bool));
            //for (int j=0; j < V; j++){
            //    new_job[j] = -1;
            //}
            if (N != 1) {
                nthreads = min(512, N);
                nblocks = ceil(float(N)/nthreads);
            printf("  CHECK launching %d threads and %d blocks for %d items\n", nthreads, nblocks, N);
                KernelCheckColor<<<nblocks, nthreads>>>(graph_d, *color, V, job, job, V_start);
                //KernelCheckColor<<<nblocks, nthreads>>>(graph_d, *color, V, job, new_job);
                //sync CUDA and CPU
            } else {
            printf("  CHECK launching CPU for 1 item\n");
                cudaError check_synced = cudaDeviceSynchronize();
                if (check_synced != cudaSuccess){
                    std::cout << "CHECK_with_CPU cuda sync ERROR happened: " << cudaGetErrorName(check_synced) << std::endl;
                    exit(check_synced);
                }
                int index = job[0] + V_start;
                job[0] = -1;
                for (int i = index + 1; i < V; i++) {
                    if (graph_h[index * V + i] and (*color)[i + V_start]==(*color)[index]) {
                        job[0] = index - V_start;
                        break;
                    }
                }
            }
            
    
    /*debug*/// for (int a=0; a<V; a++)
    /*debug*///     if (new_job[a]!=-1)
    /*debug*///         std::cout << "    new_job " << a << ": " << new_job[a] << " color: " << (*color)[new_job[a]] << "\n";
    /*debug*///     else std::cout << "    new_job " << a << ": " << new_job[a] << "\n";
            
            //swap job lists
            //cudaFree(old_job);
            //cudaFree(job);
            //job = new_job;

            cudaError synced = cudaDeviceSynchronize();
                if (synced != cudaSuccess){
                    std::cout << "CYCLE_END cuda sync ERROR happened: " << cudaGetErrorName(synced) << std::endl;
                    exit(synced);
                }
            cudaFree(near_colors);
    
            //check if done
            done = true;
            for(int i=0; i < V; i++){
                if (job[i] != -1){
    /*debug*///       printf("Need to work more\n");
                    done = false;
                    break;
                }
            }
    /*debug*/// D++;
    /*debug*/// if (D == 4) done = true;

        } //while not done

    } //for V_start
    cudaFree(job);
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