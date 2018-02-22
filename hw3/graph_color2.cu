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

//Kernel work with pairs of vertexes
__global__ void KernelNeighbourColor(bool* graph, int* colors, bool* output, int V, int* job){
    int job_index = floorf(threadIdx.x/V); //primary vertex selector from job list
    int near  = threadIdx.x % V;           //neighbor vertex index         (col of graph)
    int index = job[job_index];            //primary vertex index;

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

__global__ void KernelSearchColor(int* colors, bool* nearcolors, int V, int* job){
    int job_index = threadIdx.x; //job index
    int index = job[job_index];  //vertex index
    for (int clr = 1; clr < V; clr ++){
        if (!nearcolors[job_index * V + clr]){
            colors[index] = clr;
            break;
        }
    }
}

__global__ void KernelCheckColor(bool* graph, int* colors, int V, int* job, int* new_job){
    int job_index = floorf(threadIdx.x/V); //primary vertex selector from job list
    int near  = threadIdx.x % V;           //neighbor vertex index         (col of graph)
    int index = job[job_index];            //primary vertex index;
    if (graph[index * V + near]){
        if (colors[near] == colors[index] and near > index){
            new_job[job_index] = index;
        }
    }
}

void GraphColoringGPU(const char filename[], int** color){
    int V;         //number of vertexes
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

    //allocate list of colors per vector
    cudaError malloc_err = cudaMallocManaged(color, V * sizeof(int));
        if (malloc_err != cudaSuccess){
            std::cout << "COLOR_MALLOC cuda sync ERROR happened: " << cudaGetErrorName(malloc_err) << std::endl;
            exit(malloc_err);
        } else {
            std::cout << "COLOR_MALLOC OK\n";
        }
    for (int i=0; i < V; i++){
        (*color)[i] = 0;
    }

    //move graph to device memory
    malloc_err = cudaMalloc(&graph_d, V * V * sizeof(bool));
        if (malloc_err != cudaSuccess){
            std::cout << "GRAPH_MALLOC cuda sync ERROR happened: " << cudaGetErrorName(malloc_err) << std::endl;
            exit(malloc_err);
        } else {
            std::cout << "GRAPH_MALLOC OK\n";
        }
    cudaMemcpy(graph_d, graph_h, V * V * sizeof(bool), cudaMemcpyHostToDevice);
    
    //job for GPU (indexes of vertexes to process)
    int* job;
    malloc_err = cudaMallocManaged(&job, V * sizeof(int));
        if (malloc_err != cudaSuccess){
            std::cout << "JOB_MALLOC cuda sync ERROR happened: " << cudaGetErrorName(malloc_err) << std::endl;
            exit(malloc_err);
        } else {
            std::cout << "JOB_MALLOC OK\n";
        }
    //fill job
    for (int i=0; i < V; i++){
        job[i] = i;
/*debug*/// std::cout << "job " << i << " now is " << job[i] << "\n";
    }
    bool done = false;
    
    //start kernel

    //repeat until find solition
/*debug*/// int D = 0;
    while (!done){
        //sort job list and count amount of job
        int N = 0;
        for (int j=0; j < V; j++){
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

        //check colors nearby " << N << " vertexes

        bool* near_colors;
        cudaMallocManaged(&near_colors, V * N * sizeof(bool));
        for (int i=0; i < V*N; i++)
            near_colors[i] = false;

        KernelNeighbourColor<<<1, V*N>>>(graph_d, *color, near_colors, V, job);
        //sync CUDA and CPU
        cudaError synced = cudaDeviceSynchronize();
        if (synced != cudaSuccess){
            std::cout << "COLOR_NEARBY cuda sync ERROR happened: " << cudaGetErrorName(synced) << std::endl;
            exit(synced);
        }

/*debug*/// for (int r=0; r < N; r++){
/*debug*///   printf("    near V %d: ",job[r]);
/*debug*///   for (int c=0; c < V; c++)
/*debug*///     printf(" %d",near_colors[r*V+c]);
/*debug*///   printf("\n");
/*debug*/// }        

        //find colors

        KernelSearchColor<<<1, N>>>(*color, near_colors, V, job);
        //sync CUDA and CPU
        synced = cudaDeviceSynchronize();
        if (synced != cudaSuccess){
            std::cout << "FIND_COLOR cuda sync ERROR happened: " << cudaGetErrorName(synced) << std::endl;
            exit(synced);
        }
        cudaFree(near_colors);

/*debug*/// for (int c=0; c<V; c++){
/*debug*///     printf("    V %d - color %d\n", c, (*color)[c]);
/*debug*/// }
        
        //check if need to work again (update `near_colors`)
        cudaMallocManaged(&near_colors, V * N * sizeof(bool));
        for (int i=0; i < V*N; i++)
            near_colors[i] = false;

        KernelNeighbourColor<<<1, V*N>>>(graph_d, *color, near_colors, V, job);
        //sync CUDA and CPU
        synced = cudaDeviceSynchronize();
        if (synced != cudaSuccess){
            std::cout << "COLOR_NEARBY_CHECK cuda sync ERROR happened: " << cudaGetErrorName(synced) << std::endl;
            exit(synced);
        }

/*debug*/// for (int r=0; r < N; r++){
/*debug*///   printf("    near V %d: ",job[r]);
/*debug*///   for (int c=0; c < V; c++)
/*debug*///     printf(" %d",near_colors[r*V+c]);
/*debug*///   printf("\n");
/*debug*/// }

        //update job
        int* new_job;
        cudaMallocManaged(&new_job, V * sizeof(bool));
        for (int j=0; j < V; j++){
            new_job[j] = -1;
        }

        KernelCheckColor<<<1 , V*N>>>(graph_d, *color, V, job, new_job);
        //sync CUDA and CPU
        synced = cudaDeviceSynchronize();
        if (synced != cudaSuccess){
            std::cout << "JOB_UPDATE cuda sync ERROR happened: " << cudaGetErrorName(synced) << std::endl;
            exit(synced);
        }
        cudaFree(near_colors);

/*debug*/// for (int a=0; a<V; a++)
/*debug*///     if (new_job[a]!=-1)
/*debug*///         std::cout << "    new_job " << a << ": " << new_job[a] << " color: " << (*color)[new_job[a]] << "\n";
/*debug*///     else std::cout << "    new_job " << a << ": " << new_job[a] << "\n";
        
        //swap job lists
        //cudaFree(old_job);
        cudaFree(job);
        job = new_job;

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
    }
    cudaFree(job);

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