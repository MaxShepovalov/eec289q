//graph coloring
#include "stdio.h"
#include <iostream>
#include <math.h>

//file parcers from example
void ReadColFile(const char filename[], bool** graph, int* V);
void ReadMMFile(const char filename[], bool** graph, int* V);

__global__ void KroneckerKernelSmall(int M, int N, float* A, float* B, float* C);

void GraphColoringGPU(const char filename[], int** color);

void int main(int argc, char const *argv[])
{
    /* code */
    std::cout << argc << " items " << std::endl;
    for (int i = 0; i < argc; i++){
        std::cout << i << ": '" << argv[argc] << "'" << std::endl;
    }
    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////
// Read MatrixMarket graphs
// Assumes input nodes are numbered starting from 1
void ReadMMFile(const char filename[], bool** graph, int* V) 
{
   string line;
   ifstream infile(filename);
   if (infile.fail()) {
      printf("Failed to open %s\n", filename);
      return;
   }

   // Reading comments
   while (getline(infile, line)) {          
      istringstream iss(line);
      if (line.find("%") == string::npos)
         break;
   }

   // Reading metadata
   istringstream iss(line);
   int num_rows, num_cols, num_edges;
   iss >> num_rows >> num_cols >> num_edges;

   *graph = new bool[num_rows * num_rows];
   memset(*graph, 0, num_rows * num_rows * sizeof(bool));
   *V = num_rows;

   // Reading nodes
   while (getline(infile, line)) {          
      istringstream iss(line);
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
   string line;
   ifstream infile(filename);
   if (infile.fail()) {
      printf("Failed to open %s\n", filename);
      return;
   }

   int num_rows, num_edges;

   while (getline(infile, line)) {
      istringstream iss(line);
      string s;
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