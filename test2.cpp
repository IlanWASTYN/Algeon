#include <mpi.h>
#include <iostream>
#include "Algeon_matrix.hpp"

int main(int argc, char **argv)
{

    // ───────────────────────────────────────
    // Initialisation MPI
    // ───────────────────────────────────────
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // ───────────────────────────────────────
    // Vérification du nombre de processus
    // ───────────────────────────────────────

    if (size < 2)
    {
        if (rank == 0)
            std::cerr << "Ce programme nécessite au moins 2 processus MPI (1 master + 1 worker).\n";
        
    }

    // ───────────────────────────────────────
    // Définition des matrices
    // ───────────────────────────────────────

    std::size_t N = 10; // Taille globale

    ParallelMatrix<double> A(comm, N, N); // 3x3 matrix initialisée à 0.0

    A.display("Matrice A");               // Affiche toute la matrice

    MPI_Finalize();

    return 0;
}