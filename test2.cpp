#include <mpi.h>
#include <iostream>
#include "Algeon_matrix.hpp"
#include "Algeon_vectors.hpp"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (size < 2)
    {
        if (rank == 0)
            std::cerr << "Ce programme nécessite au moins 2 processus MPI (1 master + 1 worker).\n";
        MPI_Finalize();
        return 1;
    }

    std::size_t N = 6; // Taille globale

    ParallelMatrix<double> A(comm, N, N);
    ParallelVector<double> U(comm, N), V(comm, N);

    if (rank == 0)
    {
        // Matrice A tridiagonale
        for (std::size_t i = 0; i < N; ++i)
        {
            A[i][i] = 2;
            U[i] = 1;
            V[i] = 1;
        }
        for (std::size_t i = 0; i < N - 1; ++i)
        {
            A[i + 1][i] = 1;
            A[i][i + 1] = 1;
        }
    }

    // Distribution de la matrice et du vecteur
    A.distribute_from_master();
    U.distribute_from_master();
    V.distribute_from_master();

    // ───────────────────────────────────────
    // Produit A * V (en interne CSR + V.prepare)
    // ───────────────────────────────────────
    ParallelVector<double> Y = multiply_csr(A, V);

    Y.display();

    MPI_Finalize();
    return 0;
}
