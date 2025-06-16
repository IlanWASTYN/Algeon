#include <mpi.h>
#include <iostream>
#include "Algeon_vectors.hpp"

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
        MPI_Finalize();
        return 1;
    }

    // ───────────────────────────────────────
    // Définition des vecteurs
    // ───────────────────────────────────────
    std::size_t N = 10; // Taille globale

    ParallelVector<double> a(comm, N);
    ParallelVector<double> b(comm, N);

    // Initialisation des données locales (par les workers)

    a.fill(3.0); // Tous les éléments = 3.0
    b.fill(1.5); // Tous les éléments = 1.5

    a.print_local_data_per_proc("vecteur A");
    b.print_local_data_per_proc("vecteur B");
    MPI_Finalize();
    return 0;
}
