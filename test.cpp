#include <mpi.h>
#include <iostream>
#include "Algeon_vectors.hpp"

int main(int argc, char **argv)
{
    // ───────────────────────────────────────
    // Initialisation MPI
    // ───────────────────────────────────────
    MPI_Init(&argc, &argv);

    MPI_Comm comm_ = MPI_COMM_WORLD;
    int rank_, size_;
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &size_);

    // ───────────────────────────────────────
    // Vérification du nombre de processus
    // ───────────────────────────────────────

    if (size_ < 2)
    {
        if (rank_ == 0)
            std::cerr << "Ce programme nécessite au moins 2 processus MPI (1 master + 1 worker).\n";
        MPI_Finalize();

        return 1;
    }

    // ───────────────────────────────────────
    // Définition des vecteurs
    // ───────────────────────────────────────
    std::size_t N = 4; // Taille globale

    ParallelVector<double> a(comm_, N);
    ParallelVector<double> b(comm_, N);

    // Initialisation des données locales (par les workers)

    if (rank_ == 0)
    {
        for (int i = 0; i < N; i++)
        {
            a.local_data()[i] = i + 1;
            b.local_data()[i] = 1;
        }
    }

    a.distribute_from_master();
    b.distribute_from_master();

    substract(a, b).display("Somme a + b");

    std::cout << dot(b,b) <<std::endl;
    MPI_Finalize();

    return 0;
}
