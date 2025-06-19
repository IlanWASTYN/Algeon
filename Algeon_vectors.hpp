#pragma once

#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cassert>
#include <iostream>

#include "mpi_type.hpp"

// ─────────────────────────────────────────────
// Algeon - Bibliothèque de vecteurs distribués
// Auteur : WASTYN Ilan
// Projet : Algeon (MPI + OpenMP)
// ─────────────────────────────────────────────

template <typename T>
class ParallelVector
{
private:
    //
    // ─── VARIABLES MEMBRE ──────────────────────────────────────────────────
    //
    MPI_Comm comm_;
    int rank_, size_;
    std::size_t capacity_, global_size_, local_size_;
    std::vector<T> data_;

public:
    //
    // ─── CONSTRUCTEUR ──────────────────────────────────────────────────────
    //
    ParallelVector(MPI_Comm comm, std::size_t global_size)
        : comm_(comm), global_size_(global_size)
    {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);

        if (rank_ == 0)
        {
            // Master : pas de données locales
            local_size_ = 0;
            data_.resize(global_size_);
        }

        else
        {
            std::size_t num_workers = size_ - 1;

            if (global_size % num_workers != 0)
            {
                std::cerr << "Veuillez sélectionner un nombre de processeur tel que nb_proc - 1 divise la taille de votre vecteur\n";
            }
            else
            {
                // Distribution équitable entre workers uniquement
                local_size_ = global_size_;

                data_.resize(local_size_);
            }
        }
    }

    //
    // ─── ACCESSEURS ────────────────────────────────────────────────────────
    //
    std::size_t global_size() const { return global_size_; }
    std::size_t local_size() const { return local_size_; }
    const std::vector<T> &data() const { return data_; }
    MPI_Comm comm() const { return comm_; }

    //
    // ─── ACCES AUX DONNEES LOCALES ─────────────────────────────────────────
    //
    const std::vector<T> &local_data() const { return data_; }
    std::vector<T> &local_data() { return data_; }

    //
    // ─── OPERATEURS DE BASE ────────────────────────────────────────────────
    //
    T &operator[](std::size_t i) { return data_.at(i); }
    const T &operator[](std::size_t i) const { return data_.at(i); }

    ParallelVector<T> &operator=(const ParallelVector<T> &other)
    {
        // Protéger contre l'auto-assignment
        if (this == &other)
            return *this;

        // Copier les attributs de base
        comm_ = other.comm_;
        rank_ = other.rank_;
        size_ = other.size_;
        global_size_ = other.global_size_;
        local_size_ = other.local_size_;
        capacity_ = other.capacity_; // si tu l'utilises

        // Copier les données locales uniquement
        data_.resize(other.data_.size());

#pragma omp parallel for
        for (std::size_t i = 0; i < other.data_.size(); ++i)
            data_[i] = other.data_[i];

        return *this;
    }

    //
    // ─── ENVOIS ENTRE PROCESSUS ────────────────────────────────────────────────
    //
    void distribute_from_master()
    {
        std::size_t chunk_size = global_size_ / (size_ - 1); // Taille de chaque morceau envoyé

        if (rank_ == 0)
        {
            // Assure que le vecteur data_ du master contient bien toutes les données
            assert(data_.size() == global_size_);

            for (int r = 1; r < size_; r++)
            {
                const T *send_ptr = data_.data() + (r - 1) * chunk_size;

                MPI_Send(send_ptr, chunk_size, mpi_type<T>(), r, 0, comm_);
            }
        }
        else
        {
            std::size_t offset = (rank_ - 1) * chunk_size;
            MPI_Recv(data_.data() + offset, chunk_size, mpi_type<T>(), 0, 0, comm_, MPI_STATUS_IGNORE);
        }
    }

    void gather_to_master()
    {
        std::size_t chunk_size = global_size_ / (size_ - 1); // Taille de chaque morceau envoyé

        if (rank_ == 0)
        {
            // Assure que le vecteur data_ du master contient bien toutes les données
            assert(data_.size() == global_size_);

            for (int r = 1; r < size_; r++)
            {
                std::size_t recv_size = (r == size_ - 1) ? (global_size_ - (size_ - 2) * chunk_size) : chunk_size;
                T *recv_ptr = data_.data() + (r - 1) * chunk_size;
                MPI_Recv(recv_ptr, recv_size, mpi_type<T>(), r, 0, comm_, MPI_STATUS_IGNORE);
            }
        }
        else
        {
            std::size_t offset = (rank_ - 1) * chunk_size;
            MPI_Send(data_.data() + offset, chunk_size, mpi_type<T>(), 0, 0, comm_);
        }
    }

    //
    // ─── RESIZE ─────────────────────────────────────────
    //
    void resize(size_t new_capacity_)
    {
        double *new_data_ = new double[new_capacity_];
        for (size_t i = 0; i < global_size_; ++i)
            new_data_[i] = data_[i];
        delete[] data_;
        data_ = new_data_;
        capacity_ = new_capacity_;
    }

    //
    // ─── PUSHBACK ─────────────────────────────────────────
    //
    void push_back(int target_rank_, const T &value_)
    {
        if (rank_ == target_rank_)
        {
            data_.push_back(value_);
            local_size_ = data_.size();
        }
    }

    //
    // ─── AFFICHAGE ─────────────────────────────────────────
    //
    void display(const std::string &label = "Vecteur") const
    {
        for (int r = 0; r < size_; r++)
        {
            MPI_Barrier(comm_); // Synchronisation
            if (rank_ == r)
            {
                if (rank_ == 0)
                {
                    std::cout << "[Proc " << rank_ << "] " << label << " global : ";
                    for (std::size_t i = 0; i < global_size_; ++i)
                    {
                        std::cout << data_[i] << " ";
                    }
                    std::cout << std::endl;
                }
                else
                {
                    std::cout << "[Proc " << rank_ << "] " << label << " local : ";
                    for (std::size_t i = 0; i < local_size_; ++i)
                    {
                        std::cout << data_[i] << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }
        MPI_Barrier(comm_); // Synchronisation finale
    }
};

//
// ─── SOMME DISTRIBUEE ─────────────────────────────────────────────────
//
template <typename T>
ParallelVector<T> add(const ParallelVector<T> &a, const ParallelVector<T> &b)
{
    assert(a.global_size() == b.global_size());

    MPI_Comm comm = a.comm(); // Récupère le communicateur commun
    int rank;
    MPI_Comm_rank(comm, &rank);

    std::size_t global_size = a.global_size();
    ParallelVector<T> result(comm, global_size);

    if (rank != 0)
    {
        assert(a.local_size() == b.local_size());

#pragma omp parallel for
        for (std::size_t i = 0; i < a.local_size(); ++i)
            result.local_data()[i] = a.local_data()[i] + b.local_data()[i];
    }

    result.gather_to_master();

    return result;
}

//
// ─── SOUSTRACTION DISTRIBUEE ─────────────────────────────────────────────────
//
template <typename T>
ParallelVector<T> substract(const ParallelVector<T> &a, const ParallelVector<T> &b)
{
    assert(a.global_size() == b.global_size());

    MPI_Comm comm = a.comm(); // Récupère le communicateur commun
    int rank;
    MPI_Comm_rank(comm, &rank);

    std::size_t global_size = a.global_size();
    ParallelVector<T> result(comm, global_size);

    if (rank != 0)
    {
        assert(a.local_size() == b.local_size());

#pragma omp parallel for
        for (std::size_t i = 0; i < a.local_size(); ++i)
            result.local_data()[i] = a.local_data()[i] - b.local_data()[i];
    }

    result.gather_to_master(); // Version interne sans paramètre

    return result;
}

//
// ─── PRODUIT SCALAIRE DISTRIBUEE ────────────────────────────────────────
//
template <typename T>
T dot(const ParallelVector<T> &a, const ParallelVector<T> &b)
{
    assert(a.global_size() == b.global_size());
    assert(a.local_size() == b.local_size());

    T local_dot = 0;

#pragma omp parallel for reduction(+ : local_dot)
    for (std::size_t i = 0; i < a.local_size(); ++i)
        local_dot += a.data()[i] * b.data()[i];

    T global_dot = 0;
    MPI_Reduce(&local_dot, &global_dot, 1, mpi_type<T>(), MPI_SUM, 0, a.comm());

    return global_dot;
}

//
// ─── NORME DISTRIBUEE ────────────────────────────────────────
//
/*template<typename T>
T norm(const ParallelVector<T> &a)
{
    T local_norm = 0.0;

    if (a.rank_ != 0)
    {
#pragma omp parallel for reduction(+ : local_dot)
        for (std::size_t i = 0; i < a.local_size_; ++i)
            a.local_norm += a.data_[i] * a.data_[i];
    }

    T global_norm = 0.0;
    MPI_Reduce(&local_norm, &global_norm, 1, mpi_type<T>(), MPI_SUM, 0, a.comm_);
    global_norm = sqrt(global_norm);
    return global_norm; // seulement significatif sur rank 0
}*/

//
// ─── PREPARE VECTEUR POUR MULTIPLICATION ────────────────────────────────────────
//

template <typename T>
ParallelVector<T> prepare(ParallelVector<T> x)
{

    MPI_Comm comm = x.comm();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::size_t global_size = x.global_size();
    ParallelVector<T> result(comm, global_size);

    result = x;

    int left = rank - 1;
    int right = rank + 1;

    std::size_t chunk_size = global_size / (size - 1);
    std::size_t data_ptr = (rank - 1) * chunk_size;

    if (rank == 1)
    {
        MPI_Send(&result[chunk_size - 1], 1, mpi_type<T>(), right, 0, comm);
        MPI_Recv(&result[chunk_size], 1, mpi_type<T>(), right, 0, comm, MPI_STATUS_IGNORE);
    }
    else if (rank == size - 1)
    {
        MPI_Recv(&result[global_size - chunk_size - 1], 1, mpi_type<T>(), left, 0, comm, MPI_STATUS_IGNORE);
        MPI_Send(&result[global_size - chunk_size], 1, mpi_type<T>(), left, 0, comm);
    }
    else if (rank != 0)
    {
        MPI_Recv(&result[data_ptr - 1], 1, mpi_type<T>(), left, 0, comm, MPI_STATUS_IGNORE);
        MPI_Send(&result[data_ptr], 1, mpi_type<T>(), left, 0, comm);

        MPI_Send(&result[data_ptr + 1], 1, mpi_type<T>(), right, 0, comm);
        MPI_Recv(&result[data_ptr + 2], 1, mpi_type<T>(), right, 0, comm, MPI_STATUS_IGNORE);
    }

    return result;
}