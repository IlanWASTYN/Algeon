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
    std::size_t global_size_, local_size_, offset_;
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
            offset_ = 0;
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
                local_size_ = global_size_ / num_workers + ((rank_ - 1) < (global_size_ % num_workers) ? 1 : 0);
                offset_ = (global_size_ / num_workers) * (rank_ - 1) + std::min<std::size_t>(rank_ - 1, global_size_ % num_workers);

                data_.resize(local_size_);
            }
        }
    }

    //
    // ─── ACCESSEURS ────────────────────────────────────────────────────────
    //
    std::size_t global_size() const { return global_size_; }
    std::size_t local_size() const { return local_size_; }
    std::size_t offset() const { return offset_; }

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
        if (other.size_ > size_)
        {
            delete[] data_;
            data_ = new T[other.size_];
        }
#pragma omp parallel for
        for (std::size_t i = 0; i < other.size_; i++)
        {
            data_[i] = other.data_[i];
        }
        size_ = other.size_;
        return *this;
    }
    void fill(const T &value)
    {
        if (rank_ == 0)
            return;

#pragma omp parallel for
        for (std::size_t i = 0; i < local_size_; i++)
            data_[i] = value;
    }

    //
    // ─── ENVOIS ENTRE PROCESSUS ────────────────────────────────────────────────
    //
    void distribute_from_master(const std::vector<T> &global_data = {})
    {
        if (rank_ == 0)
        {
            assert(global_data.size() == global_size_);
            for (int r = 1; r < size_; ++r)
            {
                std::size_t send_size = global_size_ / (size_ - 1) + ((r - 1) < (global_size_ % (size_ - 1)) ? 1 : 0);
                std::size_t offset = (global_size_ / (size_ - 1)) * (r - 1) + std::min<std::size_t>(r - 1, global_size_ % (size_ - 1));
                MPI_Send(global_data.data() + offset, send_size, mpi_type<T>(), r, 0, comm_);
            }
        }
        else
        {
            MPI_Recv(data_.data(), local_size_, mpi_type<T>(), 0, 0, comm_, MPI_STATUS_IGNORE);
        }
    }

    void gather_to_master(std::vector<T> &global_data) const
    {
        if (rank_ == 0)
        {
            global_data.resize(global_size_);
            for (int r = 1; r < size_; ++r)
            {
                std::size_t recv_size = global_size_ / (size_ - 1) + ((r - 1) < (global_size_ % (size_ - 1)) ? 1 : 0);
                std::size_t offset = (global_size_ / (size_ - 1)) * (r - 1) + std::min<std::size_t>(r - 1, global_size_ % (size_ - 1));
                MPI_Recv(global_data.data() + offset, recv_size, mpi_type<T>(), r, 1, comm_, MPI_STATUS_IGNORE);
            }
        }
        else
        {
            MPI_Send(data_.data(), local_size_, mpi_type<T>(), 0, 1, comm_);
        }
    }

    //
    // ─── OPERATEUR DE SOMME VECTORIELLE ─────────────────────────────────────────────────
    //
    ParallelVector<T> operator+(const ParallelVector<T> &other) const
    {
        assert(global_size_ == other.global_size_);
        ParallelVector<T> result(comm_, global_size_);
        if (rank_ != 0)
        {
            assert(local_size_ == other.local_size_);
#pragma omp parallel for
            for (std::size_t i = 0; i < local_size_; ++i)
                result.data_[i] = data_[i] + other.data_[i];
        }

        return result;
    }

    //
    // ─── OPERATEUR DE SOUSTRACTION VECTORIELLE ──────────────────────────────────────────
    //
    ParallelVector<T> operator-(const ParallelVector<T> &other) const
    {
        assert(global_size_ == other.global_size_);
        ParallelVector<T> result(comm_, global_size_);

        if (rank_ != 0)
        {
            assert(local_size_ == other.local_size_);
#pragma omp parallel for
            for (std::size_t i = 0; i < local_size_; ++i)
                result.data_[i] = data_[i] - other.data_[i];
        }

        return result;
    }

    //
    // ─── AFFICHAGE ─────────────────────────────────────────
    //
    void print_local_data_per_proc(const std::string &label = "Vecteur") const
    {
        for (int r = 0; r < size_; r++)
        {
            MPI_Barrier(comm_); // Synchronisation
            if (rank_ == r)
            {
                std::cout << "[Proc " << rank_ << "] " << label << " local : ";
                for (std::size_t i = 0; i < local_size_; ++i)
                {
                    std::cout << data_[i] << " ";
                }
                std::cout << std::endl;
            }
        }
        MPI_Barrier(comm_); // Synchronisation finale
    }
};

//
// ─── SOMME DISTRIBUEE ─────────────────────────────────────────────────
//
/*ParallelVector<T> Add(const ParallelVector<T> &a, const ParallelVector<T> &b) const
{
    assert(global_size_ == other.global_size_);
    ParallelVector<T> result(comm_, global_size_);
    if (rank_ != 0)
    {
        assert(local_size_ == other.local_size_);
#pragma omp parallel for
        for (std::size_t i = 0; i < local_size_; ++i)
            result.data_[i] = data_[i] + other.data_[i];
    }

    return result;
}

//
// ─── SOUSTRACTION DISTRIBUEE ─────────────────────────────────────────────────
//
ParallelVector<T> Substract(const ParallelVector<T> &other) const
{
    assert(global_size_ == other.global_size_);
    ParallelVector<T> result(comm_, global_size_);
    if (rank_ != 0)
    {
        assert(local_size_ == other.local_size_);
#pragma omp parallel for
        for (std::size_t i = 0; i < local_size_; ++i)
            result.data_[i] = data_[i] + other.data_[i];
    }

    return result;
}

//
// ─── PRODUIT SCALAIRE DISTRIBUEE ────────────────────────────────────────
//
T dot(const ParallelVector<T> &other) const
{
    assert(local_size_ == other.local_size_);

    T local_dot = 0.0;

    if (rank_ != 0)
    {
#pragma omp parallel for reduction(+ : local_dot)
        for (std::size_t i = 0; i < local_size_; ++i)
            local_dot += data_[i] * other.data_[i];
    }

    T global_dot = 0.0;
    MPI_Reduce(&local_dot, &global_dot, 1, mpi_type<T>(), MPI_SUM, 0, comm_);
    return global_dot; // seulement significatif sur rank 0
}*/

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