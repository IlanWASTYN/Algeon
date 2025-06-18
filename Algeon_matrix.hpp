#ifndef ALGEON_MATRIX_HPP
#define ALGEON_MATRIX_HPP

#pragma once

#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cassert>
#include <iostream>

#include "mpi_type.hpp"

// ─────────────────────────────────────────────
// Algeon - Bibliothèque de matrices distribuées
// Auteur : WASTYN Ilan
// Projet : Algeon (MPI + OpenMP)
// ─────────────────────────────────────────────

#include <iostream>
#include <vector>
#include <stdexcept>

template <typename T>
class ParallelMatrix
{
private:
    //
    // ─── VARIABLES MEMBRE ──────────────────────────────────────────────────
    //
    MPI_Comm comm_;
    int rank_, size_;
    std::size_t global_size_rows_, global_size_cols_, local_size_rows_, local_size_cols_, offset_rows_, offset_cols_;
    std::vector<std::vector<T>> data_;

public:
    //
    // ─── CONSTRUCTEUR ──────────────────────────────────────────────────────
    //
    ParallelMatrix(MPI_Comm comm, size_t global_size_rows, size_t global_size_cols, double initial = 0.0)
        : comm_(comm), global_size_rows_(global_size_rows), global_size_cols_(global_size_cols), data_(global_size_rows, std::vector<double>(global_size_cols, initial))
    {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);

        if (rank_ == 0)
        {
            // Master : pas de données locales
            local_size_rows_ = 0;
            local_size_cols_ = 0;
            offset_rows_ = 0;
            offset_cols_ = 0;
        }

        else
        {
            std::size_t num_workers = size_ - 1;

            if (global_size_rows_ % num_workers != 0)
            {
                std::cerr << "Veuillez sélectionner un nombre de processeur tel que nb_proc - 1 divise la taille de votre vecteur\n";
                MPI_Finalize();
            }
            else
            {
                // Distribution équitable entre workers uniquement
                local_size_rows_ = global_size_rows_ / num_workers;
                local_size_cols_ = global_size_cols_ / num_workers;
                offset_rows_ = (global_size_rows_ / num_workers) * (rank_ - 1) + std::min<std::size_t>(rank_ - 1, global_size_rows_ % num_workers);
                offset_cols_ = (global_size_cols_ / num_workers) * (rank_ - 1) + std::min<std::size_t>(rank_ - 1, global_size_cols_ % num_workers);

                data_.resize(local_size_rows_ * local_size_cols_);
            }
        }
    }

    //
    // ─── ACCESSEURS ────────────────────────────────────────────────────────
    //
    std::size_t global_size() const { return global_size_rows; }
    std::size_t global_size() const { return global_size_cols; }
    std::size_t local_size() const { return local_size_rows; }
    std::size_t local_size() const { return local_size_cols; }
    std::size_t offset() const { return offset_rows; }
    std::size_t offset() const { return offset_cols; }
    //
    // ─── ACCES AUX DONNEES LOCALES ─────────────────────────────────────────
    //
    const std::vector<T> &local_data() const { return data_; }
    std::vector<T> &local_data() { return data_; }

    //
    // ─── OPERATEURS DE BASE ────────────────────────────────────────────────
    //

    // Proxy représentant une ligne
    class RowProxy
    {
    private:
        std::vector<T> &row;
        size_t cols;

    public:
        RowProxy(std::vector<T> &row, size_t cols) : row(row), cols(cols) {}

        T &operator[](size_t col)
        {
            if (col >= cols)
                throw std::out_of_range("Column index out of range");
            return row[col];
        }

        const T &operator[](size_t col) const
        {
            if (col >= cols)
                throw std::out_of_range("Column index out of range");
            return row[col];
        }
    };

    // Retourne une ligne (modifiable)
    RowProxy operator[](size_t row)
    {
        if (row >= global_size_rows_)
            throw std::out_of_range("Row index out of range");
        return RowProxy(data_[row], global_size_cols_);
    }

    // Retourne une ligne (const version)
    const RowProxy operator[](size_t row) const
    {
        if (row >= global_size_rows_)
            throw std::out_of_range("Row index out of range");
        return RowProxy(const_cast<std::vector<T> &>(data_[row]), global_size_cols_);
    }

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

    //
    // ─── AFFICHAGE ─────────────────────────────────────────
    //
    void display(const std::string &label = "Matrice") const
    {
        for (int r = 0; r < size_; r++)
        {
            MPI_Barrier(comm_); // Synchronisation
            if (rank_ == r)
            {
                std::cout << "[Proc " << rank_ << "] " << label << " local : ";
                for (std::size_t i = 0; i < global_size_rows_; i++)
                {
                    for (std::size_t j = 0; j < global_size_cols_; j++)
                    {
                        std::cout << data_[i][j] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        }
        MPI_Barrier(comm_); // Synchronisation finale
    }
};

#endif