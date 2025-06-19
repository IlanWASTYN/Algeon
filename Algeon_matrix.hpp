#ifndef ALGEON_MATRIX_HPP
#define ALGEON_MATRIX_HPP

#pragma once

#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cassert>
#include <iostream>
#include <stdexcept>

#include "Algeon_vectors.hpp"
#include "mpi_type.hpp"

// ─────────────────────────────────────────────
// Algeon - Bibliothèque de matrices distribuées
// Auteur : WASTYN Ilan
// Projet : Algeon (MPI + OpenMP)
// ─────────────────────────────────────────────

template <typename T>
struct CSRMatrixLocal
{
    std::vector<std::size_t> row_ptr; // taille local_size_row + 1
    std::vector<std::size_t> col_idx; // indices globaux des colonnes
    std::vector<T> values;            // valeurs non-nulles
};

template <typename T>
class ParallelMatrix
{
private:
    //
    // ─── VARIABLES MEMBRE ──────────────────────────────────────────────────
    //
    MPI_Comm comm_;
    int rank_, size_;
    std::size_t capacity_, global_size_rows_, global_size_cols_, local_size_rows_, local_size_cols_, offset_rows_, offset_cols_;
    std::vector<std::vector<T>> data_;           // Stockage ligne par ligne
    std::vector<std::size_t> distribution_rows_; // Ex. : {0, 3, 6, 10} pour 3 rangs

public:
    //
    // ─── CONSTRUCTEUR ──────────────────────────────────────────────────────
    //
    ParallelMatrix(MPI_Comm comm, size_t global_size_rows, size_t global_size_cols, T initial = T())
        : comm_(comm), global_size_rows_(global_size_rows), global_size_cols_(global_size_cols)
    {
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);

        std::size_t num_workers = size_ - 1;

        if (rank_ == 0)
        {
            // Master : stocke la matrice entière
            local_size_rows_ = 0;
            local_size_cols_ = 0;

            data_.resize(global_size_rows_, std::vector<T>(global_size_cols_, initial));
        }
        else
        {
            // Vérifie que les lignes sont divisibles par le nombre de workers
            if (global_size_rows_ % num_workers != 0)
            {
                if (rank_ == 1)
                    std::cerr << "Veuillez sélectionner un nombre de processeurs tel que nb_proc - 1 divise la taille de votre matrice (lignes).\n";
                MPI_Abort(comm_, 1);
            }

            // Chaque worker reçoit un bloc de lignes
            local_size_rows_ = global_size_rows_ / num_workers;
            local_size_cols_ = global_size_cols_;

            data_.resize(local_size_rows_, std::vector<T>(local_size_cols_, initial));
        }
    }

    //
    // ─── OFFSETS ────────────────────────────────────────────────────────
    //
    std::size_t global_row_offset() const
    {
        // Pour le processus maître (rank 0), il n’y a pas de partie locale
        if (rank_ == 0)
            return 0;

        // Sinon, calcul du décalage des lignes en fonction du rang et de la taille locale
        int num_workers = size_ - 1;
        return (rank_ - 1) * (global_size_rows_ / num_workers);
    }

    std::size_t global_col_offset() const
    {
        // Ici, la colonne locale commence toujours à 0, car la matrice est découpée par lignes uniquement
        return 0;
    }

    //
    // ─── ACCESSEURS ────────────────────────────────────────────────────────
    //
    std::size_t global_size_row() const { return global_size_rows_; }
    std::size_t global_size_col() const { return global_size_cols_; }
    std::size_t local_size_row() const { return local_size_rows_; }
    std::size_t local_size_col() const { return local_size_cols_; }
    MPI_Comm comm() const { return comm_; }

    //
    // ─── ACCES AUX DONNEES LOCALES ─────────────────────────────────────────
    //
    const std::vector<std::vector<T>> &local_data() const { return data_; }
    std::vector<std::vector<T>> &local_data() { return data_; }

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
        if (row >= data_.size())
            throw std::out_of_range("Row index out of range");
        return RowProxy(data_[row], global_size_cols_);
    }

    // Retourne une ligne (const version)
    const RowProxy operator[](size_t row) const
    {
        if (row >= data_.size())
            throw std::out_of_range("Row index out of range");
        return RowProxy(const_cast<std::vector<T> &>(data_[row]), global_size_cols_);
    }

    ParallelMatrix<T> &operator=(const ParallelMatrix<T> &other)
    {
        // Protéger contre l'auto-assignment
        if (this == &other)
            return *this;

        // Copier les attributs de base
        comm_ = other.comm_;
        rank_ = other.rank_;
        size_ = other.size_;
        global_size_rows_ = other.global_size_rows_;
        global_size_cols_ = other.global_size_cols_;
        local_size_rows_ = other.local_size_rows_;
        local_size_cols_ = other.local_size_cols_;
        capacity_ = other.capacity_;

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
        std::size_t num_workers = size_ - 1;
        std::size_t rows_per_worker = global_size_rows_ / num_workers;

        if (rank_ == 0)
        {
            // Envoi de chaque bloc ligne aux workers
            for (int r = 1; r < size_; ++r)
            {
                for (std::size_t i = 0; i < rows_per_worker; ++i)
                {
                    std::size_t global_row = (r - 1) * rows_per_worker + i;
                    MPI_Send(data_[global_row].data(), global_size_cols_, mpi_type<T>(), r, 0, comm_);
                }
            }
        }
        else
        {
            // Réception bloc ligne par ligne
            for (std::size_t i = 0; i < local_size_rows_; ++i)
            {
                MPI_Recv(data_[i].data(), global_size_cols_, mpi_type<T>(), 0, 0, comm_, MPI_STATUS_IGNORE);
            }
        }
    }

    void gather_to_master()
    {
        std::size_t num_workers = size_ - 1;
        std::size_t rows_per_worker = global_size_rows_ / num_workers;

        if (rank_ == 0)
        {
            // Le master reçoit les blocs ligne par ligne de chaque worker
            for (int r = 1; r < size_; ++r)
            {
                for (std::size_t i = 0; i < rows_per_worker; ++i)
                {
                    std::size_t global_row = (r - 1) * rows_per_worker + i;
                    MPI_Recv(data_[global_row].data(), global_size_cols_, mpi_type<T>(), r, 0, comm_, MPI_STATUS_IGNORE);
                }
            }
        }
        else
        {
            // Chaque worker envoie ses lignes une à une
            for (std::size_t i = 0; i < local_size_rows_; ++i)
            {
                MPI_Send(data_[i].data(), global_size_cols_, mpi_type<T>(), 0, 0, comm_);
            }
        }
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
                std::cout << "[Proc " << rank_ << "] " << label << " :\n";
                for (std::size_t i = 0; i < data_.size(); ++i)
                {
                    for (std::size_t j = 0; j < data_[i].size(); ++j)
                    {
                        std::cout << data_[i][j] << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << std::endl;
            }
        }
        MPI_Barrier(comm_); // Synchronisation finale
    }

    //
    // ─── Format CSR ─────────────────────────────────────────
    //

    CSRMatrixLocal<T> extract_local_csr() const
    {
        CSRMatrixLocal<T> csr;

        int rank;
        MPI_Comm_rank(comm_, &rank);

        if (rank == 0)
        {
            // 1. Réservation initiale
            csr.row_ptr.resize(global_size_rows_ + 1, 0);

            // 2. Comptage des non-nuls par ligne
#pragma omp parallel for
            for (std::size_t i = 0; i < global_size_rows_; ++i)
            {
                std::size_t count = 0;
                for (std::size_t j = 0; j < global_size_cols_; ++j)
                {
                    if (data_[i][j] != T(0))
                        ++count;
                }
                csr.row_ptr[i + 1] = count;
            }

            // 3. Prefix sum
            for (std::size_t i = 1; i <= global_size_rows_; ++i)
            {
                csr.row_ptr[i] += csr.row_ptr[i - 1];
            }

            std::size_t nnz = csr.row_ptr[global_size_rows_];

            // 4. Allocation
            csr.col_idx.resize(nnz);
            csr.values.resize(nnz);

            // 5. Remplissage
#pragma omp parallel for
            for (std::size_t i = 0; i < global_size_rows_; ++i)
            {
                std::size_t start = csr.row_ptr[i];
                std::size_t idx = start;
                for (std::size_t j = 0; j < global_size_cols_; ++j)
                {
                    if (data_[i][j] != T(0))
                    {
                        csr.col_idx[idx] = j;
                        csr.values[idx] = data_[i][j];
                        ++idx;
                    }
                }
            }

            // 6. Envoi aux autres processus
            for (int r = 1; r < size_; ++r)
            {
                std::size_t row_ptr_local_size;
                std::size_t col_idx_size = csr.col_idx.size();
                std::size_t values_size = csr.values.size();

                MPI_Recv(&row_ptr_local_size, 1, mpi_type<T>(), r, 0, comm_, MPI_STATUS_IGNORE);
                std::size_t offset = (r - 1) * (row_ptr_local_size - 1);
                MPI_Send(csr.row_ptr.data() + offset, row_ptr_local_size, mpi_type<T>(), r, 1, comm_);

                MPI_Send(&col_idx_size, 1, mpi_type<T>(), r, 2, comm_);
                MPI_Send(csr.col_idx.data(), col_idx_size, mpi_type<T>(), r, 3, comm_);

                MPI_Send(&values_size, 1, mpi_type<T>(), r, 4, comm_);
                MPI_Send(csr.values.data(), values_size, mpi_type<T>(), r, 5, comm_);
            }
        }
        else
        {
            // Réception du CSR complet depuis le processus 0

            std::size_t row_ptr_local_size = local_size_rows_ + 1;
            std::size_t col_idx_size, values_size;

            MPI_Send(&row_ptr_local_size, 1, mpi_type<T>(), 0, 0, comm_);
            csr.row_ptr.resize(row_ptr_local_size);
            MPI_Recv(csr.row_ptr.data(), row_ptr_local_size, mpi_type<T>(), 0, 1, comm_, MPI_STATUS_IGNORE);

            MPI_Recv(&col_idx_size, 1, mpi_type<T>(), 0, 2, comm_, MPI_STATUS_IGNORE);
            csr.col_idx.resize(col_idx_size);
            MPI_Recv(csr.col_idx.data(), col_idx_size, mpi_type<T>(), 0, 3, comm_, MPI_STATUS_IGNORE);

            MPI_Recv(&values_size, 1, mpi_type<T>(), 0, 4, comm_, MPI_STATUS_IGNORE);
            csr.values.resize(values_size);
            MPI_Recv(csr.values.data(), values_size, mpi_type<T>(), 0, 5, comm_, MPI_STATUS_IGNORE);
        }

        return csr;
    }
};

//
// ─── SOMME DISTRIBUEE ─────────────────────────────────────────────────
//
template <typename T>
ParallelMatrix<T> add(const ParallelMatrix<T> &a, const ParallelMatrix<T> &b)
{
    assert(a.global_size_row() == b.global_size_row());
    assert(a.global_size_col() == b.global_size_col());

    MPI_Comm comm = a.comm(); // Récupère le communicateur MPI
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Création de la matrice résultat (initialisée à 0)
    ParallelMatrix<T> result(comm, a.global_size_row(), a.global_size_col(), T());

    if (rank != 0)
    {
        assert(a.local_size_row() == b.local_size_row());
        assert(a.local_size_col() == b.local_size_col());

#pragma omp parallel for collapse(2)
        for (std::size_t i = 0; i < a.local_size_row(); ++i)
        {
            for (std::size_t j = 0; j < a.local_size_col(); ++j)
            {
                result.local_data()[i][j] = a.local_data()[i][j] + b.local_data()[i][j];
            }
        }
    }

    // On rassemble les blocs sur le processus maître
    result.gather_to_master();

    return result;
}

//
// ─── SOUSTRACTION DISTRIBUEE ─────────────────────────────────────────────────
//
template <typename T>
ParallelMatrix<T> substract(const ParallelMatrix<T> &a, const ParallelMatrix<T> &b)
{
    assert(a.global_size_row() == b.global_size_row());
    assert(a.global_size_col() == b.global_size_col());

    MPI_Comm comm = a.comm(); // Récupère le communicateur MPI
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Création de la matrice résultat (initialisée à 0)
    ParallelMatrix<T> result(comm, a.global_size_row(), a.global_size_col(), T());

    if (rank != 0)
    {
        assert(a.local_size_row() == b.local_size_row());
        assert(a.local_size_col() == b.local_size_col());

#pragma omp parallel for collapse(2)
        for (std::size_t i = 0; i < a.local_size_row(); ++i)
        {
            for (std::size_t j = 0; j < a.local_size_col(); ++j)
            {
                result.local_data()[i][j] = a.local_data()[i][j] - b.local_data()[i][j];
            }
        }
    }

    // On rassemble les blocs sur le processus maître
    result.gather_to_master();

    return result;
}

//
// ─── PRODUIT MATRICE VECTEUR DISTRIBUEE ─────────────────────────────────────────────────
//
template <typename T>
ParallelVector<T> multiply_csr(const ParallelMatrix<T> &A, ParallelVector<T> &x)
{
    assert(A.global_size_col() == x.global_size());

    MPI_Comm comm = x.comm();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::size_t global_rows = A.global_size_row();
    ParallelVector<T> y(comm, global_rows);

    ParallelVector<double> x_prep = prepare(x);
    CSRMatrixLocal<T> csr = A.extract_local_csr();

    std::size_t chunk_size = global_rows / (size - 1);
    std::size_t offset = (rank - 1) * chunk_size;

    // ──────────────────────────────────────────────
    // Produit local CSR × x avec affichage de y[i]
    // ──────────────────────────────────────────────
    if (rank != 0)
    {
        const auto &values = csr.values.data();
        const auto &col_idx = csr.col_idx.data();
        const auto &row_ptr = csr.row_ptr.data();
        const std::size_t row_size = csr.row_ptr.size();
        const std::vector<T> &x_data = x_prep.data();

        for (std::size_t i = 0; i < row_size - 1; i++)
        {
            y[i] = 0; // Sécurité
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
            {
                y[i + offset] += values[j] * x_data[col_idx[j]];
            }
        }
    }

    y.gather_to_master();
    return y;
}

//
// ─── TRANSPOSITION MATRICE DISTRIBUEER ─────────────────────────────────────────────────
//

template <typename T>
ParallelMatrix<T> transpose(const ParallelMatrix<T> &a)
{
    MPI_Comm comm = a.comm();
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::size_t global_rows = a.global_size_row();
    std::size_t global_cols = a.global_size_col();

    // Créer une nouvelle matrice transposée : dimensions inversées
    ParallelMatrix<T> result(comm, global_cols, global_rows, T());

    if (rank == 0)
    {
        const auto &a_data = a.local_data();
        auto &r_data = result.local_data();

// Transposition parallèle sur le maître
#pragma omp parallel for collapse(2)
        for (std::size_t i = 0; i < global_rows; ++i)
        {
            for (std::size_t j = 0; j < global_cols; ++j)
            {
                r_data[j][i] = a_data[i][j];
            }
        }
    }

    // Redistribuer la matrice transposée aux autres processus
    result.distribute_from_master();

    return result;
}

/*void transpose()
{
    std::vector<std::vector<T>> transposed(local_size_cols_, std::vector<T>(local_size_rows_));

#pragma omp parallel for collapse(2)
    for (std::size_t i = 0; i < local_size_rows_; ++i)
    {
        for (std::size_t j = 0; j < local_size_cols_; ++j)
        {
            transposed[j][i] = data_[i][j];
        }
    }

    data_ = std::move(transposed);

    // Mise à jour des dimensions locales (transposées)
    std::swap(local_size_rows_, local_size_cols_);
    std::swap(global_size_rows_, global_size_cols_);
}*/

//
// ─── INVERSION MATRICE FULL ─────────────────────────────────────────────────
//
template <typename T>
ParallelMatrix<T> invert(const ParallelMatrix<T> &a, int choosen_rank)
{
    MPI_Comm comm = a.comm();
    int rank;
    MPI_Comm_rank(comm, &rank);

    std::size_t n = a.global_size_row();
    assert(n == a.global_size_col()); // Matrice carrée uniquement

    ParallelMatrix<T> result(comm, n, n, T(0));

    if (rank == choosen_rank)
    {
        // Récupère toute la matrice (au cas où ce n'est pas encore fait)
        ParallelMatrix<T> full_matrix = a; // Copie complète sur le maître
        full_matrix.gather_to_master();

        // Construire l'identité
        std::vector<std::vector<T>> identity(n, std::vector<T>(n, T(0)));
        for (std::size_t i = 0; i < n; ++i)
            identity[i][i] = T(1);

        // Extraction des données
        std::vector<std::vector<T>> mat = full_matrix.local_data();

        // Méthode de Gauss-Jordan pour inversion
        for (std::size_t i = 0; i < n; ++i)
        {
            // Cherche un pivot non nul
            if (mat[i][i] == T(0))
            {
                std::size_t swap_row = i + 1;
                while (swap_row < n && mat[swap_row][i] == T(0))
                    ++swap_row;

                if (swap_row == n)
                    throw std::runtime_error("Matrice singulière : pas inversible");

                std::swap(mat[i], mat[swap_row]);
                std::swap(identity[i], identity[swap_row]);
            }

            // Normaliser la ligne
            T pivot = mat[i][i];
            for (std::size_t j = 0; j < n; ++j)
            {
                mat[i][j] /= pivot;
                identity[i][j] /= pivot;
            }

            // Éliminer les autres lignes
            for (std::size_t k = 0; k < n; ++k)
            {
                if (k == i)
                    continue;
                T factor = mat[k][i];
                for (std::size_t j = 0; j < n; ++j)
                {
                    mat[k][j] -= factor * mat[i][j];
                    identity[k][j] -= factor * identity[i][j];
                }
            }
        }

        // Injecter le résultat dans la matrice résultat
        result.local_data() = identity;
    }

    // Distribuer aux workers
    result.distribute_from_master();

    return result;
}

#endif
