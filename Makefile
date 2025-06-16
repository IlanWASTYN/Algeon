# Nom de l'exécutable
TARGET = algeon

# Fichiers source
SRC = test.cpp

# Compilateur MPI + OpenMP
CXX = mpicxx

# Flags de compilation
CXXFLAGS = -O2 -fopenmp -std=c++17 -Wall -Wextra

# Nombre de processus MPI par défaut
NP ?= 4

# ─────────── Cibles ───────────
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Exécution avec MPI
run: $(TARGET)
	mpirun -np $(NP) ./$(TARGET)

# Nettoyage
clean:
	rm -f $(TARGET)