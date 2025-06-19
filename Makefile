# Nom de l'exécutable
EXEC = test2

# Fichiers source
SRC = test2.cpp

# Compilateur et options
CXX = mpic++
CXXFLAGS = -O2 -Wall -fopenmp

# Nombre de processus MPI (par défaut 4)
NP ?= 4

# Règle principale
all: $(EXEC)

$(EXEC): $(SRC) Algeon_matrix.hpp
	$(CXX) $(CXXFLAGS) $(SRC) -o $(EXEC)

# Exécution avec NP processus
run: $(EXEC)
	mpirun -np $(NP) ./$(EXEC)

# Nettoyage
clean:
	rm -f $(EXEC)