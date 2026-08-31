#include <vector>
#include <fstream>
#include <iostream>
#include <random>
#include "particle.hpp"

/**
 * @brief Template function that writes on file the total number of particles
 *and the size of the area of the simulation and then the initial state of the
 *particles in a file.
 *
 * @tparam Dimension Number of dimensions of the simulation
 * @param particles Reference to the vector of particles that are to be printed
 *on the file
 * @param dim Dimension of the simulation area
 * @param fileName Name of the file in which the initial states of the particles
 *are going to be written
 * @param file Reference to the file in which the initial states of the
 *particles are going to be written
 * @param it Number of iterations
 **/

template <size_t Dimension>
void printInitialStateOnFile(const std::vector<Particle<Dimension>> &particles, int dim, const std::string &fileName, int it, int speedUp, size_t numFilesAndThreads)
{
    std::ofstream file(fileName);
    if (file.is_open())
    {
        //Scrivo ad inizio file
        file << particles.size() << std::endl;//Numero di particelle
        file << dim << std::endl;//Dimensione dell'area di simulazione
        file << Dimension << std::endl;//Numero di dimensioni della simulazione
        file << it / speedUp << std::endl;//Numero di iterazioni
        file << numFilesAndThreads << std::endl;//Numero di file e thread

        for (const Particle<Dimension> &p : particles)
            file << p.getRadius() << std::endl;
        for (const Particle<Dimension> &p : particles)
        {
            file << p.getId() << ",";
            const auto &pos = p.getPos();
            for (size_t i = 0; i < Dimension; ++i)
            {
                file << pos[i];
                if (i < Dimension - 1)
                    file << ",";
            }
            file << std::endl;
        }
    }
    else
    {
        std::cout << "Unable to open file";
    }
    file.close();
}

/**
 * @brief Template function that writes on file the particles' id and positions 
 * 
 * @tparam Dimension Number of dimensions of the simulation
 * @param particles Reference to the vector of particles that are to be written on the file
 * @param file Reference to the file in which the particles are going to be written
 * 
*/
template <size_t Dimension>
void writeParticlePositionsToFile(const std::vector<Particle<Dimension>> &particles, std::ostream &file)
{
    for (const auto &q : particles)
    {
        file << q.getId() << ",";
        const auto& pos = q.getPos();
        for (size_t i = 0; i < Dimension; ++i) {
            file << pos[i];
            if (i < Dimension - 1) {
                file << ",";
            }
        }
        file << '\n';
    }
}
